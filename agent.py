import io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["up", "down", "left", "right"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


# State dimensions for goal-relative encoding
LOCAL_WINDOW_SIZE = 5  # 5×5 window around agent
STATE_DIM = 2 + LOCAL_WINDOW_SIZE * LOCAL_WINDOW_SIZE  # Δx, Δy + 25 obstacle values = 27




# ---------------------------------------------------------------------------
# PPO Neural Network (Actor-Critic)
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """
    Actor-Critic MLP with shared backbone for goal-relative observations.

    Architecture:
        Input: (batch, STATE_DIM) - goal-relative feature vector
        Shared backbone:
            Linear(STATE_DIM, 128) -> ReLU
            Linear(128, 128) -> ReLU

        Actor Head: Linear(128, 4) -> action logits
        Critic Head: Linear(128, 1) -> state value
    """

    def __init__(self, input_dim: int = STATE_DIM, n_actions: int = 4):
        super().__init__()

        # Shared MLP backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Actor head: outputs logits for Categorical distribution
        self.actor = nn.Linear(128, n_actions)

        # Critic head: outputs scalar value estimate
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both heads.

        Returns:
            action_logits: (batch, 4) raw logits for actions
            value: (batch, 1) state value estimate
        """
        h = self.shared(x)
        action_logits = self.actor(h)
        value = self.critic(h)

        return action_logits, value

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action probabilities, log probs, entropy, and value.

        Args:
            x: (batch, STATE_DIM) goal-relative feature tensor
            action: (batch,) optional - if provided, compute log_prob for this action

        Returns:
            action: (batch,) sampled or provided action indices
            log_prob: (batch,) log probability of the action
            entropy: (batch,) entropy of the action distribution
            value: (batch,) state value estimate
        """
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


# ---------------------------------------------------------------------------
# PPO Rollout Buffer
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """
    Buffer for storing on-policy rollout data for PPO.

    Stores trajectories and computes advantages using GAE-Lambda.
    Unlike replay buffer, this is cleared after each policy update.
    """

    def __init__(self, buffer_size: int, state_dim: int = STATE_DIM):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.reset()

    def reset(self):
        """Clear all stored data."""
        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)

        # Computed after rollout is complete
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)

        self.ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        """Add a single transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def compute_gae(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        """
        Compute Generalized Advantage Estimation (GAE-Lambda).

        GAE formula (computed efficiently in reverse):
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        """
        n_steps = self.ptr
        last_gae = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            # TD error
            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )

            # GAE recursion
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        # Returns = advantages + values (for value function loss)
        self.returns[:n_steps] = self.advantages[:n_steps] + self.values[:n_steps]

    def get_batches(self, batch_size: int, device: torch.device):
        """Generate shuffled minibatches for PPO update."""
        n_steps = self.ptr
        indices = np.random.permutation(n_steps)

        for start in range(0, n_steps, batch_size):
            end = min(start + batch_size, n_steps)
            batch_indices = indices[start:end]

            yield {
                "states": torch.from_numpy(self.states[batch_indices]).to(device),
                "actions": torch.from_numpy(self.actions[batch_indices]).to(device),
                "old_log_probs": torch.from_numpy(self.log_probs[batch_indices]).to(
                    device
                ),
                "advantages": torch.from_numpy(self.advantages[batch_indices]).to(
                    device
                ),
                "returns": torch.from_numpy(self.returns[batch_indices]).to(device),
            }

    def __len__(self):
        return self.ptr


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------


class PPOAgent:
    """
    Proximal Policy Optimization agent using goal-relative observations.

    The state encoding forces goal-directed learning by providing only:
      - Relative position to goal (Δx, Δy)
      - Local 5×5 obstacle window around the agent

    1. On-policy: uses fresh trajectories, no replay buffer
    2. Actor-Critic: separate policy and value networks (shared backbone)
    3. Clipped objective: prevents catastrophic policy updates
    4. Entropy bonus: encourages exploration without epsilon decay
    """

    def __init__(self, height: int = 10, width: int = 10):
        self.height = height
        self.width = width
        self.device = torch.device("cpu")

        # ---- PPO Hyperparameters ----
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.lr = 3e-4
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        # ---- Training parameters ----
        self.rollout_length = 2048
        self.batch_size = 64
        self.n_epochs = 10

        # ---- Network (now using MLP for goal-relative features) ----
        self.network = ActorCritic(input_dim=STATE_DIM).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)

        # ---- Rollout buffer ----
        self.buffer = RolloutBuffer(self.rollout_length, state_dim=STATE_DIM)

        # ---- Step-level state ----
        self.prev_state_tensor: np.ndarray | None = None
        self.prev_action_idx: int | None = None
        self.prev_log_prob: float | None = None
        self.prev_value: float | None = None

        # ---- Counters ----
        self.total_steps = 0
        self.updates = 0
        self.total_episodes = 0

        # ---- Pseudo-epsilon for display (PPO uses entropy, not epsilon-greedy) ----
        self._pseudo_epsilon = 1.0

    @property
    def epsilon(self) -> float:
        """Pseudo-epsilon for display - PPO uses entropy regularization instead."""
        return self._pseudo_epsilon

    # ------------------------------------------------------------------
    # State encoding - goal-relative observations
    # ------------------------------------------------------------------

    def _extract_local_window(self, px: int, py: int, obstacles: set) -> np.ndarray:
        """
        Extract a 5×5 window of obstacle information centered on the agent.

        Out-of-bounds cells are treated as obstacles (walls).
        Returns a flattened array of 25 values.
        """
        half = LOCAL_WINDOW_SIZE // 2
        window = np.zeros((LOCAL_WINDOW_SIZE, LOCAL_WINDOW_SIZE), dtype=np.float32)

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                world_x = px + dx
                world_y = py + dy
                wy = dy + half
                wx = dx + half

                if world_x < 0 or world_x >= self.width or world_y < 0 or world_y >= self.height:
                    window[wy, wx] = 1.0
                elif (world_x, world_y) in obstacles:
                    window[wy, wx] = 1.0

        return window.flatten()

    def encode_state(self, position, target, obstacles) -> np.ndarray:
        """
        Build a goal-relative feature vector.

        Returns: (STATE_DIM,) array containing:
          - Δx: target_x - agent_x (normalized by grid size)
          - Δy: target_y - agent_y (normalized by grid size)
          - 5×5 local obstacle window (25 values)
        """
        px, py = position
        tx, ty = target

        max_dim = max(self.width, self.height)
        delta_x = (tx - px) / max_dim
        delta_y = (ty - py) / max_dim

        obs_set = {(o[0], o[1]) if isinstance(o, (list, tuple)) else o for o in obstacles}
        local_window = self._extract_local_window(px, py, obs_set)

        state = np.concatenate([
            np.array([delta_x, delta_y], dtype=np.float32),
            local_window
        ])

        return state

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, position, target, obstacles) -> str:
        """
        Sample action from the policy distribution.

        The policy learns to balance exploration via entropy regularization.
        """
        state_np = self.encode_state(position, target, obstacles)
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(self.device)  # (1, STATE_DIM)

        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_t)

        self.prev_log_prob = log_prob.item()
        self.prev_value = value.item()

        action_idx = int(action.item())
        return ACTIONS[action_idx]

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def update_memory(self, position, target, obstacles, action):
        """Remember (s, a) for the next learn() call."""
        self.prev_state_tensor = self.encode_state(position, target, obstacles)
        self.prev_action_idx = ACTION_TO_IDX[action]

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, position, target, obstacles, reward, done):
        """
        Store transition in rollout buffer. When buffer is full,
        compute advantages and run PPO update epochs.
        """
        if self.prev_state_tensor is None or self.prev_action_idx is None:
            return  # First call in episode

        if self.prev_log_prob is None or self.prev_value is None:
            return  # First call in episode

        # Add transition to buffer
        self.buffer.add(
            state=self.prev_state_tensor,
            action=self.prev_action_idx,
            log_prob=float(self.prev_log_prob),
            reward=reward,
            value=float(self.prev_value),
            done=done,
        )

        self.total_steps += 1

        # Check if buffer is full -> run PPO update
        if self.buffer.ptr >= self.rollout_length:
            self._ppo_update(position, target, obstacles)

        # Episode end -> clear step memory
        if done:
            self.prev_state_tensor = None
            self.prev_action_idx = None
            self.prev_log_prob = None
            self.prev_value = None
            self.total_episodes += 1
            # Decay pseudo-epsilon for display
            self._pseudo_epsilon = max(0.01, self._pseudo_epsilon * 0.995)

    def _ppo_update(self, current_position, current_target, current_obstacles):
        """
        Run PPO update after collecting rollout_length steps.

        1. Compute bootstrap value for last state
        2. Compute GAE advantages
        3. Run n_epochs of minibatch updates
        4. Clear buffer
        """
        # Bootstrap value for incomplete episode
        current_state = self.encode_state(
            current_position, current_target, current_obstacles
        )
        state_t = torch.from_numpy(current_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value = self.network.forward(state_t)
            last_value = last_value.item()

        # Compute advantages
        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        # Normalize advantages (crucial for stability)
        advantages = self.buffer.advantages[: self.buffer.ptr]
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        self.buffer.advantages[: self.buffer.ptr] = (advantages - adv_mean) / adv_std

        # Run multiple epochs of updates
        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size, self.device):
                self._update_step(batch)

        self.updates += 1
        self.buffer.reset()

    def _update_step(self, batch: dict):
        """
        Single PPO update step on a minibatch.

        Loss = L_clip + c1 * L_value + c2 * L_entropy
        """
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Get current policy outputs
        _, new_log_probs, entropy, values = self.network.get_action_and_value(
            states, actions
        )

        # ---- Policy loss (clipped surrogate objective) ----
        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        # ---- Value loss ----
        value_loss = nn.functional.mse_loss(values, returns)

        # ---- Entropy loss (negative because we want to maximize entropy) ----
        entropy_loss = -entropy.mean()

        # ---- Combined loss ----
        loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        # ---- Optimization step ----
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialize agent state for DB storage."""
        buf = io.BytesIO()
        data = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "updates": self.updates,
            "total_episodes": self.total_episodes,
            "pseudo_epsilon": self._pseudo_epsilon,
            "height": self.height,
            "width": self.width,
            "hyperparams": {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "lr": self.lr,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "rollout_length": self.rollout_length,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
            },
        }
        torch.save(data, buf)
        return buf.getvalue()

    def from_bytes(self, byte_data: bytes):
        """Restore agent state from DB bytes."""
        buf = io.BytesIO(byte_data)
        data = torch.load(buf, map_location=self.device, weights_only=False)

        self.network.load_state_dict(data["network_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.total_steps = data.get("total_steps", 0)
        self.updates = data.get("updates", 0)
        self.total_episodes = data.get("total_episodes", 0)
        self._pseudo_epsilon = data.get("pseudo_epsilon", 0.01)

        # Restore hyperparams if present
        if "hyperparams" in data:
            hp = data["hyperparams"]
            self.gamma = hp.get("gamma", self.gamma)
            self.gae_lambda = hp.get("gae_lambda", self.gae_lambda)
            self.clip_epsilon = hp.get("clip_epsilon", self.clip_epsilon)
            self.value_coef = hp.get("value_coef", self.value_coef)
            self.entropy_coef = hp.get("entropy_coef", self.entropy_coef)

        # Clear step-level memory
        self.prev_state_tensor = None
        self.prev_action_idx = None
        self.prev_log_prob = None
        self.prev_value = None
        self.buffer.reset()
