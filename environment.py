"""Server-side GridWorld environment – mirrors the JS GridWorld exactly."""


class GridWorld:
    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        starting_position: list[int] | None = None,
        target_position: list[int] | None = None,
        checkpoints: list[list[int]] | None = None,
        obstacles: list[list[int]] | None = None,
    ):
        self.height = height
        self.width = width
        self.starting_position = list(starting_position or [0, 0])
        self.target_position = list(target_position or [8, 8])
        self.checkpoints = [list(cp) for cp in (checkpoints or [])]
        self.completed_checkpoints = 0
        self.obstacles: set[tuple[int, int]] = set()
        if obstacles:
            self.obstacles = {(o[0], o[1]) for o in obstacles}
        self.current_position = list(self.starting_position)
        self.max_steps = 100
        self.steps = 0

    def reset(self) -> dict:
        self.current_position = list(self.starting_position)
        self.completed_checkpoints = 0
        self.steps = 0
        return self._get_state()

    def _current_objective(self) -> list[int]:
        if self.completed_checkpoints < len(self.checkpoints):
            return list(self.checkpoints[self.completed_checkpoints])
        return list(self.target_position)

    def _get_state(self) -> dict:
        return {
            "position": list(self.current_position),
            "target": self._current_objective(),
            "final_target": list(self.target_position),
            "checkpoints": [list(cp) for cp in self.checkpoints],
            "checkpoint_index": self.completed_checkpoints,
            "total_checkpoints": len(self.checkpoints),
        }

    @staticmethod
    def _manhattan(pos, target):
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

    def step(self, action: str) -> dict:
        x, y = self.current_position
        new_x, new_y = x, y
        done = False
        objective_before_move = self._current_objective()

        if action == "up":
            if y > 0:
                new_y = y - 1
        elif action == "down":
            if y < self.height - 1:
                new_y = y + 1
        elif action == "left":
            if x > 0:
                new_x = x - 1
        elif action == "right":
            if x < self.width - 1:
                new_x = x + 1

        # Check obstacle collision – agent stays in place
        hit_obstacle = (new_x, new_y) in self.obstacles

        # Block the goal until all checkpoints are completed
        goal_blocked = (
            self.completed_checkpoints < len(self.checkpoints)
            and new_x == self.target_position[0]
            and new_y == self.target_position[1]
        )

        if hit_obstacle or goal_blocked:
            new_x, new_y = x, y

        self.current_position = [new_x, new_y]
        self.steps += 1

        # Only apply shaping when agent actually moved (prevents wall-hit exploit)
        actually_moved = (new_x != x or new_y != y)
        if actually_moved:
            # Simple delta shaping: +1 closer, -1 farther, 0 same
            # No gamma term prevents oscillation exploits
            old_dist = self._manhattan([x, y], objective_before_move)
            new_dist = self._manhattan([new_x, new_y], objective_before_move)
            r_shape = float(old_dist - new_dist)
        else:
            r_shape = 0.0

        reached_objective = (
            new_x == objective_before_move[0] and new_y == objective_before_move[1]
        )

        # Terminal reward: +1.0 final goal, progressive checkpoint reward, -1 obstacle
        if reached_objective and self.completed_checkpoints < len(self.checkpoints):
            # Progressive checkpoint reward: later checkpoints give higher rewards
            # Formula: 0.5 + 0.5 * (checkpoint_index / max(total - 1, 1))
            # Range: 0.5 (first) to 1.0 (last)
            total = len(self.checkpoints)
            idx = self.completed_checkpoints
            progress = idx / max(total - 1, 1) if total > 1 else 1.0
            r_terminal = 0.5 + 0.5 * progress
            self.completed_checkpoints += 1
        elif reached_objective:
            r_terminal = 1.0
            done = True
        elif hit_obstacle:
            r_terminal = -1.0
        else:
            # No penalty for hitting blocked goal - agent just bounces off like a wall
            r_terminal = 0.0

        # Step penalty encourages efficiency
        r_step = -0.01

        # Combined reward: r = r_terminal + r_shape + r_step
        reward = r_terminal + r_shape + r_step

        if self.steps >= self.max_steps:
            done = True

        return {
            "state": self._get_state(),
            "reward": reward,
            "done": done,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
