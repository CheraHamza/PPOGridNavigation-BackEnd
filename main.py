from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import json
import random
from collections import deque
from agent import PPOAgent
from database import SessionLocal, init_db, SavedModel
from environment import GridWorld


# NOTE: init_db() is called in the startup event below, NOT at module level.
# This lets gunicorn bind to the port immediately so Render doesn't time out
# waiting for an open HTTP port during cold-starts.

app = FastAPI()


@app.on_event("startup")
def on_startup():
    """Run DB migrations after the server has bound to the port."""
    init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


agent = PPOAgent(height=10, width=10)
active_model_name = "Session Agent"

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class StepData(BaseModel):
    position: list[int]
    target: list[int]
    checkpoints: list[list[int]] = []
    obstacles: list[list[int]] = []
    reward: float
    done: bool

class ActionResponse(BaseModel):
    action: str
    epsilon: float

class EnvironmentConfig(BaseModel):
    height: int = 10
    width: int = 10
    starting_position: list[int] = [0, 0]
    target_position: list[int] = [8, 8]
    checkpoints: list[list[int]] = []
    obstacles: list[list[int]] = []

class ModelCreate(BaseModel):
    name: str
    environment: EnvironmentConfig = EnvironmentConfig()

class ModelList(BaseModel):
    id: int
    name: str
    algorithm: str = "PPO"
    epsilon: float = 1.0
    ppo_updates: int = 0
    total_steps: int = 0
    total_episodes: int = 0
    avg_reward: float = 0.0
    success_rate: float = 0.0
    created_at: datetime
    environment: Optional[EnvironmentConfig] = None

class AgentStatusResponse(BaseModel):
    model_name: str = "Session Agent"
    epsilon: float = 1.0
    ppo_updates: int = 0
    total_steps: int = 0
    total_episodes: int = 0
    avg_reward: float = 0.0
    success_rate: float = 0.0

class TrainRequest(BaseModel):
    episodes: int = 500
    height: int = 10
    width: int = 10
    starting_position: list[int] = [0, 0]
    target_position: list[int] = [8, 8]
    checkpoints: list[list[int]] = []
    obstacles: list[list[int]] = []
    randomize_targets: bool = True
    randomize_obstacles: bool = True
    num_random_obstacles: int = 5
    randomize_checkpoints: bool = False
    num_random_checkpoints: int = 2
    visualize: bool = False  # Stream step-by-step data for visualization
    visualize_interval: int = 10  # Only visualize every N episodes (reduces overhead)

class EpisodeResult(BaseModel):
    episode: int
    steps: int
    total_reward: float
    reached_target: bool

class TrainResponse(BaseModel):
    episodes_trained: int
    epsilon: float
    results: list[EpisodeResult]


def _compute_training_stats(results: list[dict]) -> tuple[float, float]:
    """Compute aggregate reward and success-rate metrics."""
    if not results:
        return 0.0, 0.0

    avg_reward = sum(r.get("total_reward", 0.0) for r in results) / len(results)
    success_rate = sum(1 for r in results if r.get("reached_target")) / len(results)
    return float(avg_reward), float(success_rate)


def _is_reachable(width: int, height: int, start: tuple[int, int], target: tuple[int, int], obstacles: set[tuple[int, int]]) -> bool:
    """Return True if a path exists from start to target on the current grid."""
    if start == target:
        return True

    q = deque([start])
    visited = {start}

    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if (nx, ny) in obstacles or (nx, ny) in visited:
                continue
            if (nx, ny) == target:
                return True
            visited.add((nx, ny))
            q.append((nx, ny))

    return False


def _are_waypoints_reachable(
    width: int,
    height: int,
    start: tuple[int, int],
    waypoints: list[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> bool:
    """Return True if each leg in start -> waypoints is reachable in order."""
    current = start
    for waypoint in waypoints:
        if not _is_reachable(width, height, current, waypoint, obstacles):
            return False
        current = waypoint
    return True


@app.get("/health")
def health():
    """Lightweight liveness probe – used by the frontend to detect cold-start."""
    return {"status": "ok"}


@app.get("/agent/status", response_model=AgentStatusResponse)
def agent_status():
    """Return current in-memory PPO agent statistics."""
    return {
        "model_name": active_model_name,
        "epsilon": float(getattr(agent, "epsilon", 1.0)),
        "ppo_updates": int(getattr(agent, "updates", 0)),
        "total_steps": int(getattr(agent, "total_steps", 0)),
        "total_episodes": int(getattr(agent, "total_episodes", 0)),
        "avg_reward": float(getattr(agent, "last_avg_reward", 0.0)),
        "success_rate": float(getattr(agent, "last_success_rate", 0.0)),
    }


@app.post("/act", response_model=ActionResponse)
def act(data: StepData):
    # Convert obstacles list[list[int]] -> list[tuple] for the agent
    obstacles = [tuple(o) for o in data.obstacles]

    # 1. Learn from the previous step (using current reward)
    agent.learn(data.position, data.target, obstacles, data.reward, data.done)

    # 2. If episode is done, we don't need a new action
    if data.done:
        return {"action": "stop", "epsilon": agent.epsilon}

    # 3. Choose next action based on current state (position + target + obstacles)
    action = agent.choose_action(data.position, data.target, obstacles)

    # 4. Remember this state/action for the next learning step
    agent.update_memory(data.position, data.target, obstacles, action)

    return {"action": action, "epsilon": agent.epsilon}


@app.post("/train")
def train(req: TrainRequest):
    """
    Run many episodes server-side, streaming progress as newline-delimited JSON.
    Each line is either a progress update or the final result.
    Auto-saves checkpoint every 100 episodes to prevent data loss.
    """
    def generate():
        env = GridWorld(
            height=req.height,
            width=req.width,
            starting_position=req.starting_position,
            target_position=req.target_position,
            checkpoints=req.checkpoints,
            obstacles=req.obstacles,
        )

        obstacles = [tuple(o) for o in req.obstacles]
        target = req.target_position
        checkpoints = [
            [int(cp[0]), int(cp[1])]
            for cp in req.checkpoints
            if len(cp) == 2
        ]
        start = (int(req.starting_position[0]), int(req.starting_position[1]))

        results = []
        progress_interval = max(1, req.episodes // 100)  # ~100 progress updates
        checkpoint_interval = 100  # auto-save every 100 episodes
        visualize_this_ep = False

        for ep in range(1, req.episodes + 1):
                # Decide if we should visualize this episode
                visualize_this_ep = req.visualize and (ep % req.visualize_interval == 0)
                # 1. Randomize target first so obstacles won't land on it
                occupied = {start}
                ep_checkpoints = []

                if req.randomize_targets:
                    while True:
                        tx = random.randint(0, req.width - 1)
                        ty = random.randint(0, req.height - 1)
                        pos = (tx, ty)
                        if pos not in occupied:
                            ep_target = [tx, ty]
                            break
                    env.target_position = ep_target
                else:
                    ep_target = target
                ep_target_tuple = (int(ep_target[0]), int(ep_target[1]))
                occupied.add(ep_target_tuple)

                # 2. Randomize checkpoints if enabled, otherwise use provided ones
                if req.randomize_checkpoints:
                    # Generate random checkpoints avoiding start and target
                    reserved_cells = 2  # start and target
                    max_checkpoints = max(
                        0,
                        min(req.num_random_checkpoints, req.height * req.width - reserved_cells),
                    )
                    attempts_left = 30

                    while attempts_left > 0:
                        attempts_left -= 1
                        local_occupied = set(occupied)
                        n = random.randint(1, max(1, max_checkpoints))  # at least 1 checkpoint
                        candidate_checkpoints = []

                        for _ in range(n):
                            inner_attempts = 50
                            while inner_attempts > 0:
                                inner_attempts -= 1
                                cx = random.randint(0, req.width - 1)
                                cy = random.randint(0, req.height - 1)
                                pos = (cx, cy)
                                if pos not in local_occupied:
                                    candidate_checkpoints.append([cx, cy])
                                    local_occupied.add(pos)
                                    break

                        # Check if path through all checkpoints to target is valid
                        waypoint_chain = [
                            (int(cp[0]), int(cp[1])) for cp in candidate_checkpoints
                        ] + [ep_target_tuple]
                        if _are_waypoints_reachable(
                            req.width,
                            req.height,
                            start,
                            waypoint_chain,
                            set(),  # no obstacles yet
                        ):
                            ep_checkpoints = candidate_checkpoints
                            for cp in ep_checkpoints:
                                occupied.add((cp[0], cp[1]))
                            break
                else:
                    # Use provided checkpoints
                    for cp in checkpoints:
                        cp_tuple = (cp[0], cp[1])
                        if cp_tuple not in occupied:
                            ep_checkpoints.append([cp[0], cp[1]])
                            occupied.add(cp_tuple)
                    # Remove checkpoints that overlap with target
                    ep_checkpoints = [
                        cp for cp in ep_checkpoints if (int(cp[0]), int(cp[1])) != ep_target_tuple
                    ]

                waypoint_chain = [
                    (int(cp[0]), int(cp[1])) for cp in ep_checkpoints
                ] + [ep_target_tuple]

                # 2. Randomize obstacles, avoiding start and target.
                # Regenerate until the episode remains solvable.
                if req.randomize_obstacles:
                    reserved_cells = 2 + len(ep_checkpoints)
                    max_obstacles = max(
                        0,
                        min(req.num_random_obstacles, req.height * req.width - reserved_cells),
                    )
                    attempts_left = 30
                    ep_obstacles = []

                    while attempts_left > 0:
                        attempts_left -= 1
                        local_occupied = set(occupied)
                        n = random.randint(0, max_obstacles)
                        candidate = []

                        for _ in range(n):
                            while True:
                                ox = random.randint(0, req.width - 1)
                                oy = random.randint(0, req.height - 1)
                                pos = (ox, oy)
                                if pos not in local_occupied:
                                    candidate.append(pos)
                                    local_occupied.add(pos)
                                    break

                        if _are_waypoints_reachable(
                            req.width,
                            req.height,
                            start,
                            waypoint_chain,
                            set(candidate),
                        ):
                            ep_obstacles = candidate
                            break
                else:
                    ep_obstacles = obstacles

                # Sync obstacles into the env for this episode
                env.obstacles = {(o[0], o[1]) for o in ep_obstacles}
                env.checkpoints = [list(cp) for cp in ep_checkpoints]
                env.target_position = [int(ep_target[0]), int(ep_target[1])]
                ep_obstacles_list = [list(o) for o in ep_obstacles]

                obs = env.reset()
                total_reward = 0.0
                reached_target = False

                # Emit episode start for visualization
                if visualize_this_ep:
                    yield json.dumps({
                        "type": "episode_start",
                        "episode": ep,
                        "target": obs["target"],
                        "final_target": ep_target,
                        "checkpoints": [list(cp) for cp in ep_checkpoints],
                        "checkpoint_index": obs["checkpoint_index"],
                        "obstacles": ep_obstacles_list,
                        "position": obs["position"],
                        "epsilon": round(agent.epsilon, 4),
                    }) + "\n"

                # First step: no learning yet, just pick an action
                agent.learn(obs["position"], obs["target"], ep_obstacles_list, 0.0, False)
                action = agent.choose_action(obs["position"], obs["target"], ep_obstacles_list)
                agent.update_memory(obs["position"], obs["target"], ep_obstacles_list, action)

                step_count = 0
                while True:
                    result = env.step(action)
                    state = result["state"]
                    reward = result["reward"]
                    done = result["done"]
                    total_reward += reward
                    step_count += 1

                    # Emit step data for visualization
                    if visualize_this_ep:
                        yield json.dumps({
                            "type": "step",
                            "episode": ep,
                            "step": step_count,
                            "position": state["position"],
                            "target": state["target"],
                            "checkpoint_index": state["checkpoint_index"],
                            "total_checkpoints": state["total_checkpoints"],
                            "action": action,
                            "reward": round(reward, 4),
                            "done": done,
                        }) + "\n"

                    agent.learn(state["position"], state["target"], ep_obstacles_list, reward, done)

                    if done:
                        reached_target = (
                            state["position"][0] == ep_target[0]
                            and state["position"][1] == ep_target[1]
                            and state["checkpoint_index"] == state["total_checkpoints"]
                        )
                        break

                    action = agent.choose_action(state["position"], state["target"], ep_obstacles_list)
                    agent.update_memory(state["position"], state["target"], ep_obstacles_list, action)

                results.append({
                    "episode": ep,
                    "steps": result["steps"],
                    "total_reward": round(total_reward, 4),
                    "reached_target": reached_target,
                })

                # Send progress update periodically
                if ep % progress_interval == 0 or ep == req.episodes:
                    recent = results[-min(50, len(results)):]
                    recent_success = sum(1 for r in recent if r["reached_target"])
                    recent_steps = [r["steps"] for r in recent]
                    avg_steps = sum(recent_steps) / len(recent_steps) if recent_steps else 0
                    yield json.dumps({
                        "type": "progress",
                        "episode": ep,
                        "total": req.episodes,
                        "epsilon": round(agent.epsilon, 4),
                        "recent_success_rate": round(recent_success / len(recent) * 100, 1),
                        "avg_steps": round(avg_steps, 1),
                        "ppo_updates": getattr(agent, "updates", 0),
                        "total_steps": getattr(agent, "total_steps", 0),
                    }) + "\n"

                # Auto-checkpoint to prevent data loss on crash
                if ep % checkpoint_interval == 0 and ep < req.episodes:
                    db = SessionLocal()
                    try:
                        binary_data = agent.to_bytes()
                        avg_reward, success_rate = _compute_training_stats(results)
                        env_config = json.dumps({
                            "height": req.height,
                            "width": req.width,
                            "starting_position": req.starting_position,
                            "target_position": req.target_position,
                            "checkpoints": req.checkpoints,
                            "obstacles": req.obstacles,
                        })
                        # Update or create checkpoint
                        checkpoint = db.query(SavedModel).filter(
                            SavedModel.name == "_auto_checkpoint"
                        ).first()
                        if checkpoint:
                            checkpoint.data = binary_data
                            checkpoint.algorithm = "PPO"
                            checkpoint.pseudo_epsilon = agent.epsilon
                            checkpoint.ppo_updates = getattr(agent, "updates", 0)
                            checkpoint.total_steps = getattr(agent, "total_steps", 0)
                            checkpoint.total_episodes = getattr(agent, "total_episodes", 0)
                            checkpoint.avg_reward = avg_reward
                            checkpoint.success_rate = success_rate
                            checkpoint.environment_config = env_config
                        else:
                            checkpoint = SavedModel(
                                name="_auto_checkpoint",
                                algorithm="PPO",
                                pseudo_epsilon=agent.epsilon,
                                ppo_updates=getattr(agent, "updates", 0),
                                total_steps=getattr(agent, "total_steps", 0),
                                total_episodes=getattr(agent, "total_episodes", 0),
                                avg_reward=avg_reward,
                                success_rate=success_rate,
                                data=binary_data,
                                environment_config=env_config,
                            )
                            db.add(checkpoint)
                        db.commit()
                    finally:
                        db.close()

        avg_reward, success_rate = _compute_training_stats(results)
        setattr(agent, "last_avg_reward", avg_reward)
        setattr(agent, "last_success_rate", success_rate)

        # Final result
        yield json.dumps({
            "type": "done",
            "episodes_trained": req.episodes,
            "epsilon": agent.epsilon,
            "ppo_updates": getattr(agent, "updates", 0),
            "total_steps": getattr(agent, "total_steps", 0),
            "total_episodes": getattr(agent, "total_episodes", 0),
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "results": results,
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/models", response_model=List[ModelList])
def list_models(db: Session = Depends(get_db)):
    """Return a list of all saved PPO models."""
    rows = db.query(SavedModel).all()
    result = []
    for row in rows:
        env_config = None
        if row.environment_config:
            try:
                env_config = json.loads(row.environment_config)
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(ModelList(
            id=row.id,
            name=row.name,
            algorithm=row.algorithm,
            epsilon=row.pseudo_epsilon,
            ppo_updates=row.ppo_updates,
            total_steps=row.total_steps,
            total_episodes=row.total_episodes,
            avg_reward=row.avg_reward,
            success_rate=row.success_rate,
            created_at=row.created_at,
            environment=env_config,
        ))
    return result

@app.post("/reset")
def reset_agent():
    """Reset the agent: fresh network weights, empty replay buffer, epsilon=1.0."""
    global agent, active_model_name
    agent = PPOAgent(height=10, width=10)
    active_model_name = "Session Agent"
    return {"status": "reset", "epsilon": agent.epsilon}


@app.post("/models")
def save_model(model_input: ModelCreate, db: Session = Depends(get_db)):
    """Save the current PPO agent + environment config to the DB."""
    binary_data = agent.to_bytes()
    env_json = model_input.environment.model_dump_json()

    avg_reward = getattr(agent, "last_avg_reward", 0.0)
    success_rate = getattr(agent, "last_success_rate", 0.0)

    new_model = SavedModel(
        name=model_input.name,
        algorithm="PPO",
        pseudo_epsilon=agent.epsilon,
        ppo_updates=getattr(agent, "updates", 0),
        total_steps=getattr(agent, "total_steps", 0),
        total_episodes=getattr(agent, "total_episodes", 0),
        avg_reward=avg_reward,
        success_rate=success_rate,
        data=binary_data,
        environment_config=env_json,
    )
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return {"status": "saved", "id": new_model.id, "name": new_model.name}

@app.post("/models/{model_id}/load")
def load_model(model_id: int, db: Session = Depends(get_db)):
    """Load a specific model from DB into the active agent."""
    global active_model_name
    saved_model = db.query(SavedModel).filter(SavedModel.id == model_id).first()
    if not saved_model:
        raise HTTPException(status_code=404, detail="Model not found")

    agent.from_bytes(saved_model.data)
    active_model_name = saved_model.name

    # Parse stored environment config
    env_config = None
    if saved_model.environment_config:
        try:
            env_config = json.loads(saved_model.environment_config)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "status": "loaded",
        "epsilon": agent.epsilon,
        "ppo_updates": getattr(agent, "updates", 0),
        "total_steps": getattr(agent, "total_steps", 0),
        "total_episodes": getattr(agent, "total_episodes", 0),
        "avg_reward": saved_model.avg_reward,
        "success_rate": saved_model.success_rate,
        "name": saved_model.name,
        "environment": env_config,
    }

@app.delete("/models/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Delete a model."""
    saved_model = db.query(SavedModel).filter(SavedModel.id == model_id).first()
    if not saved_model:
        raise HTTPException(status_code=404, detail="Model not found")

    db.delete(saved_model)
    db.commit()
    return {"status": "deleted"}
