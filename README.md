# PPO Grid Navigation — Backend

FastAPI backend serving a **Proximal Policy Optimization (PPO)** agent for 10×10 grid navigation.


## Architecture

### PPO Agent (`agent.py`)

| Component          | Details                                                                  |
| ------------------ | ------------------------------------------------------------------------ |
| **Network**        | Actor-Critic MLP with shared backbone                                    |
| **Input**          | `(27,)` float vector — goal-relative encoding: Δx, Δy, 5×5 obstacle grid |
| **Output**         | 4 action logits (up, down, left, right) + state value                    |
| **Rollout buffer** | On-policy, cleared after each policy update                              |
| **Exploration**    | Entropy bonus, no ε-greedy                                               |
| **Optimiser**      | Adam, lr = 1 × 10⁻³                                                      |

### Endpoints (`main.py`)

| Method | Path             | Description                         |
| ------ | ---------------- | ----------------------------------- |
| GET    | `/`              | Health check                        |
| POST   | `/act`           | Agent chooses an action             |
| POST   | `/train`         | Run N training episodes server-side |
| POST   | `/reset`         | Re-initialise agent from scratch    |
| POST   | `/models/save`   | Persist model to database           |
| GET    | `/models`        | List saved models                   |
| POST   | `/models/load`   | Load a saved model                  |
| DELETE | `/models/{name}` | Delete a saved model                |

### Persistence (`database.py`)

Model weights are serialised with `torch.save` into a byte blob and stored via SQLAlchemy (SQLite locally, PostgreSQL on Render/Heroku via `DATABASE_URL`).

## Getting Started

```bash
# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dev server
uvicorn main:app --reload --port 8000
```

## Tech Stack

| Layer     | Technology         |
| --------- | ------------------ |
| Framework | FastAPI 0.128      |
| ML        | PyTorch ≥ 2.2      |
| ORM       | SQLAlchemy 2.x     |
| Server    | Uvicorn / Gunicorn |
