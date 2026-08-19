from __future__ import annotations

import argparse
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from game import Board, DIRECTIONS, bprint, legal_actions, new_game, step


WEIGHTS_PATH = Path(__file__).resolve().with_name("dqn_2048.pt")
ACTION_TO_INDEX = {action: index for index, action in enumerate(DIRECTIONS)}
STATE_SCALE = 16.0
REWARD_SCALE = 16.0


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 2_500
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_steps: int = 2_000
    train_every: int = 4
    target_update_every: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    max_steps_per_episode: int = 10_000


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminal: bool
    next_legal_mask: np.ndarray


class QNetwork(nn.Module):
    """A 25,076-parameter approximation of Q(board, action)."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 48),
            nn.ReLU(),
            nn.Linear(48, len(DIRECTIONS)),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.layers(states)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def encode_board(board: Board) -> np.ndarray:
    """Flatten exponent-valued tiles and keep network inputs near unit scale."""
    return np.asarray(board, dtype=np.float32).reshape(16) / STATE_SCALE


def legal_mask(board: Board) -> np.ndarray:
    mask = np.zeros(len(DIRECTIONS), dtype=np.bool_)
    for action in legal_actions(board):
        mask[ACTION_TO_INDEX[action]] = True
    return mask


def epsilon_at(step_count: int, config: TrainingConfig) -> float:
    decay = math.exp(-step_count / config.epsilon_decay_steps)
    return config.epsilon_end + (config.epsilon_start - config.epsilon_end) * decay


def choose_action(
    network: QNetwork,
    state: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
    rng: random.Random,
    device: torch.device,
) -> int:
    legal_indices = np.flatnonzero(mask).tolist()
    if not legal_indices:
        raise ValueError("cannot choose an action in a terminal state")
    if epsilon > 0.0 and rng.random() < epsilon:
        return rng.choice(legal_indices)

    with torch.no_grad():
        state_tensor = torch.from_numpy(state).to(device).unsqueeze(0)
        q_values = network(state_tensor).squeeze(0)
        tensor_mask = torch.from_numpy(mask).to(device)
        q_values = q_values.masked_fill(~tensor_mask, -torch.inf)
        return int(q_values.argmax().item())


def optimize_batch(
    online_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: Deque[Transition],
    config: TrainingConfig,
    rng: random.Random,
    device: torch.device,
) -> float:
    batch = rng.sample(replay, config.batch_size)
    states = torch.from_numpy(np.stack([item.state for item in batch])).to(device)
    actions = torch.tensor([item.action for item in batch], device=device)
    rewards = torch.tensor([item.reward for item in batch], device=device)
    next_states = torch.from_numpy(np.stack([item.next_state for item in batch])).to(device)
    terminals = torch.tensor([item.terminal for item in batch], device=device)
    next_masks = torch.from_numpy(
        np.stack([item.next_legal_mask for item in batch])
    ).to(device)

    predicted_q = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_values = torch.zeros(config.batch_size, device=device)
        nonterminal = ~terminals
        if nonterminal.any():
            online_next_q = online_network(next_states[nonterminal])
            online_next_q = online_next_q.masked_fill(
                ~next_masks[nonterminal], -torch.inf
            )
            next_actions = online_next_q.argmax(dim=1, keepdim=True)
            target_next_q = target_network(next_states[nonterminal])
            next_values[nonterminal] = target_next_q.gather(1, next_actions).squeeze(1)
        targets = rewards + config.gamma * next_values

    loss = F.smooth_l1_loss(predicted_q, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train(
    config: TrainingConfig,
    seed: int,
    device: torch.device,
) -> QNetwork:
    """Train from sampled games without enumerating possible next states."""
    environment_rng = random.Random(seed)
    policy_rng = random.Random(seed + 1)
    replay_rng = random.Random(seed + 2)
    np.random.seed(seed)
    torch.manual_seed(seed)

    online_network = QNetwork().to(device)
    target_network = QNetwork().to(device)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()
    optimizer = torch.optim.Adam(
        online_network.parameters(), lr=config.learning_rate
    )
    replay: Deque[Transition] = deque(maxlen=config.replay_capacity)
    recent_scores: Deque[int] = deque(maxlen=100)
    total_steps = 0
    last_loss = 0.0
    report_every = max(1, config.episodes // 20)

    print(
        f"Training {online_network.parameter_count:,} parameters on {device} "
        f"for {config.episodes:,} episodes"
    )

    for episode in range(1, config.episodes + 1):
        board = new_game(environment_rng)
        game_score = 0

        for _ in range(config.max_steps_per_episode):
            state = encode_board(board)
            mask = legal_mask(board)
            if not mask.any():
                break

            epsilon = epsilon_at(total_steps, config)
            action_index = choose_action(
                online_network, state, mask, epsilon, policy_rng, device
            )
            next_board, merge_score, terminal = step(
                board, DIRECTIONS[action_index], environment_rng
            )
            next_mask = legal_mask(next_board) if not terminal else np.zeros(4, dtype=np.bool_)
            replay.append(
                Transition(
                    state=state,
                    action=action_index,
                    reward=merge_score / REWARD_SCALE,
                    next_state=encode_board(next_board),
                    terminal=terminal,
                    next_legal_mask=next_mask,
                )
            )

            board = next_board
            game_score += merge_score
            total_steps += 1

            if (
                total_steps >= config.warmup_steps
                and total_steps % config.train_every == 0
                and len(replay) >= config.batch_size
            ):
                last_loss = optimize_batch(
                    online_network,
                    target_network,
                    optimizer,
                    replay,
                    config,
                    replay_rng,
                    device,
                )

            if total_steps % config.target_update_every == 0:
                target_network.load_state_dict(online_network.state_dict())

            if terminal:
                break

        recent_scores.append(game_score)
        if episode == 1 or episode % report_every == 0 or episode == config.episodes:
            average_score = sum(recent_scores) / len(recent_scores)
            max_tile = 2 ** max(tile for row in board for tile in row)
            print(
                f"episode {episode:>5}/{config.episodes}  "
                f"score {game_score:>6}  avg100 {average_score:>8.1f}  "
                f"max {max_tile:>5}  epsilon {epsilon_at(total_steps, config):.3f}  "
                f"loss {last_loss:.4f}"
            )

    return online_network


def save_weights(network: QNetwork, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(network.state_dict(), temporary_path)
    temporary_path.replace(path)
    print(f"Saved weights to {path}")


def load_weights(path: Path, device: torch.device) -> QNetwork:
    network = QNetwork().to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    network.load_state_dict(state_dict)
    network.eval()
    return network


def play(
    network: QNetwork,
    games: int,
    seed: int,
    delay: float,
    device: torch.device,
) -> None:
    """Play complete games greedily with the trained Q-network."""
    rng = random.Random(seed)
    network.eval()

    for game_number in range(1, games + 1):
        board = new_game(rng)
        score = 0
        move_count = 0
        print(f"\nGame {game_number}")
        bprint(board)

        while True:
            mask = legal_mask(board)
            if not mask.any():
                break
            action_index = choose_action(
                network,
                encode_board(board),
                mask,
                epsilon=0.0,
                rng=rng,
                device=device,
            )
            board, merge_score, terminal = step(
                board, DIRECTIONS[action_index], rng
            )
            score += merge_score
            move_count += 1
            print(f"move {move_count}: {DIRECTIONS[action_index]}  +{merge_score}")
            bprint(board)
            if delay:
                time.sleep(delay)
            if terminal:
                break

        max_tile = 2 ** max(tile for row in board for tile in row)
        print(f"Game over. Score: {score}. Max tile: {max_tile}.")


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and play a model-free 2048 DQN")
    parser.add_argument("--episodes", type=int, default=2_500)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument(
        "--play-only",
        action="store_true",
        help="load existing weights instead of training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    if args.play_only:
        if not args.weights.exists():
            raise SystemExit(f"Weights do not exist: {args.weights}")
        network = load_weights(args.weights, device)
    else:
        config = TrainingConfig(episodes=args.episodes)
        network = train(config, args.seed, device)
        save_weights(network, args.weights)

    play(network, args.games, args.seed + 1, args.delay, device)


if __name__ == "__main__":
    main()
