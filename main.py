from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full
from typing import Deque, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from game import (
    Board,
    DIRECTIONS,
    bprint,
    game_over,
    legal_actions,
    new_game,
    slide,
    spawn_outcomes,
    step,
)


WEIGHTS_PATH = Path(__file__).resolve().with_name("value_2048.pt")
TILE_CHANNELS = 16
REWARD_SCALE = 16.0
FrozenBoard = tuple[int, ...]


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 2_500
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 128
    replay_capacity: int = 200_000
    warmup_examples: int = 2_000
    train_every: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    max_steps_per_episode: int = 10_000
    checkpoint_every: int = 1_000


class ValueExample(NamedTuple):
    board: np.ndarray
    target: float


class EpisodeResult(NamedTuple):
    episode: int
    boards: list[np.ndarray]
    rewards: list[float]
    score: int
    max_tile: int


class PolicySnapshot(NamedTuple):
    state_dict: dict[str, Tensor]
    epsilon: float


class WorkerFailure(NamedTuple):
    worker: int
    message: str
    details: str


class ValueNetwork(nn.Module):
    """An 86,657-parameter estimate of expected discounted game return."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(TILE_CHANNELS, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.head(self.features(states)).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def encode_board(board: Board | np.ndarray) -> np.ndarray:
    """Encode tile exponents as 16 one-hot 4x4 planes."""
    tiles = np.asarray(board, dtype=np.int64)
    if tiles.shape != (4, 4) or np.any(tiles < 0):
        raise ValueError("board must be a 4x4 grid of nonnegative exponents")
    tiles = np.minimum(tiles, TILE_CHANNELS - 1)
    encoded = np.eye(TILE_CHANNELS, dtype=np.float32)[tiles]
    return np.ascontiguousarray(encoded.transpose(2, 0, 1))


def freeze_board(board: Board) -> FrozenBoard:
    return tuple(tile for row in board for tile in row)


def thaw_board(board: FrozenBoard) -> Board:
    return [list(board[start : start + 4]) for start in range(0, 16, 4)]


@dataclass
class ActionBranch:
    action: str
    reward: float
    outcomes: list[tuple[float, SearchNode]]


@dataclass
class SearchNode:
    terminal: bool = False
    leaf_index: int | None = None
    branches: list[ActionBranch] = field(default_factory=list)


class SearchTree:
    """A finite expectimax tree whose unique leaves are evaluated in one batch."""

    def __init__(self, depth: int) -> None:
        if depth < 1:
            raise ValueError("search depth must be at least one")
        self.depth = depth
        self.cache: dict[tuple[FrozenBoard, int], SearchNode] = {}
        self.leaf_indices: dict[FrozenBoard, int] = {}
        self.leaf_boards: list[Board] = []

    def build(self, board: Board) -> SearchNode:
        return self._build_node(board, self.depth)

    def _build_node(self, board: Board, depth: int) -> SearchNode:
        key = (freeze_board(board), depth)
        if key in self.cache:
            return self.cache[key]

        node = SearchNode()
        self.cache[key] = node

        if game_over(board):
            node.terminal = True
            return node

        if depth == 0:
            frozen = key[0]
            if frozen not in self.leaf_indices:
                self.leaf_indices[frozen] = len(self.leaf_boards)
                self.leaf_boards.append(thaw_board(frozen))
            node.leaf_index = self.leaf_indices[frozen]
            return node

        for action in DIRECTIONS:
            moved, merge_score = slide(board, action)
            if moved == board:
                continue
            outcomes = [
                (probability, self._build_node(next_board, depth - 1))
                for probability, next_board in spawn_outcomes(moved)
            ]
            node.branches.append(
                ActionBranch(
                    action=action,
                    reward=merge_score / REWARD_SCALE,
                    outcomes=outcomes,
                )
            )

        if not node.branches:
            node.terminal = True
        return node


def estimate_boards(
    network: ValueNetwork,
    boards: list[Board],
    device: torch.device,
    batch_size: int = 4_096,
) -> np.ndarray:
    if not boards:
        return np.empty(0, dtype=np.float32)

    was_training = network.training
    network.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(boards), batch_size):
            batch = np.stack(
                [encode_board(board) for board in boards[start : start + batch_size]]
            )
            predictions = network(torch.from_numpy(batch).to(device))
            values.append(predictions.cpu().numpy())
    if was_training:
        network.train()
    return np.concatenate(values)


def expectimax_action_values(
    network: ValueNetwork,
    board: Board,
    depth: int,
    gamma: float,
    device: torch.device,
) -> dict[str, float]:
    """Return exact chance-weighted action values with neural leaf estimates."""
    tree = SearchTree(depth)
    root = tree.build(board)
    if root.terminal:
        return {}

    leaf_values = estimate_boards(network, tree.leaf_boards, device)
    node_values: dict[int, float] = {}

    def evaluate_node(node: SearchNode) -> float:
        node_id = id(node)
        if node_id in node_values:
            return node_values[node_id]
        if node.terminal:
            value = 0.0
        elif node.leaf_index is not None:
            value = float(leaf_values[node.leaf_index])
        else:
            value = max(evaluate_branch(branch) for branch in node.branches)
        node_values[node_id] = value
        return value

    def evaluate_branch(branch: ActionBranch) -> float:
        expected_future = sum(
            probability * evaluate_node(child)
            for probability, child in branch.outcomes
        )
        return branch.reward + gamma * expected_future

    return {
        branch.action: evaluate_branch(branch)
        for branch in root.branches
    }


def choose_action(
    network: ValueNetwork,
    board: Board,
    depth: int,
    gamma: float,
    epsilon: float,
    rng: random.Random,
    device: torch.device,
) -> tuple[str, dict[str, float]]:
    actions = legal_actions(board)
    if not actions:
        raise ValueError("cannot choose an action in a terminal state")
    if epsilon > 0.0 and rng.random() < epsilon:
        return rng.choice(actions), {}

    values = expectimax_action_values(network, board, depth, gamma, device)
    action = max(actions, key=lambda candidate: values[candidate])
    return action, values


def epsilon_at(step_count: int, config: TrainingConfig) -> float:
    decay = math.exp(-step_count / config.epsilon_decay_steps)
    return config.epsilon_end + (config.epsilon_start - config.epsilon_end) * decay


def generate_episode(
    network: ValueNetwork,
    episode: int,
    seed: int,
    epsilon: float,
    search_depth: int,
    gamma: float,
    max_steps: int,
    device: torch.device,
) -> EpisodeResult:
    """Generate one independent self-play trajectory from a fixed policy."""
    seed_offset = (episode - 1) * 2
    environment_rng = random.Random(seed + seed_offset)
    policy_rng = random.Random(seed + seed_offset + 1)
    board = new_game(environment_rng)
    boards: list[np.ndarray] = []
    rewards: list[float] = []
    score = 0

    for _ in range(max_steps):
        if game_over(board):
            break
        boards.append(np.asarray(board, dtype=np.int16))
        action, _ = choose_action(
            network,
            board,
            search_depth,
            gamma,
            epsilon,
            policy_rng,
            device,
        )
        board, merge_score, terminal = step(board, action, environment_rng)
        rewards.append(merge_score / REWARD_SCALE)
        score += merge_score
        if terminal:
            break

    max_tile = 2 ** max(tile for row in board for tile in row)
    return EpisodeResult(episode, boards, rewards, score, max_tile)


def initialize_rollout_worker() -> None:
    """Keep multiple worker processes from each starting a CPU thread pool."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def rollout_worker_loop(
    worker: int,
    worker_count: int,
    seed: int,
    search_depth: int,
    gamma: float,
    max_steps: int,
    episodes_per_snapshot: int,
    policy_queue,
    result_queue,
    stop_event,
) -> None:
    """Continuously produce trajectories while the learner trains."""
    initialize_rollout_worker()
    result_queue.cancel_join_thread()
    device = torch.device("cpu")
    network = ValueNetwork().to(device)
    episode_sequence = 0

    try:
        snapshot = policy_queue.get()
        network.load_state_dict(snapshot.state_dict)
        network.eval()

        while not stop_event.is_set():
            while True:
                try:
                    newer_snapshot = policy_queue.get_nowait()
                except Empty:
                    break
                snapshot = newer_snapshot
            network.load_state_dict(snapshot.state_dict)

            for _ in range(episodes_per_snapshot):
                if stop_event.is_set():
                    return
                episode = worker + 1 + episode_sequence * worker_count
                result = generate_episode(
                    network,
                    episode,
                    seed,
                    snapshot.epsilon,
                    search_depth,
                    gamma,
                    max_steps,
                    device,
                )
                episode_sequence += 1

                while not stop_event.is_set():
                    try:
                        result_queue.put(result, timeout=0.5)
                        break
                    except Full:
                        continue
    except BaseException as error:
        failure = WorkerFailure(worker, str(error), traceback.format_exc())
        while not stop_event.is_set():
            try:
                result_queue.put(failure, timeout=0.5)
                break
            except Full:
                continue


def cpu_state_dict(network: ValueNetwork) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in network.state_dict().items()
    }


def publish_policy(
    network: ValueNetwork,
    epsilon: float,
    policy_queues: list,
) -> None:
    snapshot = PolicySnapshot(cpu_state_dict(network), epsilon)
    for policy_queue in policy_queues:
        policy_queue.put(snapshot)


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return
    return returns


def augment_board(board: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply one of the eight rotations and reflections that preserve value."""
    transformed = np.rot90(board, rng.randrange(4))
    if rng.random() < 0.5:
        transformed = np.fliplr(transformed)
    return np.ascontiguousarray(transformed)


def optimize_batch(
    network: ValueNetwork,
    optimizer: torch.optim.Optimizer,
    replay: Deque[ValueExample],
    config: TrainingConfig,
    rng: random.Random,
    device: torch.device,
) -> float:
    examples = rng.sample(replay, config.batch_size)
    states = np.stack(
        [encode_board(augment_board(example.board, rng)) for example in examples]
    )
    targets = torch.tensor(
        [example.target for example in examples],
        dtype=torch.float32,
        device=device,
    )

    predictions = network(torch.from_numpy(states).to(device))
    loss = F.smooth_l1_loss(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train(
    config: TrainingConfig,
    seed: int,
    device: torch.device,
    search_depth: int,
    checkpoint_path: Path,
    workers: int,
    episodes_per_worker: int,
    policy_sync_every: int,
    rollout_buffer: int,
) -> ValueNetwork:
    """Train while persistent worker processes produce trajectories."""
    if workers < 1:
        raise ValueError("workers must be at least one")
    if episodes_per_worker < 1:
        raise ValueError("episodes per worker must be at least one")
    if policy_sync_every < 1:
        raise ValueError("policy sync interval must be at least one")
    if rollout_buffer < 1:
        raise ValueError("rollout buffer must be at least one")

    replay_rng = random.Random(seed + 2)
    np.random.seed(seed)
    torch.manual_seed(seed)

    network = ValueNetwork().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    replay: Deque[ValueExample] = deque(maxlen=config.replay_capacity)
    recent_scores: Deque[int] = deque(maxlen=100)
    total_steps = 0
    last_loss = 0.0
    report_every = max(1, config.episodes // 20)
    training_started = time.perf_counter()

    print(
        f"Training {network.parameter_count:,} value-network parameters on {device} "
        f"for {config.episodes:,} episodes with depth-{search_depth} expectimax "
        f"and {workers} asynchronous rollout worker{'s' if workers != 1 else ''}"
    )

    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=rollout_buffer)
    policy_queues = [context.Queue() for _ in range(workers)]
    stop_event = context.Event()
    processes = [
        context.Process(
            target=rollout_worker_loop,
            args=(
                worker,
                workers,
                seed,
                search_depth,
                config.gamma,
                config.max_steps_per_episode,
                episodes_per_worker,
                policy_queues[worker],
                result_queue,
                stop_event,
            ),
            name=f"rollout-{worker}",
        )
        for worker in range(workers)
    ]

    publish_policy(network, epsilon_at(total_steps, config), policy_queues)
    for process in processes:
        process.start()

    completed_episodes = 0
    try:
        while completed_episodes < config.episodes:
            try:
                message = result_queue.get(timeout=60.0)
            except Empty:
                failed = [
                    process
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    details = ", ".join(
                        f"{process.name} exited with {process.exitcode}"
                        for process in failed
                    )
                    raise RuntimeError(f"Rollout workers failed: {details}")
                continue

            if isinstance(message, WorkerFailure):
                raise RuntimeError(
                    f"Rollout worker {message.worker} failed: {message.message}\n"
                    f"{message.details}"
                )
            result = message
            completed_episodes += 1

            returns = discounted_returns(result.rewards, config.gamma)
            replay.extend(
                ValueExample(board=state, target=target)
                for state, target in zip(result.boards, returns)
            )
            total_steps += len(result.boards)

            if (
                len(replay) >= max(config.warmup_examples, config.batch_size)
                and result.boards
            ):
                update_count = max(1, len(result.boards) // config.train_every)
                for _ in range(update_count):
                    last_loss = optimize_batch(
                        network,
                        optimizer,
                        replay,
                        config,
                        replay_rng,
                        device,
                    )

            if completed_episodes % policy_sync_every == 0:
                publish_policy(
                    network,
                    epsilon_at(total_steps, config),
                    policy_queues,
                )

            recent_scores.append(result.score)
            if (
                completed_episodes == 1
                or completed_episodes % report_every == 0
                or completed_episodes == config.episodes
            ):
                average_score = sum(recent_scores) / len(recent_scores)
                elapsed = time.perf_counter() - training_started
                episodes_per_second = completed_episodes / max(elapsed, 1e-9)
                print(
                    f"episode {completed_episodes:>5}/{config.episodes}  "
                    f"score {result.score:>6}  avg100 {average_score:>8.1f}  "
                    f"max {result.max_tile:>5}  "
                    f"epsilon {epsilon_at(total_steps, config):.3f}  "
                    f"value-loss {last_loss:.4f}  "
                    f"rate {episodes_per_second:.2f} ep/s"
                )

            if (
                config.checkpoint_every
                and completed_episodes % config.checkpoint_every == 0
            ):
                save_weights(network, checkpoint_path)
    finally:
        stop_event.set()
        shutdown_deadline = time.monotonic() + 10.0
        for process in processes:
            process.join(timeout=max(0.0, shutdown_deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        for policy_queue in policy_queues:
            policy_queue.cancel_join_thread()
            policy_queue.close()
        result_queue.cancel_join_thread()
        result_queue.close()

    return network


def save_weights(network: ValueNetwork, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(network.state_dict(), temporary_path)
    temporary_path.replace(path)
    print(f"Saved value weights to {path}")


def load_weights(path: Path, device: torch.device) -> ValueNetwork:
    network = ValueNetwork().to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    try:
        network.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(
            f"{path} does not contain weights for the current value network"
        ) from error
    network.eval()
    return network


def play(
    network: ValueNetwork,
    games: int,
    seed: int,
    delay: float,
    device: torch.device,
    search_depth: int,
    gamma: float,
) -> None:
    """Play complete games using exact chance nodes and neural leaf values."""
    rng = random.Random(seed)
    network.eval()

    for game_number in range(1, games + 1):
        board = new_game(rng)
        score = 0
        move_count = 0
        print(f"\nGame {game_number}")
        bprint(board)

        while not game_over(board):
            action, values = choose_action(
                network,
                board,
                search_depth,
                gamma,
                epsilon=0.0,
                rng=rng,
                device=device,
            )
            board, merge_score, terminal = step(board, action, rng)
            score += merge_score
            move_count += 1
            value_text = "  ".join(
                f"{candidate}={value:.2f}"
                for candidate, value in values.items()
            )
            print(
                f"move {move_count}: {action}  +{merge_score}  "
                f"[{value_text}]"
            )
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


def default_worker_count() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and play 2048 with neural value-guided expectimax"
    )
    parser.add_argument("--episodes", type=int, default=2_500)
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="CPU processes used to generate self-play episodes",
    )
    parser.add_argument(
        "--episodes-per-worker",
        type=int,
        default=1,
        help="episodes generated before a worker checks for newer weights",
    )
    parser.add_argument(
        "--policy-sync-every",
        type=int,
        default=0,
        help="consumed episodes between weight broadcasts; zero uses worker count",
    )
    parser.add_argument(
        "--rollout-buffer",
        type=int,
        default=0,
        help="maximum queued episodes; zero uses twice the worker count",
    )
    parser.add_argument(
        "--play-only",
        action="store_true",
        help="load existing value weights instead of training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.search_depth < 1:
        raise SystemExit("Search depth must be at least one")
    if args.workers < 1:
        raise SystemExit("Workers must be at least one")
    if args.episodes_per_worker < 1:
        raise SystemExit("Episodes per worker must be at least one")
    if args.policy_sync_every < 0:
        raise SystemExit("Policy sync interval cannot be negative")
    if args.rollout_buffer < 0:
        raise SystemExit("Rollout buffer cannot be negative")
    device = select_device(args.device)

    if args.play_only:
        if not args.weights.exists():
            raise SystemExit(f"Weights do not exist: {args.weights}")
        try:
            network = load_weights(args.weights, device)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        policy_sync_every = args.policy_sync_every or args.workers
        rollout_buffer = args.rollout_buffer or args.workers * 2
        config = TrainingConfig(
            episodes=args.episodes,
            checkpoint_every=args.checkpoint_every,
        )
        network = train(
            config,
            args.seed,
            device,
            args.search_depth,
            args.weights,
            args.workers,
            args.episodes_per_worker,
            policy_sync_every,
            rollout_buffer,
        )
        save_weights(network, args.weights)

    play(
        network,
        args.games,
        args.seed + 1,
        args.delay,
        device,
        args.search_depth,
        gamma=0.99,
    )


if __name__ == "__main__":
    main()
