from __future__ import annotations

import random
from itertools import pairwise
from typing import List


Board = List[List[int]]
DIRECTIONS = "UDLR"
EMPTY_WEIGHT = 2.7
MONOTONICITY_WEIGHT = 1.0
SMOOTHNESS_WEIGHT = 0.1
MAX_TILE_WEIGHT = 1.0
CORNER_WEIGHT = 2.0


def _smoothness(board: Board) -> int:
    """Penalize large jumps between neighboring nonempty tiles."""
    score = 0
    for row in range(4):
        for col in range(4):
            tile = board[row][col]
            if not tile:
                continue

            for next_col in range(col + 1, 4):
                neighbor = board[row][next_col]
                if neighbor:
                    score -= abs(tile - neighbor)
                    break

            for next_row in range(row + 1, 4):
                neighbor = board[next_row][col]
                if neighbor:
                    score -= abs(tile - neighbor)
                    break
    return score


def _monotonicity(board: Board) -> int:
    """Score boards whose values consistently rise or fall along each axis."""
    left_to_right = 0
    right_to_left = 0
    top_to_bottom = 0
    bottom_to_top = 0

    for row in board:
        for current, following in pairwise(row):
            if current > following:
                left_to_right += following - current
            elif following > current:
                right_to_left += current - following

    for col in range(4):
        column = [board[row][col] for row in range(4)]
        for current, following in pairwise(column):
            if current > following:
                top_to_bottom += following - current
            elif following > current:
                bottom_to_top += current - following

    return max(left_to_right, right_to_left) + max(top_to_bottom, bottom_to_top)


def heuristic(board: Board) -> float:
    """Estimate how promising a board is at the search depth limit."""
    empty_count = len(empty_cells(board))
    max_tile = max((tile for row in board for tile in row), default=0)
    corners = (board[0][0], board[0][3], board[3][0], board[3][3])
    corner_bonus = max_tile if max_tile and max_tile in corners else 0

    return (
        EMPTY_WEIGHT * empty_count
        + MONOTONICITY_WEIGHT * _monotonicity(board)
        + SMOOTHNESS_WEIGHT * _smoothness(board)
        + MAX_TILE_WEIGHT * max_tile
        + CORNER_WEIGHT * corner_bonus
    )


def game_over(board: Board) -> bool:
    for row in board:
        for a, b in pairwise(row):
            if a == b:
                return False
            if not a or not b:
                return False
    for col in range(4):
        for row in range(3):
            if board[row][col] == board[row + 1][col]:
                return False
            if not board[row][col] or not board[row + 1][col]:
                return False
    return True


def _merge_line(line: List[int]) -> tuple[List[int], int]:
    """Slide one line left and merge each tile at most once."""
    tiles = [tile for tile in line if tile]
    merged = []
    score = 0
    index = 0

    while index < len(tiles):
        if index + 1 < len(tiles) and tiles[index] == tiles[index + 1]:
            exponent = tiles[index] + 1
            merged.append(exponent)
            score += 2**exponent
            index += 2
        else:
            merged.append(tiles[index])
            index += 1

    return merged + [0] * (4 - len(merged)), score


def slide(board: Board, direction: str) -> tuple[Board, int]:
    """Apply a direction without spawning a tile.

    Board entries are exponents, so 1 represents 2, 2 represents 4, and so on.
    The returned score contains only points earned by merges in this slide.
    """
    if direction not in DIRECTIONS or len(direction) != 1:
        raise ValueError("direction must be one of U, D, L, or R")

    result = [[0] * 4 for _ in range(4)]
    score = 0

    for index in range(4):
        if direction in "LR":
            line = board[index][:]
        else:
            line = [board[row][index] for row in range(4)]

        if direction in "DR":
            line.reverse()

        merged, line_score = _merge_line(line)
        score += line_score

        if direction in "DR":
            merged.reverse()

        if direction in "LR":
            result[index] = merged
        else:
            for row, tile in enumerate(merged):
                result[row][index] = tile

    return result, score


def empty_cells(board: Board) -> List[tuple[int, int]]:
    return [
        (row, col)
        for row in range(4)
        for col in range(4)
        if board[row][col] == 0
    ]


def spawn_tile(board: Board, rng: random.Random | None = None) -> Board:
    """Return a copy with one sampled tile added using standard 2048 odds."""
    cells = empty_cells(board)
    if not cells:
        return [row[:] for row in board]

    rng = rng or random
    row, col = rng.choice(cells)
    result = [line[:] for line in board]
    result[row][col] = 1 if rng.random() < 0.9 else 2
    return result


def new_game(rng: random.Random | None = None) -> Board:
    """Create a standard starting board with two independently sampled tiles."""
    board = [[0] * 4 for _ in range(4)]
    board = spawn_tile(board, rng)
    return spawn_tile(board, rng)


def legal_actions(board: Board) -> List[str]:
    """Return directions that change the board before a random tile spawns."""
    return [
        direction
        for direction in DIRECTIONS
        if slide(board, direction)[0] != board
    ]


def step(
    board: Board,
    direction: str,
    rng: random.Random | None = None,
) -> tuple[Board, int, bool]:
    """Sample one environment transition for model-free training."""
    moved, score = slide(board, direction)
    if moved == board:
        raise ValueError(f"illegal action {direction!r} for this board")

    next_board = spawn_tile(moved, rng)
    return next_board, score, game_over(next_board)


def spawn_outcomes(board: Board) -> List[tuple[float, Board]]:
    """Enumerate every random spawn and its probability for Bellman search."""
    cells = empty_cells(board)
    if not cells:
        return [(1.0, [row[:] for row in board])]

    outcomes = []
    for row, col in cells:
        for exponent, tile_probability in ((1, 0.9), (2, 0.1)):
            next_board = [line[:] for line in board]
            next_board[row][col] = exponent
            outcomes.append((tile_probability / len(cells), next_board))
    return outcomes


def move(
    board: Board,
    direction: str,
    rng: random.Random | None = None,
) -> tuple[int, Board]:
    """Play one complete turn, preserving the original status-code interface."""
    if game_over(board):
        return -2, board

    moved, _ = slide(board, direction)
    if moved == board:
        return -1, board

    return 0, spawn_tile(moved, rng)


def bprint(board: Board) -> None:
    print("|" + "-" * 15 + "|")
    for row in board:
        print(f"|{row[0]:3}|{row[1]:3}|{row[2]:3}|{row[3]:3}|")
        print("|" + "-" * 15 + "|")


if __name__ == "__main__":
    board = [[0] * 4 for _ in range(4)]
    board[1][1] = 1
    bprint(board)
    while True:
        direction = input("> ")
        try:
            status, board = move(board, direction)
        except ValueError as error:
            print(error)
            continue
        if status == -2:
            raise SystemExit("Game over")
        bprint(board)
