from __future__ import annotations
from game import bprint, game_over, heuristic, move, slide, spawn_outcomes
from math import inf
from typing import List

gamma = 0.7
depth = 3

memo = {}
bmemo = {}

def bellman(board: List[List[int]], depth):
    if game_over(board): return 0
    if depth == 0: return heuristic(board)
    key = (tuple(val for row in board for val in row), depth)
    if key not in memo:
        brew = -inf
        bmove = '~'
        for mv in "UDLR":
            moved, merge_score = slide(board, mv)
            if moved == board:
                continue
            expected_future = sum(
                probability * bellman(nboard, depth - 1)
                for probability, nboard in spawn_outcomes(moved)
            )
            rew = merge_score + gamma * expected_future
            if brew < rew:
                brew = rew
                bmove = mv
        if bmove == '~': return 0
        memo[key] = brew
        bmemo[key] = bmove
    return memo[key]

if __name__ == "__main__":
    board = [[0] * 4 for _ in range(4)]
    board[1][1] = 1
    while True:
        bprint(board)
        bellman(board, depth)
        key = (tuple(val for row in board for val in row), depth)
        if key not in bmemo:
            print(f"Final board value: {heuristic(board):.2f}")
            raise SystemExit("Game over")
        r, board = move(board, bmemo[key])
        if r == -2 or game_over(board):
            print(f"Final board value: {heuristic(board):.2f}")
            raise SystemExit("Game over")
        memo.clear()
        bmemo.clear()
