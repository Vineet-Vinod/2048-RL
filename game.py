from __future__ import annotations
from random import randint
from typing import List

def reward(board: List[List[int]]):
    return sum(sum(row) for row in board)

def move(board: List[List[int]], dir: str):
    if all(all(row) for row in board): return -2, board # Game over
    ret = [row[:] for row in board]
    match dir:
        case "U":
            for st in range(1, 4):
                for col in range(4):
                    s = st
                    if ret[s][col]:
                        for d in range(s-1, -1, -1):
                            if ret[d][col]:
                                if ret[d][col] == ret[s][col]:
                                    ret[d][col] += 1
                                    ret[s][col] = 0
                                break
                            ret[d][col], ret[s][col] = ret[s][col], 0
                            s = d
        case "D":
            for st in range(2, -1, -1):
                for col in range(4):
                    s = st
                    if ret[s][col]:
                        for d in range(s+1, 4):
                            if ret[d][col]:
                                if ret[d][col] == ret[s][col]:
                                    ret[d][col] += 1
                                    ret[s][col] = 0
                                break
                            ret[d][col], ret[s][col] = ret[s][col], 0
                            s = d
        case "L":
            for st in range(1, 4):
                for row in range(4):
                    s = st
                    if ret[row][s]:
                        for d in range(s-1, -1, -1):
                            if ret[row][d]:
                                if ret[row][d] == ret[row][s]:
                                    ret[row][d] += 1
                                    ret[row][s] = 0
                                break
                            ret[row][d], ret[row][s] = ret[row][s], 0
                            s = d
        case "R":
            for st in range(2, -1, -1):
                for row in range(4):
                    s = st
                    if ret[row][s]:
                        for d in range(s+1, 4):
                            if ret[row][d]:
                                if ret[row][d] == ret[row][s]:
                                    ret[row][d] += 1
                                    ret[row][s] = 0
                                break
                            ret[row][d], ret[row][s] = ret[row][s], 0
                            s = d

    if ret == board: return -1, board # Invalid move (nothing happpens)
    while True:
        sqr = randint(0, 15)
        if not ret[sqr >> 2][sqr & 3]:
            ret[sqr >> 2][sqr & 3] = randint(1,2)
            break
    return 0, ret # Valid move

def bprint(board: List[List[int]]):
    print("|" + "-" * 15 + "|")
    for row in board:
        print(f"|{row[0]:3}|{row[1]:3}|{row[2]:3}|{row[3]:3}|")
        print("|" + "-" * 15 + "|")

if __name__ == "__main__":
    board = [[0] * 4 for _ in range(4)]
    board[1][1] = 1
    bprint(board)
    while True:
        mv = input("> ")
        if mv not in "UDLR" or len(mv) != 1: raise("Move must be U,D,L,R")
        r, board = move(board, mv)
        if r == -2: raise("Game over")
        bprint(board)
