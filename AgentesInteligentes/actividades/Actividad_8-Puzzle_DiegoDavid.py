import random
from heapq import heappop, heappush
from copy import deepcopy

def print_menu():
    print("Welcome to Puzzle Solver")
    print("--- MENU ---")
    print("[1] Start solving the puzzle")
    print("[0] Exit")

def find_zero(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                return (i, j)
    return None

def possible_moves(zero_pos, matrix):
    row, col = zero_pos
    moves = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]):
            moves.append((new_row, new_col))
    return moves

def swap_positions(matrix, pos1, pos2):
    new_matrix = deepcopy(matrix)
    new_matrix[pos1[0]][pos1[1]], new_matrix[pos2[0]][pos2[1]] = new_matrix[pos2[0]][pos2[1]], new_matrix[pos1[0]][pos1[1]]
    return new_matrix

def manhattan_distance(matrix):
    target = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (2, 0), 8: (2, 1), 0: (2, 2)}
    distance = 0
    for i in range(3):
        for j in range(3):
            number = matrix[i][j]
            if number != 0:
                target_pos = target[number]
                distance += abs(target_pos[0] - i) + abs(target_pos[1] - j)
    return distance

def solve_puzzle(start_matrix):
    goal = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
    open_set = [(manhattan_distance(start_matrix), start_matrix, [])]
    visited = set()

    while open_set:
        current_distance, current_matrix, path = heappop(open_set)
        if current_matrix == goal:
            return path

        if str(current_matrix) in visited:
            continue
        visited.add(str(current_matrix))

        zero_pos = find_zero(current_matrix)
        for move in possible_moves(zero_pos, current_matrix):
            new_matrix = swap_positions(current_matrix, zero_pos, move)
            new_path = path + [new_matrix]
            heappush(open_set, (manhattan_distance(new_matrix) + len(new_path), new_matrix, new_path))

    return None

def main():
    print_menu()
    choice = int(input("Select an option: "))
    while choice != 0:
        if choice == 1:
            initial_state = [[2, 8, 3], [1, 6, 4], [7, 0, 5]]
            print("Initial state of the puzzle:")
            for row in initial_state:
                print(row)
            solution = solve_puzzle(initial_state)
            if solution:
                print("\nSolution steps:")
                for step in solution:
                    for row in step:
                        print(row)
                    print()
            else:
                print("No solution found.")
        print_menu()
        choice = int(input("Select an option: "))
    print("Goodbye")

if __name__ == "__main__":
    main()