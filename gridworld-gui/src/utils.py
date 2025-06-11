def save_grid_configuration(grid, filename):
    with open(filename, 'w') as file:
        for row in grid:
            file.write(','.join(map(str, row)) + '\n')

def load_grid_configuration(filename):
    grid = []
    with open(filename, 'r') as file:
        for line in file:
            row = list(map(int, line.strip().split(',')))
            grid.append(row)
    return grid

def grid_to_obstacle_set(grid):
    obstacle_cells = set()
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 1:  # Assuming 1 represents an obstacle
                obstacle_cells.add((i, j))
    return obstacle_cells