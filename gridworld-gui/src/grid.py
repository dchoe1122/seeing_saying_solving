class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.obstacle_cells = set()

    def add_obstacle(self, row, col):
        """Add an obstacle at the specified position"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.obstacle_cells.add((row, col))
            self.grid[row][col] = 1

    def remove_obstacle(self, cell):
        if cell in self.obstacle_cells:
            self.obstacle_cells.remove(cell)
            self.free_cells.append(cell)

    def save_to_file(self, filename):
        """Save the grid configuration to a file with dimensions"""
        with open(filename, 'w') as f:
            # First line contains grid dimensions
            f.write(f"{self.rows},{self.cols}\n")
            # Following lines contain obstacle coordinates
            for cell in self.obstacle_cells:
                f.write(f"{cell[0]},{cell[1]}\n")

    def load_configuration(self, filename):
        """Load grid configuration including dimensions"""
        self.obstacle_cells.clear()
        with open(filename, 'r') as f:
            # First line contains dimensions
            rows, cols = map(int, f.readline().strip().split(','))
            if rows != self.rows or cols != self.cols:
                self.rows = rows
                self.cols = cols
                self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
            
            # Read obstacle coordinates
            for line in f:
                row, col = map(int, line.strip().split(','))
                self.add_obstacle(row, col)