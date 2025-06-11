from tkinter import Label, Entry, Button, Canvas, Frame

class GridWorldGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Grid World Configuration")
        
        # Create input frame
        input_frame = Frame(master)
        input_frame.pack(pady=5)
        
        # Row input
        self.row_label = Label(input_frame, text="Rows:")
        self.row_label.pack(side='left', padx=5)
        self.row_entry = Entry(input_frame, width=5)
        self.row_entry.pack(side='left', padx=5)
        
        # Column input
        self.col_label = Label(input_frame, text="Columns:")
        self.col_label.pack(side='left', padx=5)
        self.col_entry = Entry(input_frame, width=5)
        self.col_entry.pack(side='left', padx=5)

        self.create_button = Button(master, text="Create Grid", command=self.create_grid)
        self.create_button.pack(pady=5)

        self.canvas = Canvas(master)
        self.canvas.pack(pady=5)

        self.save_button = Button(master, text="Save Configuration", command=self.save_configuration)
        self.save_button.pack(pady=5)

        self.grid = None
        self.obstacle_cells = set()
        self.rows = 0
        self.cols = 0


    def create_grid(self):
        try:
            self.rows = int(self.row_entry.get())
            self.cols = int(self.col_entry.get())
            
            if self.rows <= 0 or self.cols <= 0:
                print("Rows and columns must be positive numbers")
                return
                
            self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
            
            # Calculate cell size to fit the window (using min to maintain square cells)
            window_size = 500
            cell_size = min(window_size // self.rows, window_size // self.cols)
            canvas_width = cell_size * self.cols
            canvas_height = cell_size * self.rows
            
            self.canvas.config(width=canvas_width, height=canvas_height)
            self.canvas.bind("<Button-1>", self.add_obstacle)
            self.obstacle_cells.clear()
            self.draw_grid()
            
        except ValueError:
            print("Please enter valid numbers for rows and columns")

    def draw_grid(self):
        self.canvas.delete("all")
        cell_size = min(self.canvas.winfo_width() // self.cols, 
                       self.canvas.winfo_height() // self.rows)
                       
        for i in range(self.rows):
            for j in range(self.cols):
                # Calculate positions with x starting from left
                x1 = j * cell_size
                y1 = (self.rows - 1 - i) * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                # Check for obstacles with correct coordinate system
                color = "white" if (i, self.cols - 1 - j) not in self.obstacle_cells else "black"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

    def add_obstacle(self, event):
        if not self.grid:
            return
            
        cell_size = min(self.canvas.winfo_width() // self.cols, 
                       self.canvas.winfo_height() // self.rows)
        # Calculate x coordinate starting from left
        x = event.x // cell_size
        # Keep y coordinate calculation for bottom-left origin
        y = event.y // cell_size
        
        if 0 <= y < self.rows and 0 <= x < self.cols:
            if (y, x) in self.obstacle_cells:
                self.obstacle_cells.remove((y, x))
            else:
                self.obstacle_cells.add((y, x))
            self.draw_grid()

    def save_configuration(self):
        with open("grid_configuration.txt", "w") as f:
            # First line: grid dimensions
            f.write(f"{self.rows},{self.cols}\n")
            # Following lines: obstacle coordinates
            for cell in self.obstacle_cells:
                f.write(f"{cell[0]},{cell[1]}\n")