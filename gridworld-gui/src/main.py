from tkinter import Tk, Canvas, Button, simpledialog, messagebox
from grid import Grid
from gui import GridWorldGUI

class GridWorldGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Grid World Configuration")
        
        # Ask for rows and columns separately
        self.rows = simpledialog.askinteger("Input", "Enter number of rows:", minvalue=1)
        self.cols = simpledialog.askinteger("Input", "Enter number of columns:", minvalue=1)
        
        if self.rows is None or self.cols is None:
            self.master.destroy()
            return
            
        self.grid = Grid(self.rows, self.cols)
        
        self.canvas = Canvas(master, width=500, height=500)
        self.canvas.pack()
        
        self.draw_grid()
        
        self.save_button = Button(master, text="Save Configuration", command=self.save_configuration)
        self.save_button.pack()
        
        self.canvas.bind("<Button-1>", self.add_obstacle)

    def draw_grid(self):
        self.canvas.delete("all")
        # Calculate cell size based on both dimensions
        cell_size = min(500 // self.rows, 500 // self.cols)
        canvas_width = cell_size * self.cols
        canvas_height = cell_size * self.rows
        
        # Update canvas size
        self.canvas.config(width=canvas_width, height=canvas_height)
        
        for i in range(self.rows):
            for j in range(self.cols):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                color = "white" if (i, j) not in self.grid.obstacle_cells else "black"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

    def add_obstacle(self, event):
        cell_size = min(500 // self.rows, 500 // self.cols)
        x = event.x // cell_size
        y = event.y // cell_size
        if 0 <= y < self.rows and 0 <= x < self.cols:
            if (y, x) in self.grid.obstacle_cells:
                self.grid.obstacle_cells.remove((y, x))
            else:
                self.grid.add_obstacle(y, x)
            self.draw_grid()

    def save_configuration(self):
        try:
            self.grid.save_to_file("grid_configuration.txt")
            messagebox.showinfo("Success", "Grid configuration saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

if __name__ == "__main__":
    root = Tk()
    app = GridWorldGUI(root)
    root.mainloop()