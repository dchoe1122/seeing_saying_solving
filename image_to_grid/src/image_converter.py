from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors

class ImageToGrid:
    def __init__(self, image_path, max_cells=500, threshold=0.95):
        self.image_path = image_path
        self.max_cells = max_cells
        self.threshold = threshold
        self.grid_rows = 0
        self.grid_cols = 0
        self.grid = None
        self.obstacle_cells = set()

    def process_image(self):
        # Load and convert image to grayscale
        img = Image.open(self.image_path).convert('L')
        width, height = img.size

        # Calculate appropriate grid size
        aspect_ratio = width / height
        max_grid_dim = math.sqrt(self.max_cells)
        
        if (aspect_ratio > 1):
            self.grid_cols = min(int(max_grid_dim), width)
            self.grid_rows = int(self.grid_cols / aspect_ratio)
        else:
            self.grid_rows = min(int(max_grid_dim), height)
            self.grid_cols = int(self.grid_rows * aspect_ratio)

        # Ensure we don't exceed max cells
        while self.grid_rows * self.grid_cols > self.max_cells:
            self.grid_rows = int(self.grid_rows * 0.9)
            self.grid_cols = int(self.grid_cols * 0.9)

        # Resize image using version-agnostic resampling
        try:
            # For newer Pillow versions
            img_resized = img.resize((self.grid_cols, self.grid_rows), Image.Resampling.LANCZOS)
        except AttributeError:
            # For older Pillow versions
            img_resized = img.resize((self.grid_cols, self.grid_rows), Image.LANCZOS)

        img_array = np.array(img_resized)

        # Calculate cell size in original image
        cell_height = height / self.grid_rows
        cell_width = width / self.grid_cols

        # Create grid
        self.grid = [[0 for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        
        # Process each grid cell
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                # If average pixel value is less than 128 (middle of 0-255),
                # and more than threshold% of pixels are black, mark as obstacle
                if img_array[i, j] < 128:
                    self.grid[i][j] = 1
                    self.obstacle_cells.add((i, j))

    def save_configuration(self, output_path="grid_configuration.txt"):
        with open(output_path, 'w') as f:
            # Write grid dimensions
            f.write(f"{self.grid_rows},{self.grid_cols}\n")
            # Write obstacle coordinates with flipped y-coordinates
            for cell in self.obstacle_cells:
                # Flip the y-coordinate (row) by subtracting from grid_rows - 1
                # flipped_row = (self.grid_rows - 1) - cell[0]
                flipped_row = cell[0]  # No flipping needed for this example
                f.write(f"{flipped_row},{cell[1]}\n")

    def plot_configuration(self, show=True, save_path=None):
        """
        Plots the current grid configuration using matplotlib.
        
        Args:
            show (bool): Whether to display the plot
            save_path (str): Optional path to save the plot image
        """
        # Create base grid plot data
        grid_plot = np.zeros((self.grid_rows, self.grid_cols))
        for (i, j) in self.obstacle_cells:
            grid_plot[i][j] = 1  # Obstacles marked as 1

        fig, ax = plt.subplots(figsize=(8, 12))
        
        # Custom colormap: white for free cells, gray for obstacles
        cmap = colors.ListedColormap(['white', 'gray'])
        
        # Plot the grid
        ax.imshow(grid_plot, cmap=cmap, origin='lower', aspect='auto')
        
        # Set proper grid dimensions
        ax.set_xlim(-0.5, self.grid_cols - 0.5)
        ax.set_ylim(-0.5, self.grid_rows - 0.5)
        
        # Configure ticks
        ax.set_xticks(range(self.grid_cols))
        ax.set_yticks(range(self.grid_rows))
        ax.set_xticklabels(range(self.grid_cols))
        ax.set_yticklabels(range(self.grid_rows))
        
        # Add grid lines
        ax.grid(True)

        # Add legend
        legend_elements = [
            Patch(facecolor='white', edgecolor='gray', label='Free Cell'),
            Patch(facecolor='gray', edgecolor='gray', label='Obstacle')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=14, frameon=True, labelspacing=1.0)

        plt.title(f"Grid Configuration ({self.grid_rows}x{self.grid_cols})", 
                 fontsize=14)
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            
        if show:
            plt.show()
            
        plt.close(fig)

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python image_converter.py <image_path>")
        return

    converter = ImageToGrid(sys.argv[1])
    converter.process_image()
    converter.save_configuration()
    converter.plot_configuration(save_path="grid_visualization.png")
    print(f"Grid configuration saved with dimensions: {converter.grid_rows}x{converter.grid_cols}")

if __name__ == "__main__":
    main()