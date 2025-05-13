import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot(grid, requester_stl, agent, original_results, cost_optimal_results, save_as_file=False,
         filename_prefix='agent_paths', save_dir='Scenario 1 tests'):

    # Unpack results
    requester_status, requester_path, requester_t_earliest, _, _ = requester_stl
    original_status, original_path, original_t_earliest, _, original_solve_time = original_results        
    cost_optimal_status, cost_optimal_path, cost_optimal_t_earliest, _, cost_optimal_solve_time = cost_optimal_results

    # Truncate paths based on t_earliest
    requester_path = requester_path[:int(requester_t_earliest) + 1]
    original_path = original_path[:int(original_t_earliest) + 1]
    cost_optimal_path = cost_optimal_path[:int(cost_optimal_t_earliest) + 1]

    # Prepare base grid plotting data
    grid_plot = np.zeros((grid.grid_size, grid.grid_size))
    for (i, j) in grid.obstacle_cells:
        grid_plot[i][j] = 1  # Obstacles
    k, l = grid.conflict_cell
    grid_plot[k][l] = 2      # Conflict cell
    start_pos = agent.initial_pos
    help_drop_k, help_drop_l = agent.ltl_locs['help_site_drop']
    grid_plot[help_drop_k][help_drop_l] = 5
    for (i, j) in agent.pickups:
        grid_plot[i][j] = 3
    for (i, j) in agent.dropoffs:
        grid_plot[i][j] = 4

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = colors.ListedColormap(['white', 'gray', "orange", "green", "purple", "orange"])
    ax.imshow(grid_plot, cmap=cmap, origin='lower')

    def annotate_path(ax, path, color, direction='left'):
        offset = 0.15
        sign = 1 if direction == 'left' else -1
        for ((x0, y0), (x1, y1)) in zip(path[:-1], path[1:]):
            dx, dy = (y1 - y0), (x1 - x0)
            norm = np.hypot(dx, dy) or 1
            dx_norm, dy_norm = dx / norm, dy / norm
            perp_dx = sign * (-dy_norm * offset)
            perp_dy = sign * (dx_norm * offset)
            ax.annotate('',
                        xy=(y1 + perp_dx, x1 + perp_dy),
                        xytext=(y0 + perp_dx, x0 + perp_dy),
                        arrowprops=dict(facecolor=color, edgecolor=color, shrink=0.175,
                                        width=3.0, headwidth=10))

    # Plot original and updated trajectories
    if original_status == 2:
        annotate_path(ax, original_path, 'red', direction='left')
    if cost_optimal_status == 2:
        annotate_path(ax, cost_optimal_path, 'blue', direction='right')

    # Mark the start
    ax.plot(start_pos[1], start_pos[0], marker='*', color='red', markersize=15)

    # Add hatched help drop cell
    rect = plt.Rectangle((help_drop_l - 0.5, help_drop_k - 0.5), 1, 1,
                         fill=True, facecolor='orange', hatch='//', edgecolor='grey')
    ax.add_patch(rect)

    # Grid styling
    ax.set_xticks(range(grid.grid_size))
    ax.set_yticks(range(grid.grid_size))
    ax.set_xticklabels(range(grid.grid_size))
    ax.set_yticklabels(range(grid.grid_size))
    ax.grid(True)

    # Legend
    legend_elements = [
        Patch(facecolor='gray', edgecolor='gray', label='Obstacle'),
        Patch(facecolor='orange', edgecolor='orange', label='Help Site'),
        Patch(facecolor='orange', edgecolor='grey', hatch='//', label='Adjacent free cell'),
        Patch(facecolor='green', edgecolor='green', label='Pallet Pickup'),
        Patch(facecolor='purple', edgecolor='purple', label='Pallet Dropoff'),
        Line2D([0], [0], color='red', marker=r'$\rightarrow$', linestyle='None', markersize=14, label='Original Path'),
        Line2D([0], [0], color='blue', marker=r'$\rightarrow$', linestyle='None', markersize=14, label='Updated Path')
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=14, frameon=True, labelspacing=1.0)


    plt.title(f"Gridworld Paths: Original vs Updated Trajectories\nWorld Seed: {grid.seed}", fontsize=14)
    plt.tight_layout()

    if save_as_file:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(
            save_dir, f"SEED_{grid.seed}_CASE1_{filename_prefix}_agent_{agent.id}_world_seed_{grid.seed}.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

