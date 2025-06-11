from robot import GridWorldAgent
from GridWorld import GridWorld
from utils import (
    compute_distance,
    start_pos_init,
    get_ltl_spec,
    parse_formula_string,
    parse_pickup_dropoff_with_coords,
    mobile_robot_stl,
    forklift_stl_generation,
    save_results_to_csv
)
from plot import plot
import random
import os
from datetime import datetime


#Number of seeds to test
seed_iterations = 1
    
# Create timestamped directory
timestamp = datetime.now().strftime("%Y-%m-%d")
save_dir = f'Scenario 1 tests_{timestamp}'
os.makedirs(save_dir, exist_ok=True)

#Run experiment for each seed
for i in range(seed_iterations):
    seed = 7400 + i*100
    world = GridWorld(grid_size_rows=10, grid_size_cols=6, seed=seed)

    #number of helper agents to compare for each experiment
    num_helper_agents_feasible=1
    agents = []
    results = []
    requester = GridWorldAgent(
        ltl_locs=get_ltl_spec(grid=world, num_existing_locs=3),
        capabilities=['move'],
        initial_pos=world.conflict_cell,
        agent_type="mobile_robot",
        gridworld=world,
        needs_help=False,
        T=25
    )

    # Solving the Requester (Mobile Robot Solution)
    llm_stl_expression_req = mobile_robot_stl(requester.ltl_locs)
    original_formula_req = parse_formula_string(llm_stl_expression_req)
    original_results_req = requester.solve_physical_visits(original_formula_req,optimize_tstar=True)

    #get only feasible solutions
    i = 0  # Number of successfully found agents with feasible pairs

    while i < num_helper_agents_feasible:
        attempts = 0
        max_attempts = 10
        found_pair = False

        while not found_pair and attempts < max_attempts:
            initial_pos = start_pos_init(world)

            # Create a helper agent with random initial position
            helper = GridWorldAgent(
                ltl_locs=get_ltl_spec(grid=world, num_existing_locs=4),
                capabilities=["lift pallet", "move"],
                initial_pos=initial_pos,
                agent_type='forklift',
                has_pallet=1 if random.random() < 0.5 else 0,
                gridworld=world,
                needs_help=False,
                T=30
            )

            llm_stl_expression_help = forklift_stl_generation(helper.ltl_locs)
            print(f"Forklift agent_{i}, attempt {attempts + 1}, initial position: {initial_pos}, spec: {llm_stl_expression_help}")

            helper.pickups, helper.dropoffs = parse_pickup_dropoff_with_coords(llm_stl_expression_help, helper.ltl_locs)
            original_formula_help = parse_formula_string(llm_stl_expression_help)
            original_results_help = helper.solve_physical_visits(original_formula_help)
            distance_from_help_site = compute_distance(world.conflict_cell, helper.initial_pos)

            if original_results_help[0] == 2:
                print(f"Feasible original position found for helper_{i}")
                updated_formula_help = parse_formula_string(f"F(IMPLIES_NEXT(help_site,help_site_drop))&({llm_stl_expression_help})")
                updated_results_help = helper.solve_physical_visits(updated_formula_help, optimize_cost=True, is_original=False)

                if updated_results_help[0] == 2:
                    print(f"Updated solution found for helper_{helper.id - 1}")
                    results.append({
                        'reqeuster_stl': original_results_req,
                        'agent': helper,
                        'original_results_help': original_results_help,
                        'updated_results_help': updated_results_help,
                        'distance_from_help_site': distance_from_help_site
                    })

                    plot(
                        grid=world,
                        requester_stl=original_results_req,
                        agent=helper,
                        original_results=original_results_help,
                        cost_optimal_results=updated_results_help,
                        save_as_file=True,
                        save_dir=save_dir)
                    

                    found_pair = True
                    i += 1  # Successfully found one feasible pair
                else:
                    attempts += 1
                    print(f"Updated solution infeasible for helper {helper.id - 1}, retrying...")
            else:
                attempts += 1
                print(f"Original solution infeasible for attempt {attempts}, retrying...")

        if not found_pair:
            print(f"Failed to find feasible pair for agent_{i} after {max_attempts} attempts.")

    # save results to CSV
    save_results_to_csv(
        grid=world,
        requester_stl=original_results_req,
        agent_results_list=results,
        filename_prefix='agent_results',
        save_dir=save_dir
    )

def plot(grid, requester_stl, agent, original_results, cost_optimal_results, save_as_file=False,
         filename_prefix='agent_paths', save_dir='Scenario 1 tests'):

    # Use grid.grid_size_rows and grid.grid_size_cols for rectangular grids
    rows = getattr(grid, 'grid_size_rows', getattr(grid, 'grid_size', None))
    cols = getattr(grid, 'grid_size_cols', getattr(grid, 'grid_size', None))
    if rows is None or cols is None:
        raise ValueError("Grid object must have grid_size_rows and grid_size_cols attributes for rectangular grids.")

    # Unpack results
    requester_status, requester_path, requester_t_earliest, _, _ = requester_stl
    original_status, original_path, original_t_earliest, _, original_solve_time = original_results        
    cost_optimal_status, cost_optimal_path, cost_optimal_t_earliest, _, cost_optimal_solve_time = cost_optimal_results

    # Truncate paths based on t_earliest
    requester_path = requester_path[:int(requester_t_earliest) + 1]
    original_path = original_path[:int(original_t_earliest) + 1]
    cost_optimal_path = cost_optimal_path[:int(cost_optimal_t_earliest) + 1]

    # Prepare base grid plotting data
    grid_plot = np.zeros((rows, cols))
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
    

    fig, ax = plt.subplots(figsize=(8,12))
    cmap = colors.ListedColormap(['white', 'gray', "orange", "green", "purple", "orange"])
    ax.imshow(grid_plot, cmap=cmap, origin='lower', aspect='auto')
    ax.set_xlim(-0.5, cols - 0.5)

    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows))

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
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows))
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

    plt.title(f"Gridworld Paths: Original vs Updated Trajectories\nWorld Seed: {getattr(grid, 'seed', 'N/A')}", fontsize=14)
    plt.tight_layout()

    if save_as_file:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(
            save_dir, f"SEED_{getattr(grid, 'seed', 'N/A')}_CASE1_{filename_prefix}_agent_{agent.id}_world_seed_{getattr(grid, 'seed', 'N/A')}.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()