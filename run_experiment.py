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
    world = GridWorld(seed=seed, grid_size=8)

    #number of helper agents to compare for each experiment
    num_helper_agents_feasible=6
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