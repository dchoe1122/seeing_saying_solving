import string
import re
import ast
from GridWorld import GridWorld
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import math
import csv
import os


def get_incremental_cost(original_completion_time,updated_completion_time,t_h,alpha=1,beta=1):
    """Computes
      (1) t_h (time it took for the requester to be helped), i.e time to help_site
       (2) t*, the additional time it cost for the helper
       (3) cost = alpha* t_h + betha * t* where default value is alpha =1 and beta =1 
       """
    #first compute t_original
    t_original = original_completion_time
    #Then compute t_updated
    t_updated = updated_completion_time
    t_star = t_updated-t_original
    t_h = t_h
    cost = alpha*t_star + beta*t_h
    return t_star, t_h, cost


def count_transitions_until_target(path, target_location):
    if len(path) < 2:
        return 0

    transitions = 0
    prev_point = path[0]
    #need to add the case where we start at helpsite
    if path[0] == target_location:
        return 0
        

    for point in path[1:]:
        if point != prev_point:
            transitions += 1
        if point == target_location:
            break
        prev_point = point

    return transitions


def start_pos_init(grid:GridWorld,needs_help=False):
    if needs_help:
        start_pos = grid.conflict_cell
    else:    
        free_except_conflict = [c for c in grid.free_cells if c != grid.conflict_cell]
        start_pos = random.choice(free_except_conflict)
    return start_pos


def get_adjacent_free_cell(grid, cell):
    """Returns a random free cell adjacent to the given cell."""
    i, j = cell
    adjacent_cells = [
        (i + 1, j),
        (i - 1, j),
        (i, j + 1),
        (i, j - 1)
    ]
    valid_adjacent = [c for c in adjacent_cells if c in grid.free_cells]

    if not valid_adjacent:
        raise ValueError("No adjacent free cells available.")

    return random.choice(valid_adjacent)
def compute_distance(point1, point2):
    """
    Compute the Euclidean distance between two points represented by tuples.
    
    Parameters:
        point1 (tuple): First point (x1, y1)
        point2 (tuple): Second point (x2, y2)
        
    Returns:
        float: Euclidean distance
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_ltl_spec(grid: GridWorld, num_existing_locs=2, needs_help=False, is_drone=False):
    """
    Returns a dictionary where key is 'ltl_i_j' and value is (i,j) coordinates
    for randomly selected interest points.
    """
    ltl_spec = {}

    if not is_drone:
        all_free = list(grid.free_cells)
        random.shuffle(all_free)

        selected = all_free[:num_existing_locs]

        if not needs_help:
            ltl_spec["help_site"] = grid.conflict_cell
            ltl_spec["help_site_drop"] = get_adjacent_free_cell(grid, grid.conflict_cell)

        for (i, j) in selected:
            key = f"loc_{i}_{j}"
            ltl_spec[key] = (i, j)

    else:
        ltl_spec["help_site"] = grid.conflict_cell
        four_corners_and_charge = [
            (0, 0),
            (0, grid.grid_size-1),
            (grid.grid_size-1, 0),
            (grid.grid_size-1, grid.grid_size-1),
            (round((grid.grid_size-1)/2), round((grid.grid_size-1)/2))
        ]

        for (i, j) in four_corners_and_charge:
            key = f"loc_{i}_{j}"
            ltl_spec[key] = (i, j)

    return ltl_spec



def find_ltl_spec(llm_output:string):
    match = re.search(r'new_ltl_spec:\s*(\[[^\]]+\])', llm_output)
    if match:
        new_ltl_spec = ast.literal_eval(match.group(1))
        print(new_ltl_spec)
    else:
        print("new_ltl_spec not found.")
    return new_ltl_spec

def generate_dummy_llm_stl_expression(ltl_locs):
    """
    Randomly picks 5 locations from ltl_locs and inserts them into the STL template.
    Returns a formula string like:
    (F(ltl_1_2)&F(ltl_3_4)) | (ltl_5_6) & IMPLIES_NEXT(ltl_7_8,ltl_9_0)
    """
    if len(ltl_locs) < 5:
        raise ValueError("Need at least 5 locations in ltl_locs to generate dummy expression.")

    loc_keys = random.sample(list(ltl_locs.keys()), 5)
    a, b, c, d, e = loc_keys

    llm_stl_expression = f"(F({a}) & F({b})) | ({c}) & IMPLIES_NEXT({d}, {e})"
    return llm_stl_expression

def generate_dummy_llm_requester_stl_expression(ltl_locs):
    """
    Randomly picks 3 locations from self.ltl_locs and inserts them into the STL template.
    Returns a formula string like:
    (task1 & task2 & task3)
    """
    if len(ltl_locs) < 3:
        raise ValueError("Need at least 5 locations in ltl_locs to generate dummy expression.")

    loc_keys = random.sample(list(ltl_locs.keys()), 3)
    a, b, c = loc_keys

    llm_stl_expression = f"(({a}) & {b}) & ({c}))"
    return llm_stl_expression


def plot_just_gridworld(grid: GridWorld):
    grid_plot = [[0 for _ in range(grid.grid_size)] for _ in range(grid.grid_size)]

    # Mark obstacles
    for (i, j) in grid.obstacle_cells:
        grid_plot[i][j] = 1

    # Mark conflict cell (if applicable)

    print(grid_plot)
    cmap = colors.ListedColormap(['white', 'gray'])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_plot, cmap=cmap, origin='lower')

    # Grid settings
    ax.set_xticks(range(grid.grid_size))
    ax.set_yticks(range(grid.grid_size))
    ax.set_xticklabels(range(grid.grid_size))
    ax.set_yticklabels(range(grid.grid_size))
    ax.grid(True)

    plt.title("Simple Gridworld Layout")
    plt.show()

# Usage


def parse_formula_string(s):
    """
    Parse a string like:
        "F(p1) & F(p2) & (p3 | p4)"
    or
        "IMPLIES_NEXT(pickup, dropoff) & F(p3)"
    into a nested tuple structure.

    Allowed forms (for demonstration):
      - "p1", "p2" for atomic propositions
      - "F(...)" => ("F", subformula)
      - "G(...)" => ("G", subformula)
      - "IMPLIES_NEXT(x, y)" => ("IMPLIES_NEXT", x, y) [special 2-arg operator]
      - "p1 & p2" => ("AND", p1, p2)
      - "p1 | p2" => ("OR", p1, p2)
      - parentheses grouping "( ... )"
      - "~" for NOT => "~p1" => ("NOT", "p1")

    We'll do a naive approach with recursion or splitting on & and |.
    """
    tokens = tokenize(s)
    # parse_expr returns (node, next_pos)
    return parse_expr(tokens, 0)[0]

def tokenize(s):
    """
    Convert, e.g., "F(p1) & F(p2) & (p3 | p4)"
    to tokens like ["F","(","p1",")","&","F","(","p2",")","&","(","p3","|","p4",")"]
    
    Now also we look for IMPLIES_NEXT as a single token if typed in uppercase or so.
    """
    # We can add 'IMPLIES_NEXT' to the pattern or rely on letters+underscores:
    # a simple approach with regex capturing words/parentheses/operators:
    pattern = r'[A-Za-z0-9_]+|\(|\)|&|\||~|,'
    return re.findall(pattern, s)

def parse_expr(tokens, i):
    """
    We'll parse left to right, building an 'AND/OR chain' with
    parentheses, F(), G(), IMPLIES_NEXT(...).
    returns (node, next_pos).
    """
    (node, i) = parse_factor(tokens, i)

    while i < len(tokens):
        if tokens[i] in ['&', '|']:
            op = tokens[i]
            i += 1
            (right_node, i) = parse_factor(tokens, i)
            if op == '&':
                node = ("AND", node, right_node)
            else:  # '|'
                node = ("OR", node, right_node)
        else:
            break

    return (node, i)

def parse_factor(tokens, i):
    """
    factor can be:
      - "F(...)" or "G(...)" or "IMPLIES_NEXT(...)" or "FIRST(...)" or "~"
      - "(" expr ")"
      - a plain proposition like "p1"
    returns (node, next_pos)
    """
    if i >= len(tokens):
        return (None, i)

    t = tokens[i]

    if t == '(':
        # parse subexpression
        (node, i2) = parse_expr(tokens, i+1)
        if i2 < len(tokens) and tokens[i2] == ')':
            return (node, i2+1)
        else:
            raise ValueError("Missing closing parenthesis")

    elif t in ('F','G','IMPLIES_NEXT','FIRST','UNTIL'):
        op = t.upper()
        # must see '(' after
        if i+1 < len(tokens) and tokens[i+1] == '(':
            if op in ('F', 'G'):
                # parse single sub-expression
                (subnode, i2) = parse_expr(tokens, i+2)  # skip '('
                # expect ')'
                if i2 < len(tokens) and tokens[i2] == ')':
                    return ((op, subnode), i2+1)
                else:
                    raise ValueError(f"Missing closing parenthesis after {op}(")

            elif op in ('IMPLIES_NEXT', 'FIRST', 'UNTIL'):
                # special 2-argument form: IMPLIES_NEXT(expr, expr) or FIRST(expr, expr)
                i2 = i+2  # skip '('
                # parse first sub-expression
                (left_node, i2) = parse_expr(tokens, i2)
                # expect ','
                if i2 < len(tokens) and tokens[i2] == ',':
                    i2 += 1  # skip comma
                else:
                    raise ValueError(f"Expected comma in {op}(...)")

                # parse second sub-expression
                (right_node, i2) = parse_expr(tokens, i2)
                # expect ')'
                if i2 < len(tokens) and tokens[i2] == ')':
                    return ((op, left_node, right_node), i2+1)
                else:
                    raise ValueError(f"Missing closing parenthesis in {op}(...)")
        else:
            raise ValueError(f"Expected '(' after {t}")

    else:
        # maybe "~" or a proposition
        if t == '~':
            # parse factor after ~
            (subnode, i2) = parse_factor(tokens, i+1)
            return (("NOT", subnode), i2)
        else:
            # assume t is an atomic proposition
            return (t, i+1)


def parse_pickup_dropoff_with_coords(input_string, ltl_locs):
    """
    Parses the input string and returns lists of pickup and dropoff coordinates.

    Parameters:
        input_string (str): Logical string containing IMPLIES_NEXT statements.
        ltl_locs (dict): Dictionary mapping location identifiers to coordinates.

    Returns:
        pickups (list): List of pickup coordinate tuples.
        dropoffs (list): List of dropoff coordinate tuples.
    """
    pattern = r'IMPLIES_NEXT\s*\(\s*(loc_\d+_\d+)\s*,\s*(loc_\d+_\d+)\s*\)'
    matches = re.findall(pattern, input_string)

    pickups = []
    dropoffs = []

    for pickup_key, dropoff_key in matches:
        if pickup_key in ltl_locs and dropoff_key in ltl_locs:
            pickups.append(ltl_locs[pickup_key])
            dropoffs.append(ltl_locs[dropoff_key])

    return pickups, dropoffs



def mobile_robot_stl(ltl_locs):
    loc_keys_filtered = [key for key in ltl_locs if key not in ("help_site", "help_site_drop")]
    loc_keys = random.sample(loc_keys_filtered, 3)
    a, b, c = loc_keys
    return f"F({a})&F({b})&F({c})"

def forklift_stl_generation(ltl_locs):
    loc_keys_filtered = [key for key in ltl_locs if key not in ("help_site", "help_site_drop")]

    loc_keys = random.sample(loc_keys_filtered, 4)
    a, b, c, d = loc_keys
    return f"IMPLIES_NEXT({a}, {b}) & (IMPLIES_NEXT({c},{d}))"


def save_results_to_csv(grid, requester_stl, agent_results_list,
                        filename_prefix='Test_Results', save_dir='Scenario 1 tests_April_4th'):

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Unpack requester results
    requester_status, requester_path, requester_t_earliest, _,solve_time = requester_stl

    # Prepare data for CSV
    data = [
        ['World Seed', grid.seed],
        ['Requester STL', requester_status, requester_t_earliest, '', ''],
        ['Agent ID', 'Method', 'Feasibility Status', 'Completion Time', 't_h', 't_star','total_cost','Gurobi Solve Time','World Seed','Distance from help site',\
         'Additional Solve Time Required','Had Pallets on Forks?']
    ]

    # Iterate over all agents' results
    for agent_dict in agent_results_list:
        agent = agent_dict['agent']
        original_results = agent_dict['original_results_help']
        cost_optimal_results = agent_dict['updated_results_help']
        distance_from_help_site = agent_dict['distance_from_help_site']

        original_status, original_path, original_t_earliest, _,original_solve_time = original_results
        
        cost_optimal_status, cost_optimal_path,cost_optimal_t_earliest,_,cost_optimal_solve_time = cost_optimal_results

        # Calculate t_h and t_star

        t_h_cost_optimal = count_transitions_until_target(cost_optimal_path, grid.conflict_cell) if cost_optimal_status == 2 else None
        t_star_cost_optimal = (cost_optimal_t_earliest - original_t_earliest) if cost_optimal_status == 2 else None


        
        if t_h_cost_optimal is not None and t_star_cost_optimal is not None:
            cost_cost_optimal = t_h_cost_optimal+t_star_cost_optimal
        else:
            cost_tstar_optimal= ''

        additional_solve_time = cost_optimal_solve_time-original_solve_time
        data.extend([
            [agent.id, 'Original', original_status, original_t_earliest, '', '','',original_solve_time,grid.seed,distance_from_help_site,0,agent.has_pallet],
             [agent.id, 'cost_optimal', cost_optimal_status,cost_optimal_t_earliest,t_h_cost_optimal,t_star_cost_optimal,\
              cost_cost_optimal,cost_optimal_solve_time,grid.seed,distance_from_help_site,additional_solve_time, agent.has_pallet]])

    # Write data to CSV
    csv_filename = os.path.join(save_dir, f"{filename_prefix}_world_{grid.seed}_with_also_cost_optimum.csv")

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"Results saved to {csv_filename}")
