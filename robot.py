import re
import ast
import gurobipy as gp
from gurobipy import GRB
from GridWorld import GridWorld
from Message import Message



class GridWorldAgent:
    def __init__(self, capabilities, initial_pos,
                 gridworld: GridWorld, T, agent_type,actuation=1,
                 needs_help=False, visit_first=False,ltl_locs=[],has_pallet=0):
        """
        A comprehensive GridWorldAgent that handles:
         - Movement constraints in a Gurobi model
         - Communication with LLM or other robots
         - Possibly multiple ways to encode/solve STL or 'must-visit' constraints
        """
        self.ltl_locs      = ltl_locs
        self.capabilities  = capabilities
        self.initial_pos   = initial_pos
        self.position      = initial_pos
        self.gridworld     = gridworld
        self.id            = self.gridworld.register_robot(self)
        self.T             = T
        self.actuation     = actuation
        self.agent_type = agent_type
        self.needs_help    = needs_help
        self.visit_first   = visit_first
        self.path = []
        self.T_all = None
        self.all_visited = {}
        self.visited = set()
        self.visited_locations = []
        self.has_pallet = has_pallet
        # Create the Gurobi model
        self.model = gp.Model(f"STL_gridworld_{self.id}")
        self.pickups = []
        self.dropoffs = []
        self.pickup_dropoff_pairs = []
        if self.has_pallet ==1:
            self.ltl_locs[f'loc_{self.initial_pos[0]}_{self.initial_pos[1]}'] = self.initial_pos

        # Movement variables
        self.b  = {}
        self.x  = {}
        self.y  = {}
        self.dx = {}
        self.dy = {}

        # For proposition dictionaries: "p1" => { t: var }, etc.
        self.propositions = {}

        # Build the base constraints (single-occupancy, movement, etc.)
        self.status = None
        self._build_model()

        #build propositions
        self.init_default_ltl_locations(self.ltl_locs)

    #--------------------------------------------------------------------------
    # Initialization of LTL locations and propositions
    #--------------------------------------------------------------------------
    def init_default_ltl_locations(self, points):
    #Initialize default LTL locations and propositions.Populates self.propositions.
        for name, coords in points.items():
            self.propositions[name] = self.define_position_propositions(name, coords)

    # --------------------------------------------------------------------------
    # Basic Communication / LLM Methods
    # --------------------------------------------------------------------------
    def generate_help_request(self, scene_description):
        """
        Query the LLM to generate a help request based on the current scene and self.capabilities.
        """
        prompt = f"""
        You are a helpful assistant, onboard a warehouse robot with capabilities listed in the Inferred capabilities.
        "Scene description" explains potential issues your robot might face.
        Generate a clear help request to other robots if you can not resolve this issue on your own.
        Otherwise, "no_help_needed=True".
        Inferred capabilities: {self.capabilities}
        Scene description: {scene_description}
        Current location: {self.position}
        """

        response = self.gridworld.llm_query(prompt)
        help_needed = response.split("no_help_needed=")[1].split()[0] == 'False'
        if help_needed:
            help_request = response.split("Help Request")[1]
            self.gridworld.broadcast_message(self.id, Message("help_request", help_request))
            self.needs_help = True
            self.help_request_content = help_request

    def propose_help(self, help_request, sender_id):
        """
        Query the LLM to propose help based on help request and self.capabilities
        """
        prompt = f"""
        You are a helpful assistant, onboard a warehouse robot with capabilities in the Inferred capabilities list.
        "Help request" describes the issue and the suggested capabilities that can help.
        Evaluate if you can help, and either:
          - Provide a 'help_proposal' with your relevant capabilities and "Location: (x,y)"
          - or "can't_help_you:True" if you cannot help.
        Inferred capabilities: {self.capabilities}
        Help request: {help_request}
        """
        response = self.gridworld.llm_query(prompt)
        self.help_proposal = response
        return Message("help_proposal", response)

    def receive_message(self, sender_id, message):
        """
        Handle inbound messages from other agents.
        """
        print(f"Robot {self.id} received from {sender_id}: {message}")

        if message.type == "help_request":
            # Possibly propose help
            help_proposal = self.propose_help(message.content, sender_id)
            self.gridworld.send_message(self.id, sender_id, help_proposal)

        elif message.type == "help_proposal":
            # For now, auto-confirm
            help_confirm = Message(type='help_confirmation', content=message.content)
            self.gridworld.send_message(self.id, sender_id, help_confirm)

        elif message.type == "help_confirmation":
            # Extract location, append to spec, rebuild
            pattern = r"Location\*{0,2}:\*{0,2}\s*(\(\s*\d+\s*,\s*\d+\s*\))"
            location_match = re.search(pattern, message.content)
            if location_match:
                goto_location = ast.literal_eval(location_match.group(1))
                self.ltl_spec.append(goto_location)
                self._build_model()

    # --------------------------------------------------------------------------
    # Basic Movement Constraints in Gurobi
    # --------------------------------------------------------------------------
    def _build_model(self):
        """
        Set up standard movement constraints (occupancy, no diagonal if actuation=1, etc.)
        """
        # Faster solving to prioiritize feasbility
        T = self.T
        free_cells = self.gridworld.free_cells
        grid_size = self.gridworld.grid_size
        obstacle_cells = self.gridworld.obstacle_cells

        # b[t,i,j], x[t], y[t], dx[t], dy[t]
        for t in range(T+1):
            # if it is not a drone, it can only be in free_cells
            if self.agent_type!='drone':
                for (i,j) in free_cells:
                    self.b[t, i, j] = self.model.addVar(vtype=GRB.BINARY,
                                                        name=f"b_{self.id}_{t}_{i}_{j}")
            else:
                # if it is a drone, gridworld is your oyster! 
                for (i,j) in set(free_cells) | set(obstacle_cells):
                    self.b[t, i, j] = self.model.addVar(vtype=GRB.BINARY,
                                                        name=f"b_{self.id}_{t}_{i}_{j}")

            self.x[t] = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=grid_size-1,
                                          name=f"x_{self.id}_{t}")
            self.y[t] = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=grid_size-1,
                                          name=f"y_{self.id}_{t}")

        for t in range(T):
            self.dx[t] = self.model.addVar(vtype=GRB.INTEGER, lb=-grid_size, ub=grid_size,
                                           name=f"dx_{self.id}_{t}")
            self.dy[t] = self.model.addVar(vtype=GRB.INTEGER, lb=-grid_size, ub=grid_size,
                                           name=f"dy_{self.id}_{t}")

        self.model.update()

        # Initial position
        (start_i, start_j) = self.initial_pos
        self.model.addConstr(self.b[0, start_i, start_j] == 1)
        self.model.addConstr(self.x[0] == start_i)
        self.model.addConstr(self.y[0] == start_j)

        # Exactly one cell at each time, link x[t], y[t]
        for t in range(T+1):
            self.model.addConstr(
                gp.quicksum(self.b[t,i,j] for (i,j) in free_cells) == 1,
                name=f"one_cell_{t}"
            )
            self.model.addConstr(
                self.x[t] == gp.quicksum(i * self.b[t,i,j] for (i,j) in free_cells)
            )
            self.model.addConstr(
                self.y[t] == gp.quicksum(j * self.b[t,i,j] for (i,j) in free_cells)
            )

        # Movement constraints
        for t in range(T):
            self.model.addConstr(self.dx[t] == self.x[t+1] - self.x[t])
            self.model.addConstr(self.dy[t] == self.y[t+1] - self.y[t])
            self.model.addConstr(self.dx[t] <= self.actuation)
            self.model.addConstr(self.dx[t] >= -self.actuation)
            self.model.addConstr(self.dy[t] <= self.actuation)
            self.model.addConstr(self.dy[t] >= -self.actuation)
            if self.agent_type != "drone":
            # Cardinal direction constraint (no diagonal)
                is_vertical_move = self.model.addVar(vtype=GRB.BINARY, name=f"is_vertical_{self.id}_t{t}")
                M = self.gridworld.grid_size

                self.model.addConstr(self.dx[t] <= (1 - is_vertical_move) * M, name=f"dx_zero_if_vert_pos_{self.id}_t{t}")
                self.model.addConstr(self.dx[t] >= -(1 - is_vertical_move) * M, name=f"dx_zero_if_vert_neg_{self.id}_t{t}")
                self.model.addConstr(self.dy[t] <= is_vertical_move * M, name=f"dy_zero_if_hor_pos_{self.id}_t{t}")
                self.model.addConstr(self.dy[t] >= -is_vertical_move * M, name=f"dy_zero_if_hor_neg_{self.id}_t{t}")

        # (Optional) no objective here by default. We'll define it in some solve methods.

    # --------------------------------------------------------------------------
    # Proposition Definitions
    # --------------------------------------------------------------------------
    def define_position_propositions(self, p_name, cell):
        """
        p_vars[t] = 1 iff robot is physically in 'cell' at time t
        """
        i, j = cell
        p_vars = {}
        for t in range(self.T+1):
            p_vars[t] = self.model.addVar(vtype=GRB.BINARY,
                                          name=f"{p_name}_{i}_{j}_t{t}")
        self.model.update()

        # Link p_vars[t] == b[t,i,j]
        for t in range(self.T+1):
            self.model.addConstr(
                p_vars[t] == self.b[t, i, j],
                name=f"link_{p_name}_{i}_{j}_{t}"
            )
        return p_vars



    def solve(self):
        """
        Solve the agent's model with the current objective/constraints
        """
        self.model.Params.OutputFlag = 1  # show Gurobi logs
        self.model.optimize()

    def get_path(self):
        """
        Return the path [ (i,j), ... ] for t=0..T if solution is optimal
        """
        if self.model.Status == GRB.OPTIMAL:
            path = []
            free_cells = self.gridworld.free_cells
            for t in range(self.T+1):
                for (i,j) in free_cells:
                    if self.b[t, i, j].X > 0.5:
                        path.append((i,j))
                        break
            return path
        else:
            return []

    # --------------------------------------------------------------------------
    # Basic STL Operators (optional)
    # --------------------------------------------------------------------------
    def stl_and(self, p_vars1, p_vars2, name="AND"):
        T = self.T
        p_and = {}
        for t in range(T+1):
            p_and[t] = self.model.addVar(vtype=GRB.BINARY,
                                         name=f"{name}_and_{t}")
        self.model.update()
        for t in range(T+1):
            self.model.addConstr(p_and[t] <= p_vars1[t])
            self.model.addConstr(p_and[t] <= p_vars2[t])
            self.model.addConstr(p_and[t] >= p_vars1[t] + p_vars2[t] - 1)
        return p_and

    def stl_or(self, p_vars1, p_vars2, name="OR"):
        T = self.T
        p_or = {}
        for t in range(T+1):
            p_or[t] = self.model.addVar(vtype=GRB.BINARY, name=f"{name}_or_{t}")
        self.model.update()
        for t in range(T+1):
            self.model.addConstr(p_or[t] >= p_vars1[t])
            self.model.addConstr(p_or[t] >= p_vars2[t])
            self.model.addConstr(p_or[t] <= p_vars1[t] + p_vars2[t])
        return p_or

    def stl_not(self, p_vars, name="NOT"):
        T = self.T
        p_not = {}
        for t in range(T+1):
            p_not[t] = self.model.addVar(vtype=GRB.BINARY,
                                         name=f"{name}_not_{t}")
        self.model.update()
        for t in range(T+1):
            self.model.addConstr(p_not[t] + p_vars[t] == 1)
        return p_not

    def stl_globally(self, p_vars, name="G"):
        """
        gl[t] = AND_{tau >= t} p_vars[tau].
        """
        T = self.T
        gl = {}
        for t in range(T+1):
            gl[t] = self.model.addVar(vtype=GRB.BINARY, name=f"{name}_gl_{t}")
        self.model.update()

        self.model.addConstr(gl[T] == p_vars[T])
        for t in reversed(range(T)):
            self.model.addConstr(gl[t] <= p_vars[t])
            self.model.addConstr(gl[t] <= gl[t+1])
            self.model.addConstr(gl[t] >= p_vars[t] + gl[t+1] - 1)
        return gl

    def stl_eventually(self, p_vars, name="F"):
        """
        ev[t] = OR_{tau >= t} p_vars[tau].
        """
        T = self.T
        ev = {}
        for t in range(T+1):
            ev[t] = self.model.addVar(vtype=GRB.BINARY, name=f"{name}_ev_{t}")
        self.model.update()

        self.model.addConstr(ev[T] == p_vars[T])
        for t in reversed(range(T)):
            self.model.addConstr(ev[t] >= p_vars[t])
            self.model.addConstr(ev[t] >= ev[t+1])
            self.model.addConstr(ev[t] <= p_vars[t] + ev[t+1])
        return ev
    def stl_until(self, p_vars, q_vars, name="UNTIL"):
        """
        Build the 'Until' operator for finite-horizon STL.
        until[t] = 1 if there is some tp >= t with q_vars[tp] = 1
                and for all tau in [t..tp-1], p_vars[tau] = 1.
        """
        model = self.model
        T = self.T

        # until[t] = the main variable for (p U q)[t].
        until = {}
        for t in range(T+1):
            until[t] = model.addVar(vtype=GRB.BINARY, name=f"{name}_{t}")
        model.update()

        # next_q[t, tp] = 1 means we "choose" time tp as the witness that q
        # holds at tp, and p holds from t..(tp-1).
        next_q = {}
        for t in range(T+1):
            for tp in range(t, T+1):
                next_q[t, tp] = model.addVar(vtype=GRB.BINARY,
                                            name=f"{name}_nextq_{t}_{tp}")
        model.update()

        # Enforce the "strong" part: if until[t] = 1, we MUST pick some tp >= t
        # with q[tp] = 1.  In other words, sum_{tp >= t} next_q[t,tp] >= until[t].
        # And we also limit sum_{tp>=t} next_q[t,tp] <= until[t] * bigM, so if
        # until[t]=0, we pick no tp. 
        M = T + 1
        for t in range(T+1):
            model.addConstr(
                gp.quicksum(next_q[t, tp] for tp in range(t, T+1)) >= until[t],
                name=f"{name}_min_{t}"
            )
            model.addConstr(
                gp.quicksum(next_q[t, tp] for tp in range(t, T+1)) <= M * until[t],
                name=f"{name}_max_{t}"
            )

        # If next_q[t,tp] = 1 => q_vars[tp] must be 1
        # Also, p_vars[k] must be 1 for all k in [t..tp-1].
        for t in range(T+1):
            for tp in range(t, T+1):
                # Force q[tp] to be 1 if chosen
                model.addConstr(next_q[t,tp] <= q_vars[tp],
                                name=f"{name}_q_req_{t}_{tp}")
                # Force p[k] = 1 for k in [t..tp-1]
                for k in range(t, tp):
                    model.addConstr(next_q[t,tp] <= p_vars[k],
                                    name=f"{name}_p_req_{t}_{tp}_{k}")

        return until



    def stl_first(self, main_vars, name="FIRST"):
        """
        Enforce that 'main_vars' (which is a dict phi[t] for t=0..T) is satisfied 
        strictly earlier (in the sense of earliest time of satisfaction) 
        than *any* other proposition/formula in self.propositions.

        'main_vars' might be atomic (like self.propositions["pA"]) 
        or a compound formula (like stl_implies_next(...)).

        We'll:
        1) compute T_main = earliest time main_vars[t] can be 1
        2) for each other subformula p_dict in self.propositions,
            compute T_p = earliest time p_dict[t] can be 1
        3) require T_main + 1 <= T_p whenever p_dict might actually be true at some time
        4) return a dict phi[t], forced to 1 for all t, 
            so that stl_first() can be used at the top level 
            or composed further if needed.
        """
        model = self.model
        T = self.T

        # 1) earliest time that main_vars[t] = 1
        T_main, sum_main_chosen = self._define_earliest_time_of_formula(main_vars, f"{name}_main")

        # 2) For each other subformula/proposition in self.propositions, do the same
        #    then enforce T_main < T_other (strictly) if the other is actually satisfied.
        for p_name, p_dict in self.propositions.items():
            if p_dict is main_vars:
                continue  # skip itself

            T_other, sum_other_chosen = self._define_earliest_time_of_formula(p_dict, f"{name}_{p_name}")

            # If the other formula is actually satisfied (sum_other_chosen=1),
            # then we want T_main < T_other. We'll do T_main + 1 <= T_other.
            # We'll add a big-M so that if sum_other_chosen=0 (never satisfied),
            # the constraint is vacuously skipped.
            bigM = T+1
            model.addConstr(
                T_main + 1 <= T_other + bigM*(1 - sum_other_chosen),
                name=f"{name}_strict_{p_name}"
            )

        # 3) Return a dict of phi[t], forced to 1. If constraints can't be satisfied,
        #    the solver goes infeasible. If they can be, phi[t]=1 means "FIRST(...) is true."
        phi = {}
        for t in range(T+1):
            phi[t] = model.addVar(vtype=GRB.BINARY, name=f"{name}_phi_{t}")
            model.addConstr(phi[t] == 1, name=f"{name}_all_true_{t}")
        model.update()

        return phi


    def _define_earliest_time_of_formula(self, formula_vars, label):
        """
        Helper: define T_phi = the earliest time formula_vars[t] can be 1.
        Also define a binary sum_of_chosen that indicates whether formula
        is *ever* satisfied. If sum_of_chosen=0, the formula is never satisfied.
        If sum_of_chosen=1, T_phi tracks the earliest time it was satisfied.

        Returns (T_phi, sum_of_chosen).
        """
        model = self.model
        T = self.T

        # For each t, define a binary var pick_earliest[t].
        pick_earliest = {}
        for t in range(T+1):
            pick_earliest[t] = model.addVar(vtype=GRB.BINARY,
                                            name=f"{label}_pick_{t}")
        model.update()

        # sum_of_chosen = 0 or 1
        sum_of_chosen = model.addVar(vtype=GRB.BINARY, name=f"{label}_any_time")

        # sum_{t} pick_earliest[t] <= sum_of_chosen
        # sum_{t} pick_earliest[t] >= sum_of_chosen
        model.addConstr(
            gp.quicksum(pick_earliest[t] for t in range(T+1)) <= sum_of_chosen
        )
        model.addConstr(
            gp.quicksum(pick_earliest[t] for t in range(T+1)) >= sum_of_chosen
        )

        # If pick_earliest[t] = 1 => formula_vars[t] = 1
        # and formula_vars[tau] = 0 for tau < t
        for t in range(T+1):
            model.addConstr(pick_earliest[t] <= formula_vars[t],
                            name=f"{label}_earliest_implies_form_{t}")
            for tau in range(t):
                # Can't have formula_vars[tau] = 1 if we say earliest is t
                model.addConstr(pick_earliest[t] + formula_vars[tau] <= 1,
                                name=f"{label}_no_earlier_{t}_{tau}")

        # T_phi is an integer var
        T_phi = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f"{label}_T")
        model.update()

        # big-M approach so T_phi = sum_{t} t * pick_earliest[t] IF sum_of_chosen=1
        M = T + 2
        chosen_expr = gp.quicksum(t * pick_earliest[t] for t in range(T+1))

        model.addConstr(T_phi >= chosen_expr, name=f"{label}_T_lb")
        model.addConstr(T_phi <= chosen_expr + M*(1 - sum_of_chosen),
                        name=f"{label}_T_ub")

        return T_phi, sum_of_chosen



    def stl_implies_next(self, left_vars, right_vars, name="IMPLIES_NEXT"):
        """
        For each time t:
        phi[t] = 1 iff (left[t] = 0) OR
                    (there is exactly one future tp>t with right[tp]=1 and
                    no left/right/others in t+1..tp-1)
        ...
        """
        model = self.model
        T = self.T

        other_vars_list = []
        for p_name, p_dict in self.propositions.items():
            # if p_dict is not left_vars and p_dict is not right_vars
            # and you want them blocked in the gap, add to other_vars_list
            if p_dict is not left_vars and p_dict is not right_vars:
                other_vars_list.append(p_dict)
 
        # Create phi[t], a binary var that might be 0 if the property fails at time t
        phi = {}
        for t in range(T+1):
            phi[t] = model.addVar(vtype=GRB.BINARY, name=f"{name}_phi_{t}")
        model.update()

        # next_right[t,tp]
        next_right = {}
        for t in range(T+1):
            for tp in range(t+1, T+1):
                next_right[t,tp] = model.addVar(vtype=GRB.BINARY,
                                                name=f"{name}_nr_{t}_{tp}")
        model.update()

        # Must visit left at least once, right at least once
        model.addConstr(gp.quicksum(left_vars[t] for t in range(T+1)) >= 1)
        model.addConstr(gp.quicksum(right_vars[t] for t in range(T+1)) >= 1)

        # No right before left
        for t in range(T+1):
            model.addConstr(
                right_vars[t] <= gp.quicksum(left_vars[tau] for tau in range(t)),
                name=f"{name}_no_right_before_left_{t}"
            )

        # If left[t]=1, at most 1 next_right[t,tp]
        for t in range(T+1):
            model.addConstr(
                gp.quicksum(next_right[t,tp] for tp in range(t+1, T+1))
                == left_vars[t],
                name=f"{name}_maxone_tp_{t}"
            )

        # If next_right[t,tp]=1 => right[tp]=1, no left/right/others in gap
        for t in range(T+1):
            for tp in range(t+1, T+1):
                model.addConstr(next_right[t,tp] <= right_vars[tp])
                for tau in range(t+1, tp):
                    model.addConstr(next_right[t,tp] + left_vars[tau] + right_vars[tau] <= 1)
                    for o_vars in other_vars_list:
                        model.addConstr(o_vars[tau] + next_right[t,tp] <= 1)

        # Now define phi[t]:
        # phi[t] = 1 if (left[t]=0) OR we can pick exactly one next_right[t,tp].
        # This is done with the constraints:
        for t in range(T+1):
            sum_nr_t = gp.quicksum(next_right[t,tp] for tp in range(t+1,T+1))
            # If left[t]=0 => phi[t]=1. So phi[t]>=1-left[t]
            model.addConstr(phi[t] >= 1 - left_vars[t])
            # If left[t]=1 => we need sum_nr_t=1 to get phi[t]=1
            model.addConstr(phi[t] <= 1 - left_vars[t] + sum_nr_t)
            model.addConstr(phi[t] >= sum_nr_t)
            # phi[t] <= 1 is implicit from binary vtype

        return phi


    def encode_stl(self, formula, top_level = True):
        """
        Recursively build constraints for an STL formula:
          - string => proposition dictionary self.propositions["pX"]
          - ("NOT", sub) => stl_not(...)
          - ("AND", l, r)
          - ("OR", l, r)
          - ("G", sub)
          - ("F", sub)
           ("UNTIL", p, q)
        Returns dictionary phi_vars[t].
        """
        if isinstance(formula, str):
            return self.propositions[formula]

        op = formula[0].upper()
        if op == "NOT":
            sub = self.encode_stl(formula[1])
            return self.stl_not(sub)
        elif op == "AND":
            left = self.encode_stl(formula[1])
            right = self.encode_stl(formula[2])
            return self.stl_and(left, right)
        elif op == "OR":
            left = self.encode_stl(formula[1])
            right = self.encode_stl(formula[2])
            return self.stl_or(left, right)
        if op == "G":
            sub = self.encode_stl(formula[1], top_level=False)
            gl = self.stl_globally(sub)
            if top_level:
                # Force it to hold from time 0
                self.model.addConstr(gl[0] == 1, name="global_at_0")
            return gl

        if op == "F":
            sub = self.encode_stl(formula[1], top_level=False)
            ev = self.stl_eventually(sub)
            if top_level:
                # Force eventually from time 0
                self.model.addConstr(ev[0] == 1, name="eventually_at_0")
            return ev
        elif op == "UNTIL":
            p_sub = self.encode_stl(formula[1], top_level=False)
            q_sub = self.encode_stl(formula[2], top_level=False)
            u = self.stl_until(p_sub, q_sub, name="UNTIL")
            self.model.addConstr(u[0] == 1, name="until_at_0")
            if top_level:
                # Typically, we say "the formula holds at time 0"
                self.model.addConstr(u[0] == 1, name="until_at_0")
            return u

        elif op =="IMPLIES_NEXT":
            left_sub = self.encode_stl(formula[1])
            right_sub = self.encode_stl(formula[2])
            return self.stl_implies_next(left_sub, right_sub)
        elif op == "FIRST":
    # formula is ("FIRST", left_formula, right_formula)
            left_sub  = self.encode_stl(formula[1])
            right_sub = self.encode_stl(formula[2])
            return self.stl_first(left_sub)   
        else:
            # unrecognized
            return {}



    # --------------------------------------------------------------------------
    # "Physical must-visit" aggregator approach
    # --------------------------------------------------------------------------
    def parse_must_visit_sets(self, formula):
        """
        A simplistic parser that returns a list of sets of proposition names
        that must be *eventually visited* physically.
        
        We skip 'NOT', 'G' because they represent 'avoid' or global constraints,
        not visits. We skip 'IMPLIES_NEXT' except we do interpret its subformulas
        as must-visit sets.

        So if we see ("IMPLIES_NEXT", left, right),
        we treat it similarly to "AND" (meaning we must visit both left and right).
        """
        if isinstance(formula, str):
            # single proposition => must eventually visit it
            return [ { formula } ]

        op = formula[0].upper()

        if op == "AND":
            left_sets = self.parse_must_visit_sets(formula[1])
            right_sets = self.parse_must_visit_sets(formula[2])
            return left_sets + right_sets

        elif op == "OR":
            # for a disjunction, we union everything into one set
            left_sub = self.parse_must_visit_sets(formula[1])
            right_sub = self.parse_must_visit_sets(formula[2])
            combined = set()
            for s in left_sub:
                combined |= s
            for s in right_sub:
                combined |= s
            return [ combined ]

        elif op == "F":
            # "Eventually subformula"
            return self.parse_must_visit_sets(formula[1])

        elif op in ("IMPLIES_NEXT","FIRST","UNTIL"):
            # interpret "IMPLIES_NEXT(left, right)" as meaning
            # must eventually do 'left' *and* must eventually do 'right'
            left_sets = self.parse_must_visit_sets(formula[1])
            right_sets = self.parse_must_visit_sets(formula[2])
            # effectively treat it like an AND for "must visit" logic
            return left_sets + right_sets
 

        elif op == "NOT":
            return []

        elif op == "G":
            return self.parse_must_visit_sets(formula[1])

        else:
            # unknown => skip
            return []

    def define_visited_by_single(self, prop_vars, name="visited_by_single"):
        """
        visited[t] = 1 if we have visited the cell of prop_vars at or before time t.
        We want:
        visited[0] = prop_vars[0]
        For t >= 0:
            visited[t+1] >= visited[t]
            visited[t+1] >= prop_vars[t+1]
            visited[t+1] <= visited[t] + prop_vars[t+1]
        """
        T = self.T
        visited = {}
        for t in range(T+1):
            visited[t] = self.model.addVar(vtype=GRB.BINARY,
                                        name=f"{name}_{t}")
        self.model.update()

        # Base case
        self.model.addConstr(visited[0] == prop_vars[0],
                            name=f"{name}_base")

        for t in range(T):
            # Monotonic: can't lose the "visited" status once gained
            self.model.addConstr(visited[t+1] >= visited[t],
                                name=f"{name}_mono_{t}")

            # If we physically stand in p_vars at t+1, we must be visited
            self.model.addConstr(visited[t+1] >= prop_vars[t+1],
                                name=f"{name}_cap_{t}")

            # *** CRITICAL: upper bound constraint ***
            # visited[t+1] <= visited[t] + prop_vars[t+1]
            # ensures it can't become 1 out of nowhere
            self.model.addConstr(visited[t+1] <= visited[t] + prop_vars[t+1],
                                name=f"{name}_ub_{t}")

        return visited

    def define_visited_by_disjunction(self, prop_list, name="visited_by_disj"):
        """
        aggregator[t] = 1 if we have visited ANY of prop_list by time t
        """
        T = self.T
        visited = {}

        # aggregator for each single proposition
        single_arrays = []
        for i, p_vars in enumerate(prop_list):
            single_arrays.append(self.define_visited_by_single(p_vars, name=f"{name}_single{i}"))

        for t in range(T+1):
            visited[t] = self.model.addVar(vtype=GRB.BINARY, name=f"{name}_{t}")
        self.model.update()

        # For t=0
        self.model.addConstr(visited[0] >= gp.quicksum(sv[0] for sv in single_arrays))
        self.model.addConstr(visited[0] <= gp.quicksum(sv[0] for sv in single_arrays))

        # for t>0
        for t in range(T):
            self.model.addConstr(visited[t+1] >= visited[t])
            lhs = visited[t]
            for sv in single_arrays:
                self.model.addConstr(visited[t+1] >= sv[t+1])
                lhs += sv[t+1]
            self.model.addConstr(visited[t+1] <= lhs)

        return visited

    def define_all_visited_aggregator(self, list_of_sets):
        """
        For e.g. [ {"p1"}, {"p2","p3"} ], define aggregator[t] = 1 if
        set1 was visited_by t, set2 was visited_by t, etc. => AND
        """
        T = self.T
        aggregator_list = []
        for i, prop_set in enumerate(list_of_sets):
            prop_list = [self.propositions[p] for p in prop_set]
            if len(prop_set) == 1:
                aggregator_list.append(
                    self.define_visited_by_single(prop_list[0], name=f"visit_single_{i}")
                )
            else:
                aggregator_list.append(
                    self.define_visited_by_disjunction(prop_list, name=f"visit_disj_{i}")
                )

        # now aggregator_list is e.g. [ agg1[t], agg2[t], ... ] => we want an AND
        all_visited = {}
        for t in range(T+1):
            all_visited[t] = self.model.addVar(vtype=GRB.BINARY, name=f"all_visited_{t}")
        self.model.update()

        for t in range(T+1):
            # all_visited[t] <= each aggregator_list[i][t]
            for agg in aggregator_list:
                self.model.addConstr(all_visited[t] <= agg[t])
            # all_visited[t] >= sum(aggregator_list[i][t]) - (N-1)
            self.model.addConstr(
                all_visited[t] >= gp.quicksum(agg[t] for agg in aggregator_list)
                - (len(aggregator_list)-1)
            )

        return all_visited

    def define_earliest_all_visited_time(self, all_visited, name="T_all"):
        """
        pick the earliest t => all_visited[t] = 1
        """
        T = self.T
        first_sat = {}
        for t in range(T+1):
            first_sat[t] = self.model.addVar(vtype=GRB.BINARY, name=f"{name}_first_{t}")

        T_all = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=name)
        self.model.update()

        self.model.addConstr(
            gp.quicksum(first_sat[t] for t in range(T+1)) == 1,
            name=f"{name}_one_t"
        )

        for t in range(T+1):
            self.model.addConstr(all_visited[t] >= first_sat[t])
            for tau in range(t):
                self.model.addConstr(all_visited[tau] <= 1 - first_sat[t])

        self.model.addConstr(
            T_all == gp.quicksum(t * first_sat[t] for t in range(T+1)),
            name=f"{name}_def"
        )
        return T_all



    def solve_physical_visits(self, formula, is_original=True,optimize_thelp=False,optimize_tstar=False,optimize_cost=False,use_dist=False,alpha=1000.0,help_site_name='help_site'):
        # 1) Possibly parse formula into a nested structure if it's a string
        #    or if you already have it as a structure, skip this step

        # 2) Build constraints for the entire STL formula, including IMPLIES_NEXT
        if help_site_name not in self.propositions:
            raise ValueError(f"help_site '{help_site_name}' not in self.propositions!")
        help_vars = self.propositions[help_site_name]
        T_help, sum_help_chosen = self._define_earliest_time_of_formula(help_vars, "help_site")

    # Force the model to actually visit help_site at least once
    # (sum_help_chosen >= 1). If we want to ensure it *must* be visited:
        if not is_original:
            self.model.addConstr(sum_help_chosen == 1, name="must_visit_help_site")
        phi_vars = self.encode_stl(formula)
        self.model.addConstr(phi_vars[0]==1,"froce global")
 
            
        list_of_sets = self.parse_must_visit_sets(formula)
        self.all_visited = self.define_all_visited_aggregator(list_of_sets)
        self.T_all = self.define_earliest_all_visited_time(self.all_visited)

        # 4) Define your objective
        dist_expr = gp.QuadExpr()
        for t in range(self.T):
            dist_expr += (self.dx[t]*self.dx[t] + self.dy[t]*self.dy[t])

        if  optimize_thelp:
            if use_dist:
                obj = alpha*T_help + self.T_all + dist_expr
            else:
                obj = alpha*T_help  + self.T_all
                # obj = alpha * T_help + self.T_all
        elif optimize_tstar:
            if use_dist:
                obj = alpha*self.T_all + dist_expr
            else:
                obj = alpha*self.T_all
        elif optimize_cost:
            if use_dist:
                obj = T_help + self.T_all + dist_expr
            else:
                obj = T_help + self.T_all
                # obj = alpha*self.T_all
        else:
            obj = self.T_all 
        self.model.setObjective(obj, GRB.MINIMIZE)
        # 5) Solve
        self.model.Params.OutputFlag = 0  # turn off solver logs

        self.model.optimize()

        self.status = self.model.Status
        if self.status == GRB.OPTIMAL:
            self.path = self.get_path()
            self.get_visited_proposition_locations()
            # print(f"Type T_all is {type(self.T_all.X)}")
            status, path, T_all, all_visited, solve_time = self.status, self.path, self.T_all.X, self.all_visited,self.model.Runtime

        else:
            status, path, T_all, all_visited, solve_time = self.status, [], None, None, None
        self.reset_and_rebuild_model()
        return status, path, T_all, all_visited, solve_time

    def get_visited_proposition_locations(self):
        """
        Returns a list of tuples indicating the order in which required proposition locations were visited.
        Each tuple contains (timestep, proposition_name, proposition_location).
        """
        if not self.path or not self.all_visited:
            raise ValueError("Solve the model and populate self.path first.")
 
        # Iterate through each timestep to check visited propositions
        for t, position in enumerate(self.path):
            for prop_name, prop_loc in self.ltl_locs.items():
                if position == prop_loc and prop_name not in self.visited:
                    self.visited.add(prop_name)
                    self.visited_locations.append((t, prop_name, prop_loc))





    def reset_and_rebuild_model(self):
        """
        Completely clear existing constraints, variables, and rebuild the Gurobi model.
        """
        self.model.reset()
        self.model.remove(self.model.getConstrs())
        self.model.remove(self.model.getVars())
        self.model.update()

        self.b.clear()
        self.x.clear()
        self.y.clear()
        self.dx.clear()
        self.dy.clear()
        self.propositions.clear()

        self._build_model()
        self.init_default_ltl_locations(self.ltl_locs)