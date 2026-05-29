import mesa

# Agent state constants
#
# Five-state machine representing each agent's position in the
# mobilization-repression-reconstitution cycle:
#
#   dormant     — not yet mobilized; initial state (unused after __init__)
#   mobilized   — actively participating in collective action at or near L
#   dispersing  — fleeing after repression; moving outward from L
#   sheltered   — hiding in a refuge cell; evaluating whether to remobilize
#   demobilized — gave up; exited collective action permanently
#
# Terminal states: mobilized (reconstituted), demobilized (terminated).
# Transitional states: dispersing, sheltered.
# The model stops when no transitional agents remain.

DORMANT = "dormant"
MOBILIZED = "mobilized"
DISPERSING = "dispersing"
SHELTERED = "sheltered"
DEMOBILIZED = "demobilized"


class ResidentAgent(mesa.Agent):
    """
    Agent = one resident who may participate in collective action after repression.

    Decision rules implement a spatially explicit threshold model (Granovetter 1978):
    agents remobilize when the local mobilization rate exceeds their personal threshold. 
    Thresholds are heterogeneous (drawn from N(mean, std) in the model). 
    Cascade dynamics emerge from the interaction of individual thresholds with the spatial distribution of peers during the post-dispersal reconstitution phase.

    Attributes:
        threshold (float [0,1]):proportion of local peers who must be mobilized before this agent remobilizes. 
                                Lower threshold = easier to activate. 

        state (str):             current position in the five-state machine.
        steps_dispersing (int):  counter for how many steps this agent has been in the DISPERSING state. Triggers shelter attempt when it reaches dispersal_duration.
    """
    def __init__(self, model, threshold):
        super().__init__(model)
        self.threshold = threshold
        self.state = DORMANT
        self.steps_dispersing = 0

    def step(self):
        """
        Dispatch to the behavior appropriate to this agent's current state.

        MOBILIZED agents do nothing (they are the reference point for others' threshold calculations). 
        DORMANT and DEMOBILIZED agents are also inactive.
        Only DISPERSING and SHELTERED agents take actions in each step.
        """
        if self.state == MOBILIZED:
            return
        elif self.state == DISPERSING:
            self._disperse_step()
        elif self.state == SHELTERED:
            self._check_threshold()

        # DORMANT and DEMOBILIZED: no action

    def _disperse_step(self):
        """
        Move one step away from the repression site L, then attempt shelter after dispersal_duration steps have elapsed.

        Movement logic (three components):
          1. Outward bias: prefer cells farther from L (repressed protest site).
             If no outward cells exist (e.g., at grid boundary), use all neighbors.
          2. Social attraction: weight candidate cells by the number of other active agents already there. 
             Agents cluster with co-dispersers, producing the local density needed for cascade ignition.
          3. Permeability gate: movement to the chosen cell succeeds only with probability p (from the cell's p_grid value). 
             Failed movement means the agent stays in place but still increments steps_dispersing.

        After dispersal_duration steps, _attempt_shelter() is called.
        Dispersal_duration operationalizes repression intensity: longer duration means agents are pushed farther from L before sheltering, 
        reducing post-dispersal density.
        """
        L = self.model.repression_location
        current_dist = self._chebyshev_dist(self.pos, L)
    
        # Step 1: identify outward candidate cells
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        outward = [n for n in neighbors if self._chebyshev_dist(n, L) > current_dist]
        candidates = outward if outward else neighbors

        # Step 2: weight candidates by social attraction
        weights = [self._cell_attraction(c) for c in candidates]
        target = self.random.choices(candidates, weights=weights, k=1)[0]

        # Step 3: permeability gate --> move only if cell allows passage
        if self.random.random() < self.model.p_grid[target]:
            self.model.grid.move_agent(self, target)

        # Increment dispersal counter --> attempt shelter when duration is reached
        self.steps_dispersing += 1
        if self.steps_dispersing >= self.model.dispersal_duration:
            self._attempt_shelter()

    # helper for cell attraction
    def _cell_attraction(self, cell):
        """
        Compute the social attraction weight for a candidate cell.

        Weight = number of other active agents in the cell + 1.
        The +1 baseline ensures every cell has non-zero weight so that
        random.choices() never receives an all-zero weight list. 

        Counts DISPERSING, SHELTERED, and MOBILIZED agents as 'active' (visible social signal). 
        DEMOBILIZED agents are not counted — they provide no mobilizing signal to co-dispersers.
        """
        agents_here = self.model.grid.get_cell_list_contents([cell])
        n_active = sum(
            1 for a in agents_here
            if a is not self
            and a.state in (DISPERSING, SHELTERED, MOBILIZED)
        )
        return n_active + 1

    def _attempt_shelter(self):
        """
        Try to shelter at the agent's current cell after dispersal_duration steps.

        Three possible outcomes:
          1. Cell is over capacity (sheltered_here >= cell_capacity):
             move to a random adjacent cell and retry next step by resetting steps_dispersing to dispersal_duration - 1.
          2. Cell has space AND refuge density check passes (random < rho):
             transition to SHELTERED. Agent will evaluate threshold next step.
          3. Cell has space but refuge density check fails (random >= rho):
             transition to DEMOBILIZED. Agent exits collective action permanently
        """
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        sheltered_here = sum(1 for a in cellmates if a.state == SHELTERED)

        # Outcome 1: cell is full --> move and retry next step
        if sheltered_here >= self.model.cell_capacity:
            neighbors = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False
            )
            self.model.grid.move_agent(self, self.random.choice(neighbors))
            # Set to duration --> 1 so counter reaches duration again next step
            self.steps_dispersing = self.model.dispersal_duration - 1
            return

        # Outcome 2 or 3: attempt shelter via refuge density gate
        rho = self.model.rho_grid[self.pos]
        if self.random.random() < rho:
            self.state = SHELTERED      # found a refuge, will check threshold for next step
        else:
            self.state = DEMOBILIZED    # no refuge available, agent goes terminal home

    # instigator logic
    def _check_threshold(self):
        """
        Sheltered agent decides whether to remobilize based on local peer activity.

        Two rules applied in order:

        Rule 1 — Instigator rule:
            Agents with threshold <= instigator_threshold remobilize immediately
            regardless of peer counts. These are the 'always-on, reactive' activists whose unconditional commitment seeds the initial cascade for others.

        Rule 2 — Spatial threshold rule:
            Observe all agents within observation_radius and compute:
            local_rate = mobilized_peers / all_peers_in_observation_area
            If local_rate >= self.threshold, remobilize.

            observation_radius=0: observe own cell only (baseline behavior).
            observation_radius=1: observe Moore neighborhood, 9 cells.
            Higher radius = more information, but in sparse environments this can suppress mobilization by increasing the denominator with demobilized or absent agents.

        If no peers exist in the observation area, the agent waits (returns without changing state). This prevents division-by-zero and reflects the realistic condition that isolated agents cannot assess local rates.
        """
        if self.threshold <= self.model.instigator_threshold:
            self.state = MOBILIZED
            return

        # Rule 2: spatial threshold check at configured radius
        if self.model.observation_radius == 0:
            observe_cells = [self.pos]   # own cell only
        else:
            observe_cells = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=True,
                radius=self.model.observation_radius,
            )

        contents = self.model.grid.get_cell_list_contents(observe_cells)
        others = [a for a in contents if a is not self]  # exclude self from rate

        if not others:
            return  # isolated; cannot compute local rate

        mobilized_count = sum(1 for a in others if a.state == MOBILIZED)
        local_rate = mobilized_count / len(others)

        if local_rate >= self.threshold:
            self.state = MOBILIZED

    @staticmethod
    def _chebyshev_dist(a, b):
        """
        Compute Chebyshev (chessboard) distance between two grid positions.
        """
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
