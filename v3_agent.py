import mesa

DORMANT = "dormant"
MOBILIZED = "mobilized"
DISPERSING = "dispersing"
SHELTERED = "sheltered"
DEMOBILIZED = "demobilized"


class ResidentAgent(mesa.Agent):
    def __init__(self, model, threshold):
        super().__init__(model)
        self.threshold = threshold
        self.state = DORMANT
        self.steps_dispersing = 0

    def step(self):
        if self.state == MOBILIZED:
            return
        elif self.state == DISPERSING:
            self._disperse_step()
        elif self.state == SHELTERED:
            self._check_threshold()

    def _disperse_step(self):
        L = self.model.repression_location
        current_dist = self._chebyshev_dist(self.pos, L)

        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        outward = [n for n in neighbors if self._chebyshev_dist(n, L) > current_dist]
        candidates = outward if outward else neighbors

        # weighted chance based on social attraction
        weights = [self._cell_attraction(c) for c in candidates]
        target = self.random.choices(candidates, weights=weights, k=1)[0]

        if self.random.random() < self.model.p_grid[target]:
            self.model.grid.move_agent(self, target)

        self.steps_dispersing += 1
        if self.steps_dispersing >= self.model.dispersal_duration:
            self._attempt_shelter()

    # helper for cell attraction
    def _cell_attraction(self, cell):
        agents_here = self.model.grid.get_cell_list_contents([cell])
        n_others = sum(
            1 for a in agents_here
            if a is not self
            and a.state in (DISPERSING, SHELTERED, MOBILIZED)
        )
        return n_others + 1

    def _attempt_shelter(self):
        rho = self.model.rho_grid[self.pos]
        if self.random.random() < rho:
            self.state = SHELTERED
        else:
            self.state = DEMOBILIZED

    # instigator logic
    def _check_threshold(self):
        if self.threshold <= self.model.instigator_threshold:
            self.state = MOBILIZED
            return

        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        others = [a for a in cellmates if a is not self]
        if not others:
            return

        mobilized_count = sum(1 for a in others if a.state == MOBILIZED)
        local_rate = mobilized_count / len(others)

        if local_rate >= self.threshold:
            self.state = MOBILIZED

    @staticmethod
    def _chebyshev_dist(a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
