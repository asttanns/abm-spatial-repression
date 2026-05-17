import mesa
import numpy as np

from v3_agent import (
    ResidentAgent,
    DORMANT, MOBILIZED, DISPERSING, SHELTERED, DEMOBILIZED,
)

ENV_PARAMS = {
    "kampung": {"rho": 0.85, "p": 0.80},
    "mixed":   {"rho": 0.45, "p": 0.50},
    "open":    {"rho": 0.15, "p": 0.25},
}


class RepressionModel(mesa.Model):
    def __init__(
        self,
        env_type="kampung",
        width=30,
        height=30,
        n_agents=200,
        threshold_mean=0.3,        
        threshold_std=0.15,
        instigator_threshold=0.15,  
        repression_step=5,
        dispersal_duration=3,
        repression_jitter=2,
        max_steps=100,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.env_type = env_type
        self.width = width
        self.height = height
        self.n_agents = n_agents
        self.threshold_mean = threshold_mean
        self.threshold_std = threshold_std
        self.instigator_threshold = instigator_threshold
        self.repression_step = repression_step
        self.dispersal_duration = dispersal_duration
        self.max_steps = max_steps
        self.repression_triggered = False
        self.running = True

        self.grid = mesa.space.MultiGrid(width, height, torus=False)

        env = ENV_PARAMS[env_type]
        self.rho_grid = np.full((width, height), env["rho"])
        self.p_grid = np.full((width, height), env["p"])

        cx, cy = width // 2, height // 2
        jx = self.random.randint(-repression_jitter, repression_jitter)
        jy = self.random.randint(-repression_jitter, repression_jitter)
        self.repression_location = (cx + jx, cy + jy)

        thresholds = self.rng.normal(
            loc=threshold_mean, scale=threshold_std, size=n_agents
        )
        thresholds = np.clip(thresholds, 0.0, 1.0)

        for theta in thresholds:
            agent = ResidentAgent(self, threshold=float(theta))
            agent.state = MOBILIZED
            self.grid.place_agent(agent, self.repression_location)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Mobilized":   lambda m: m._count_state(MOBILIZED),
                "Dispersing":  lambda m: m._count_state(DISPERSING),
                "Sheltered":   lambda m: m._count_state(SHELTERED),
                "Demobilized": lambda m: m._count_state(DEMOBILIZED),
                "MobilizedProportion": lambda m: m._count_state(MOBILIZED) / m.n_agents,
                "Step": lambda m: m.steps,
            },
        )
        self.datacollector.collect(self)

    def step(self):
        if self.steps == self.repression_step and not self.repression_triggered:
            self._trigger_repression()
            self.repression_triggered = True

        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
        self._check_stopping_rules()

    def _trigger_repression(self):
        affected = self.grid.get_cell_list_contents([self.repression_location])
        for agent in affected:
            if agent.state == MOBILIZED:
                agent.state = DISPERSING
                agent.steps_dispersing = 0

    def _check_stopping_rules(self):
        if self.steps >= self.max_steps:
            self.running = False
            return
        if self.repression_triggered:
            active = [a for a in self.agents
                      if a.state in (MOBILIZED, DISPERSING, SHELTERED)]
            if len(active) == 0:
                self.running = False

    def _count_state(self, state):
        return sum(1 for a in self.agents if a.state == state)

    # post-repression metrics, not global peak
    def get_outcome_metrics(self):
        df = self.datacollector.get_model_vars_dataframe()
        post_rep = df[df["Step"] > self.repression_step].reset_index(drop=True)

        if len(post_rep) == 0:
            return {
                "peak_post_repression": 0.0, "time_to_peak": 0,
                "persistence": 0, "reconstitution_rate": 0.0,
                "env_type": self.env_type, "threshold_mean": self.threshold_mean,
                "threshold_std": self.threshold_std,
            }

        mob_prop = post_rep["MobilizedProportion"]
        return {
            "peak_post_repression": float(mob_prop.max()),
            "time_to_peak": int(mob_prop.idxmax()),
            "persistence": int((mob_prop > 0.1).sum()),
            "reconstitution_rate": float(mob_prop.iloc[-1]),
            "env_type": self.env_type,
            "threshold_mean": self.threshold_mean,
            "threshold_std": self.threshold_std,
        }
