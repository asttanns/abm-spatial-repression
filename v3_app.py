
from mesa.visualization import SolaraViz, make_space_component, make_plot_component, Slider
from v3_model import RepressionModel
from v3_agent import MOBILIZED, DISPERSING, SHELTERED, DEMOBILIZED, DORMANT

STATE_COLORS = {
    DORMANT:     "#B4B2A9",
    MOBILIZED:   "#D85A30",
    DISPERSING:  "#EF9F27",
    SHELTERED:   "#5DCAA5",
    DEMOBILIZED: "#444441",
}

def agent_portrayal(agent):
    return {"color": STATE_COLORS.get(agent.state, "#888780"), "marker": "o", "size": 30}

model_params = {
    "env_type": {
        "type": "Select", "value": "kampung",
        "values": ["kampung", "mixed", "open"], "label": "Environment typology",
    },
    "n_agents": Slider(label="Agents", value=200, min=50, max=500, step=10),
    "threshold_mean": Slider(label="Threshold mean (θ)", value=0.3, min=0.1, max=0.9, step=0.05),
    "threshold_std": Slider(label="Threshold std", value=0.15, min=0.05, max=0.3, step=0.05),
    "instigator_threshold": Slider(label="Instigator threshold", value=0.15, min=0.0, max=0.3, step=0.05),
    "repression_step": Slider(label="Repression step", value=5, min=1, max=20, step=1),
    "dispersal_duration": Slider(label="Dispersal duration", value=3, min=1, max=10, step=1),
    "max_steps": Slider(label="Max steps", value=100, min=20, max=300, step=10),
    "width": 30, "height": 30, "seed": 42,
}

initial_model = RepressionModel()

page = SolaraViz(
    initial_model,
    components=[
        make_space_component(agent_portrayal),
        make_plot_component(["Mobilized", "Dispersing", "Sheltered", "Demobilized"]),
    ],
    model_params=model_params,
    name="Versi 3 (Final): Cascade ignite di kampung",
)
