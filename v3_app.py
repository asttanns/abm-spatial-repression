
from mesa.visualization import SolaraViz, make_space_component, make_plot_component, Slider
from mesa.visualization.components import AgentPortrayalStyle 
from v3_model import RepressionModel
from v3_agent import MOBILIZED, DISPERSING, SHELTERED, DEMOBILIZED, DORMANT

STATE_COLORS = {
    DORMANT:     "#BBBBBB",   
    MOBILIZED:   "#0077BB",   
    DISPERSING:  "#EE7733",   
    SHELTERED:   "#AA3377",   
    DEMOBILIZED: "#333333",   
}

def agent_portrayal(agent):
    """Map agent state to visual style."""
    return AgentPortrayalStyle(
        color=STATE_COLORS.get(agent.state, "#888780"),
        marker="o",
        size=30,
    )

model_params = {
    # rng
    "rng": {
        "type":  "InputText",
        "value": 42,
        "label": "rng",
    },
    
    # Urban morphology
    "env_type": {
        "type":   "Select",
        "value":  "kampung",
        "values": ["kampung", "mixed", "open"],
        "label":  "Environment typology",
    },

    # Population and threshold distribution
    "n_agents":             Slider("Agents", 200, 50, 500, 10),
    "threshold_mean":       Slider("Threshold mean (θ)", 0.3, 0.1, 0.9, 0.05),
    "threshold_std":        Slider("Threshold std", 0.15, 0.05, 0.3, 0.05),
    "instigator_threshold": Slider("Instigator threshold", 0.15, 0.0, 0.3, 0.05),

    # Repression parameters
    "repression_step":      Slider("Repression step", 1, 1, 20, 1),
    "dispersal_duration":   Slider("Dispersal duration", 3, 1, 10, 1),
                            # Higher dispersal_duration = harsher repression
                            # agents travel farther before sheltering, reducing density

    # State monitoring parameter
    "cell_capacity":        Slider("Cell capacity", 15, 5, 30, 1),
                            # Lower cell_capacity = more aggressive state monitoring
                            # smaller gatherings are broken up before cascade can seed

    # Observation radius: 0 = cell only, 1 = Moore neighborhood
    "observation_radius":   Slider("Observation radius (0=cell, 1=neighborhood)", 0, 0, 3, 1),

    # Simulation ceiling
    "max_steps":            Slider("Max steps", 100, 20, 300, 10),

    # Fixed grid dimensions
    "width": 30, "height": 30,
}

initial_model = RepressionModel()

page = SolaraViz(
    initial_model,
    components=[
        make_space_component(agent_portrayal),
        make_plot_component(["Mobilized", "Dispersing", "Sheltered", "Demobilized"]),
    ],
    model_params=model_params,
    name="Spatial Repression-Mobilization Model (V4)",
)
