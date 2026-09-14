"""Shared-Bottom multi-task structure: one trunk, one tower per task."""

from tensorflow.keras.layers import Dense


def build_shared_bottom_structure(features, num_tasks=2, bottom_units=64, tower_units=32):
    """One shared trunk over the encoded features, then one tower per task."""
    shared = Dense(bottom_units, activation="relu", name="shared_bottom")(features)
    outputs = []
    for i in range(num_tasks):
        tower = Dense(tower_units, activation="relu", name=f"tower_{i + 1}")(shared)
        outputs.append(Dense(1, activation="sigmoid", name=f"output_{i + 1}")(tower))
    return outputs

