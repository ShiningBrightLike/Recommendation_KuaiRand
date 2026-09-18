"""Progressive Layered Extraction with Customized Gate Control."""

import numbers

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.saving import register_keras_serializable


def _require_positive_int(name, value):
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


@register_keras_serializable()
class PLECGC(Layer):
    """Progressive shared/task expert extraction with CGC gates.

    The layer accepts one shared representation followed by one representation
    per task and returns the next shared representation plus one representation
    per task. Each progressive layer owns independent experts and gates. A task
    gate mixes shared experts with that task's experts; the shared gate mixes
    shared experts with all task-specific experts for the next layer.
    """

    def __init__(
        self,
        num_tasks,
        num_layers=1,
        num_shared_experts=2,
        num_task_experts=2,
        expert_units=48,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_tasks = _require_positive_int("num_tasks", num_tasks)
        self.num_layers = _require_positive_int("num_layers", num_layers)
        self.num_shared_experts = _require_positive_int(
            "num_shared_experts", num_shared_experts
        )
        self.num_task_experts = _require_positive_int(
            "num_task_experts", num_task_experts
        )
        self.expert_units = _require_positive_int("expert_units", expert_units)

        # Keep every child in a flat, eagerly assigned list. Keras tracks flat
        # lists of layers reliably; nested lists or append-after-assignment can
        # leave expert kernels out of a `.keras` archive.
        self.shared_experts = [
            Dense(
                self.expert_units,
                activation="relu",
                name=f"layer_{layer_index}_shared_expert_{expert_index}",
            )
            for layer_index in range(self.num_layers)
            for expert_index in range(self.num_shared_experts)
        ]
        self.task_experts = [
            Dense(
                self.expert_units,
                activation="relu",
                name=(
                    f"layer_{layer_index}_task_{task_index}_expert_{expert_index}"
                ),
            )
            for layer_index in range(self.num_layers)
            for task_index in range(self.num_tasks)
            for expert_index in range(self.num_task_experts)
        ]
        self.shared_gates = [
            Dense(
                self.num_shared_experts + self.num_tasks * self.num_task_experts,
                activation="softmax",
                name=f"layer_{layer_index}_shared_gate",
            )
            for layer_index in range(self.num_layers)
        ]
        self.task_gates = [
            Dense(
                self.num_shared_experts + self.num_task_experts,
                activation="softmax",
                name=f"layer_{layer_index}_task_{task_index}_gate",
            )
            for layer_index in range(self.num_layers)
            for task_index in range(self.num_tasks)
        ]

    def build(self, input_shape):
        """Build every child explicitly so loading never creates lazy weights."""
        initial_shared_shape = tf.TensorShape(input_shape[0])
        initial_task_shapes = [
            tf.TensorShape(shape) for shape in input_shape[1:]
        ]
        for layer_index in range(self.num_layers):
            shared_shape = (
                initial_shared_shape
                if layer_index == 0
                else tf.TensorShape((None, self.expert_units))
            )
            task_shapes = (
                initial_task_shapes
                if layer_index == 0
                else [tf.TensorShape((None, self.expert_units))] * self.num_tasks
            )
            shared_start = layer_index * self.num_shared_experts
            for expert in self.shared_experts[
                shared_start : shared_start + self.num_shared_experts
            ]:
                expert.build(shared_shape)
            task_start = layer_index * self.num_tasks * self.num_task_experts
            for task_index, task_shape in enumerate(task_shapes):
                start = task_start + task_index * self.num_task_experts
                for expert in self.task_experts[start : start + self.num_task_experts]:
                    expert.build(task_shape)
            self.shared_gates[layer_index].build(shared_shape)
            gate_start = layer_index * self.num_tasks
            for task_index, task_shape in enumerate(task_shapes):
                self.task_gates[gate_start + task_index].build(task_shape)
        super().build(input_shape)

    @staticmethod
    def _mix(gate_weights, candidates):
        stacked = tf.stack(candidates, axis=1)
        return tf.reduce_sum(stacked * tf.expand_dims(gate_weights, axis=-1), axis=1)

    def call(self, inputs):
        if len(inputs) != self.num_tasks + 1:
            raise ValueError(
                f"PLECGC expects one shared input plus {self.num_tasks} task inputs, "
                f"got {len(inputs)}"
            )
        shared_input = inputs[0]
        task_inputs = list(inputs[1:])

        for layer_index in range(self.num_layers):
            shared_start = layer_index * self.num_shared_experts
            shared_expert_outputs = [
                expert(shared_input)
                for expert in self.shared_experts[
                    shared_start : shared_start + self.num_shared_experts
                ]
            ]
            task_expert_outputs = []
            task_start = layer_index * self.num_tasks * self.num_task_experts
            for task_index in range(self.num_tasks):
                start = task_start + task_index * self.num_task_experts
                task_expert_outputs.append(
                    [
                        expert(task_inputs[task_index])
                        for expert in self.task_experts[
                            start : start + self.num_task_experts
                        ]
                    ]
                )

            all_expert_outputs = shared_expert_outputs + [
                output
                for task_outputs in task_expert_outputs
                for output in task_outputs
            ]
            shared_weights = self.shared_gates[layer_index](shared_input)
            next_shared = self._mix(shared_weights, all_expert_outputs)

            next_tasks = []
            for task_index, task_input in enumerate(task_inputs):
                candidates = shared_expert_outputs + task_expert_outputs[task_index]
                task_gate_index = layer_index * self.num_tasks + task_index
                task_weights = self.task_gates[task_gate_index](task_input)
                next_tasks.append(self._mix(task_weights, candidates))

            shared_input = next_shared
            task_inputs = next_tasks

        return [shared_input, *task_inputs]

    def get_config(self):
        return {
            **super().get_config(),
            "num_tasks": self.num_tasks,
            "num_layers": self.num_layers,
            "num_shared_experts": self.num_shared_experts,
            "num_task_experts": self.num_task_experts,
            "expert_units": self.expert_units,
        }


def build_ple_cgc_structure(
    features,
    num_tasks=2,
    num_layers=1,
    num_shared_experts=2,
    num_task_experts=2,
    expert_units=48,
    tower_units=32,
):
    """Build PLE-CGC and one sigmoid tower per task."""
    ple = PLECGC(
        num_tasks=num_tasks,
        num_layers=num_layers,
        num_shared_experts=num_shared_experts,
        num_task_experts=num_task_experts,
        expert_units=expert_units,
        name="ple_cgc",
    )
    ple_outputs = ple([features] + [features] * num_tasks)
    task_features = ple_outputs[1:]
    outputs = []
    for task_index, task_feature in enumerate(task_features):
        tower = Dense(
            tower_units,
            activation="relu",
            name=f"ple_cgc_tower_{task_index + 1}",
        )(task_feature)
        outputs.append(
            Dense(1, activation="sigmoid", name=f"output_{task_index + 1}")(tower)
        )
    return outputs
