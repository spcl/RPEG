# Model Workers

A worker owns model state and performs tensor operations on the devices assigned by the compiled plan.
The graph executor decides when an operation may run, while the worker implements that operation under its current parallel layout and weight version.

[base.py](base.py) defines the worker interface.
[megatron_worker.py](megatron_worker.py) implements model construction, generation, inference, training, optimizer updates, and state movement for the distributed backend.
The surrounding [semantic worker](../semantic_worker.py) manages operation requests and transfer endpoints, and the [controller](../semantic_controller.py) manages worker lifecycle and replay.

Workers may persist across graph nodes and reuse compatible serving state.
A transition to training must normalize or invalidate inference-only tensors, and a layout change must establish the destination's parameter and optimizer ownership before computation resumes.
[reshard.py](../reshard.py) and [optimizer_sharding.py](../optimizer_sharding.py) implement those state transformations.

FP32 master parameters and Adam state can remain resident, while operational BF16 or FP16 weights are released and rematerialized.
Separate serving workers retain their installed active version through consumption.
The analytical [worker-lifetime memory model](../../model/README.md#co-residency-aware-device-peaks) accounts for these distinct lifetimes and per-worker overhead.
