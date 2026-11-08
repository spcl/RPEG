# Model Architecture Adapters

Model adapters translate a supported Hugging Face architecture description into the dimensions and ownership information needed by the runtime.
They keep model-family interpretation separate from graph scheduling and algorithm losses.

<div align="center">

| Module | Responsibility |
|---|---|
| [base.py](base.py) | Shared architecture and adapter interfaces |
| [registry.py](registry.py) | Select an adapter and load architecture metadata |
| [llama.py](llama.py) | Dense Llama-family architecture handling |
| [qwen3.py](qwen3.py) | Dense Qwen3 architecture handling |
| [hf.py](../hf.py) | Local or pinned snapshot resolution and parameter metadata |

</div>

A directory containing `config.json` is sufficient for the metadata examples to describe geometry.
Real execution additionally requires compatible weights and tokenizer files.
A Hugging Face source must identify an immutable revision, but already downloaded, local models are also supported.

RPEG and its runtime target dense decoder-only Transformers using DP, PP, and TP, while context parallelism, expert parallelism, and MoE are not supported.
Adding a family requires supplying its parameter names and shapes, attention geometry, checkpoint import/export, and supported heads as well as its configuration file.
