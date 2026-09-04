# Open Badger

Early-stage framework for supervised robot policy training and inference.

The initial implementation will support model integrations, remote LeRobot data from S3, configurable fine-tuning, and a generic PyTorch training loop. Runtime backends such as TensorRT, ONNX Runtime, and JAX will come later.

## Status

Repository structure only. Implementation is intentionally pending.

## Layout

- `configs/` — experiment configuration
- `src/open_badger/` — framework package
- `scripts/` — training, evaluation, and benchmarking entry points
- `tests/` — unit and smoke tests
