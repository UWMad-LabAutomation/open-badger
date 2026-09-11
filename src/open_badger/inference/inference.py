# Initial model-inference framework.
# Load the selected model through the registry and execute it with PyTorch.

# Implement:
# - model and runtime configuration loading;
# - evaluation mode, device placement, and inference_mode execution;
# - model-specific input preparation and action output decoding;
# - optional torch.compile after the correctness path is working;
# - backend selection for future ONNX and TensorRT implementations.
# CUDA Graphs belong to a later optimized backend, not the first implementation.
