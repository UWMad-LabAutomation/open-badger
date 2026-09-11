# PyTorch training framework.
# This module owns the training loop rather than model-specific behavior.

# Implement:
# - loading model and training-strategy YAML configuration;
# - model construction through models.registry;
# - LoRA, QLoRA, DoRA, and full-fine-tuning parameter setup;
# - forward, loss, backward, gradient accumulation, and optimizer updates;
# - validation mode that computes loss without applying updates;
# - DDP/FSDP selection using configs/distributed settings;
# - checkpointing and configured W&B or TensorBoard logging.
# Keep the first implementation small enough to validate with MolmoAct2.
