# Central logging integration for training jobs.
# Keep experiment backends behind one interface used by the trainer.

# Implement:
# - a common metric and run-metadata interface;
# - W&B and TensorBoard backends selected by configuration;
# - lifecycle handling for start, log, flush, and close.
# Logging configuration belongs to the training job, not model code.
