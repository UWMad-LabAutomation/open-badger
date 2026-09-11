# Main training-job entry point.

# Implement configuration loading and command-line selection of a training
# strategy, model, distributed mode, and execution target.
# Run locally or on an EC2 host reached through SSH.
# If a cloud target is requested, delegate job construction and submission
# to scripts/cloud_jobs.py while reusing the same training entry point.
# Keep training logic in open_badger.training.training.
