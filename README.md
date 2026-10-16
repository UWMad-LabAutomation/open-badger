# Open Badger

Open Badger is an early-stage framework for training and running robot policies.
The long-term goal is to support multiple model families, including VLA policies,
diffusion policies, ACT, and future world-model and specialized VLM workflows.
The current integration focus is supervised fine-tuning with MolmoAct2.

## Current status

- A canonical data schema and LeRobot v3 reader normalize dataset metadata and
  provide direct frame references.
- Valid training-example references are streamed to a Parquet manifest. The
  manifest stores frame addresses and action-chunk offsets, not decoded tensors.
- The manifest-backed loader resolves observations and action chunks lazily.
  Parquet row groups are divided among workers so they do not all scan the whole
  manifest.
- The MolmoAct2 adapter uses LeRobot's official preprocessing pipeline for
  normalization, camera/text input preparation, and action formatting.
- `scripts/train.py` composes the job, model, and data YAML configs. It supports
  a one-batch VLA forward/loss check (`--vla-smoke-test`) and a one-batch
  action-prediction check (`--vla-inference-smoke-test`). The previous
  `--smoke-test` and `--inference-smoke-test` names remain accepted aliases.
- `./container.sh` builds and runs the CUDA development image and keeps the
  Hugging Face cache in the repository-mounted `.cache/huggingface` directory.

The real-checkpoint, real-dataset smoke tests are intended to validate the
integration on a GPU. Automated tests currently cover the data, preprocessing,
and orchestration contracts with test fixtures.

## Quick start

Build the development image on a host with Docker and NVIDIA Container Toolkit:

```bash
./container.sh build
```

For a small local test, download one dataset directory from the LeRobot community
collection:

```bash
hf download lerobot/community_dataset_v3 \
  --repo-type dataset \
  --local-dir datasets/community_dataset_v3 \
  --include '00ri/so100_test/**'
```

Prepare the local dataset by adding missing state/action quantile statistics
without uploading or modifying the Hugging Face source:

```bash
./container.sh run python3 scripts/prepare_dataset.py \
  --backend local \
  --dataset-id local \
  --data-root /workspace/open-badger/datasets/community_dataset_v3/00ri/so100_test
```

Then run an action-prediction smoke test:

```bash
./container.sh run python3 scripts/train.py \
  --backend local \
  --dataset-id local \
  --data-root /workspace/open-badger/datasets/community_dataset_v3/00ri/so100_test \
  --episodes 0 \
  --image-keys observation.images.top \
  --normalize-gripper \
  --manifest-dir /workspace/open-badger/outputs/manifests/00ri-so100-test-ep0 \
  --batch-size 1 \
  --num-workers 1 \
  --vla-inference-smoke-test
```

The first run downloads the MolmoAct2 checkpoint. Subsequent container runs
reuse the Hugging Face cache. Use `--vla-smoke-test` instead to run the
supervised forward pass and loss calculation without backward or optimizer
updates.

## Training metrics and dataset passes

The training loop logs `train/loss`, scalar model metrics, and optimizer-step
duration to the console every `training.logging.every_steps` updates. TensorBoard
and Weights & Biases are optional and disabled by default. Enable either from
the CLI:

```bash
--tensorboard
--wandb
```

TensorBoard events are written under `outputs/tensorboard/<run-name>` by
default; override the location with `--tensorboard-dir`. To view them, run:

```bash
./container.sh run tensorboard \
  --logdir /workspace/open-badger/outputs/tensorboard \
  --bind_all
```

For W&B online logging, export `WANDB_API_KEY` on the host before running the
container. The wrapper forwards it without writing the key to the repository.
Use `--wandb-mode offline` to save a local W&B run under `outputs/wandb`.

`--max-updates` (also available as `--max-steps`) is the target number of
training updates; the loader repeats the
manifest if it is exhausted before that count. With the current 564-example
manifest, batch size 1, and gradient accumulation 1, `--max-updates 564` consumes
one full manifest pass. The model YAML's current default is 50,000 updates, so
set the update limit explicitly for a bounded end-to-end training trial.

Training saves a checkpoint every `training.checkpointing.save_every_steps`
training updates and at the end of the run. By default, checkpoints are written under
`outputs/checkpoints/<model>/<dataset>/<strategy>`; `--checkpoint-dir` overrides
the root. Full fine-tuning writes `model.safetensors`; LoRA writes
`adapter_model.safetensors` containing the trainable adapters and action expert;
action-expert-only writes `action_expert.safetensors`. Each checkpoint also
contains optimizer/RNG state for resuming and a metadata file identifying the
model strategy and dataset revision. These are Open Badger resume artifacts,
not yet a standalone PEFT deployment bundle. Only the newest
`training.checkpointing.keep_last_n` checkpoints are retained.

Resume from the latest checkpoint in that run directory with
`--resume-from latest`, or pass a specific checkpoint directory. When resuming,
the update limit is still the target total, not the number of additional
training updates. To restore the exact position in the current iterable
manifest loader, resume with at most one data-loader worker or use batch size 1;
other multi-worker, multi-example batches are rejected rather than silently
replaying the wrong part of the manifest.

## MolmoAct2 fine-tuning strategies

`--strategy lora` versus `--strategy full_finetune` selects the VLM/backbone
fine-tuning method. Independently, `--action-expert-only` restricts training to
the action expert, regardless of the selected method. Without that modifier:

- `lora` trains LoRA adapters on transformer and vision-backbone target layers,
  plus the action expert.
- `full_finetune` trains the model parameters, including the vision backbone.
  The current `freeze_embedding: true` setting keeps input embeddings frozen.

There is not currently a "train all except the vision encoder" option. LeRobot's
LoRA optimizer groups previously supplied hard-coded rates for some groups;
its `vlm_params` bucket is the language transformer, not the vision encoder.
Open Badger keeps those groups distinct and applies
`language_learning_rate`, `vision_learning_rate`, `connector_learning_rate`,
and `action_expert_learning_rate` to the corresponding parameters.

## Configuration

- [`configs/training/train.yaml`](configs/training/train.yaml) selects the model,
  fine-tuning strategy, data config, and shared experiment-logging defaults.
- [`configs/models/vla/molmoact2.yaml`](configs/models/vla/molmoact2.yaml)
  contains the checkpoint, runtime, action limits, and training settings.
- [`configs/data/data_reader.yaml`](configs/data/data_reader.yaml) contains the
  data source, sampling, manifest, and loader settings.

`scripts/train.py --help` lists command-line overrides, including dataset root,
episodes, camera keys, gripper normalization, batch size, and worker count.

## Current limitations

- One LeRobot dataset is opened per run; combining datasets is not implemented.
- Local and Hugging Face sources are supported by the current entry point. S3,
  SageMaker, and CHTC staging/launch workflows are not wired end to end.
- The generic loop currently uses a constant learning rate and has no
  validation or distributed-training path. `finetuning.scheduler`,
  `training.validation`, and `training.mixed_precision` are not consumed;
  `runtime.dtype` sets the model dtype, but the loop does not add an AMP
  autocast/scaler context.
- `runtime.backend`, `runtime.requires_grad`, and the `optimization` block are
  placeholders and do not affect execution. Trainable parameters are selected
  by the model's fine-tuning strategy.
- MolmoAct2 history windows beyond one observation frame are not yet assembled
  by the universal loader.

## Repository layout

```text
configs/                 Model, data, training, launch, and distributed settings
scripts/                 Training entry point and cloud-job scaffolding
src/open_badger/
├── data/                Schemas, readers, sampling, manifests, and datasets
├── inference/           Inference scaffolding
├── logging/             Logging scaffolding
├── models/              Model registry and MolmoAct2 adapter
└── training/            DataLoader composition and training loop
tests/                   Data, model, and training contract tests
tools/docker/            CUDA development container
```
