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
  a one-batch supervised forward/loss check (`--smoke-test`) and a one-batch
  action-prediction check (`--inference-smoke-test`).
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
  --inference-smoke-test
```

The first run downloads the MolmoAct2 checkpoint. Subsequent container runs
reuse the Hugging Face cache. Use `--smoke-test` instead to run the supervised
forward pass and loss calculation without backward or optimizer updates.

## Configuration

- [`configs/training/train.yaml`](configs/training/train.yaml) selects the model,
  fine-tuning strategy, and data config.
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
- The training loop supports gradient accumulation and optimizer updates, but
  checkpoint management, scheduling, experiment logging, and distributed
  training still need implementation.
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
