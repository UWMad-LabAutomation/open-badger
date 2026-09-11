# Open Badger

Early-stage framework for supervised robot policy training and inference.

The project is being built around configurable PyTorch training and inference,
with ONNX Runtime and TensorRT planned as later deployment backends.

## Purpose

Open Badger is a research framework for training and running vision-language-action
policies for robots. Its initial focus is fine-tuning MolmoAct2 with PyTorch while
providing a consistent configuration and job-launch workflow across local machines,
EC2, SageMaker, and CHTC. Over time, the framework will support optimized inference
backends such as ONNX Runtime and TensorRT for deployment.

## Progress

- Python package setup is in place under `src/open_badger/`.
- CUDA-enabled development container builds and runs through `container.sh`.
- MolmoAct2 model configuration loads its checkpoint from Hugging Face.
- Training configurations are separated into LoRA and full fine-tuning.
- Source responsibilities are organized around inference, logging, models, and
  training.
- Local/EC2 training and SageMaker/CHTC job-launch entry points are scaffolded.

## Layout

```text
configs/
├── data/
├── distributed/
├── launch/
├── models/
└── training/
    ├── lora.yaml
    └── full_finetune.yaml

src/open_badger/
├── inference/
├── logging/
├── models/
└── training/

scripts/
├── train.py
└── cloud_jobs.py
```
The core training and inference behavior is still to be implemented incrementally.
