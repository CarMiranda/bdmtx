# bdmtx CLI Commands

This document lists the primary CLI commands for the bdmtx project and examples of how to run them locally and on GCP.

## Local development

Prerequisites:
- Python 3.12, pip
- Install runtime and dev dependencies:

  python -m pip install --upgrade pip
  pip install -r requirements.txt
  pip install -r requirements-dev.txt

### Run pipeline on single image

Command:

  python -m bdmtx.cli.run <image_path>

Example:

  python -m bdmtx.cli.run images/sample.png

### Evaluate dataset

Command:

  bdmtx eval <dataset_root> [--out eval_results.json] [--max N]

Example:

  bdmtx eval ./data --out results.json --max 100

## Training

### Train enhancement model locally

Script:

  ./scripts/train_local.sh /path/to/dataset

This invokes bdmtx.train.train_enhancement_entry with the specified dataset path.

### Train segmentation (Ultralytics YOLO-seg)

Use the Ultralytics training helper to prepare a dataset YAML and launch training:

  python -c "from bdmtx.segmentation.train_yoloseg import make_ultralytics_dataset_yaml; make_ultralytics_dataset_yaml('path/to/data', 'dataset.yaml')"
  python -c "from bdmtx.segmentation.train_yoloseg import train_ultralytics; train_ultralytics('dataset.yaml')"


## GCP/containerized workflows

Artifacts and infra are under `infra/`. See infra/BUILD_IMAGE.md to build and push the training image to Artifact Registry.

### Run training container on trainer VM

On trainer VM (or via `docker run`):

  docker run --gpus all -v /workspace:/workspace -e GCS_BUCKET=your-bucket -e CHECKPOINT_PATH=/workspace/runs/train/exp/weights/last.pt REGION-docker.pkg.dev/PROJECT/REPO/bdmtx-train:TAG /workspace/infra/artifacts/train_entrypoint.sh /data /workspace /workspace/dataset.yaml

### Evaluate on trainer VM and upload results to GCS

Run evaluation inside container and copy results to GCS:

  bdmtx eval /workspace/data --out /workspace/results.json --max 500
  gsutil cp /workspace/results.json gs://your-bucket/results.json


## Notes

- Many commands assume required Python packages (ultralytics, torch, onnxruntime) are installed in the environment or container.
- For GCP runs, ensure the instance service account has permissions to write to the configured GCS bucket.
