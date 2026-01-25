Cloud bdmtx training
---

This folder contains a minimal OpenTofu scaffold to provision resources for training on GCP following common deep learning best practices:

- Dedicated GCS bucket for datasets and artifacts
- Service account for training workloads
- Compute instance with GPU accelerator and startup script
- Firewall rule allowing SSH

## Quickstart

1. Install OpenTofu and authenticate with gcloud:
```bash
  gcloud auth application-default login
```

2. Copy `terraform.tfvars.example` -> `terraform.tfvars` and edit values.

3. Initialize and apply:
```bash
  tofu init
  tofu apply
```

## Notes & best practices

- In production prefer using instance templates + managed instance groups or AI Platform (Vertex AI) for scalable training.
- Do NOT store credentials in code. Use Workload Identity or provide least-privilege service accounts.
- Replace the startup script with a vetted driver and container runtime installation for your target image.


# GCP Run Guide — training and evaluation

This document details steps to provision the training VM, build and push the training image,
run a training job, and run evaluation with checkpoint upload on preemption.

Prerequisites

- `gcloud` CLI installed and authenticated (`gcloud auth login`)
- OpenTofu installed
- Docker installed and configured

Steps

1. Configure variables

```bash
  export PROJECT=your-gcp-project-id
  export REGION=us-central1
  export ZONE=us-central1-a
  export BUCKET=bdmtx-training-bucket-yourid
  export REPO=bdmtx-training-repo
  export IMAGE_TAG=latest
```

2. Create an artifact registry:
```bash
  cd infra
  tofu apply

  # or create artifact registry (if not using tofu to create it)
  gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION || true
  gcloud auth configure-docker ${REGION}-docker.pkg.dev
  ...
```

3. Build and push the training image
```bash
  docker build -t ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG} -f Dockerfile.train ..
  docker push ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG}
```

3. Provision infra with Terraform
```bash
  tofu init
  tofu apply -var="project=$PROJECT" -var="region=$REGION" -var="zone=$ZONE" -var="bucket_name=$BUCKET" -var="training_image=${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG}" -auto-approve
```

4. Run training on VM
```bash
   gcloud compute ssh bdmtx-trainer --zone $ZONE --command "sudo docker run --gpus all -v /workspace:/workspace -e GCS_BUCKET=$BUCKET ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG} /workspace/infra/artifacts/train_entrypoint.sh /data /workspace /workspace/dataset.yaml"
```

5. Monitor and fetch results

   - Check /workspace/runs/train inside the VM for checkpoints
   - On preemption the VM will run the shutdown script to upload the latest checkpoint to `gs://$BUCKET/checkpoints/`
   - After training, copy results: `gsutil cp gs://$BUCKET/checkpoints/* ./checkpoints/`

## Notes and recommendations

- Use preemptible GPUs for cost savings; ensure training checkpoints frequently (Ultralytics saves `last.pt` regularly).
- Verify that the training image includes a CUDA-compatible torch matching the VM drivers.
- Consider using a managed service (Vertex AI) for scalable/robust training.
