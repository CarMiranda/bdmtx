GCP Run Guide — training and evaluation

This document details steps to provision the training VM, build and push the training image,
run a training job, and run evaluation with checkpoint upload on preemption.

Prerequisites

- gcloud CLI installed and authenticated (gcloud auth login)
- Terraform installed
- Docker installed and configured

Steps

1. Configure variables

   export PROJECT=your-gcp-project-id
   export REGION=us-central1
   export ZONE=us-central1-a
   export BUCKET=bdmtx-training-bucket-yourid
   export REPO=bdmtx-training-repo
   export IMAGE_TAG=latest

2. Build and push the training image

   cd infra
   # create artifact registry (if not using Terraform to create it)
   gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION || true
   gcloud auth configure-docker ${REGION}-docker.pkg.dev
   docker build -t ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG} -f Dockerfile.train ..
   docker push ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG}

3. Provision infra with Terraform

   terraform init
   terraform apply -var="project=$PROJECT" -var="region=$REGION" -var="zone=$ZONE" -var="bucket_name=$BUCKET" -var="training_image=${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG}" -auto-approve

4. Run training on VM

   gcloud compute ssh bdmtx-trainer --zone $ZONE --command "sudo docker run --gpus all -v /workspace:/workspace -e GCS_BUCKET=$BUCKET ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG} /workspace/infra/artifacts/train_entrypoint.sh /data /workspace /workspace/dataset.yaml"

5. Monitor and fetch results

   - Check /workspace/runs/train inside the VM for checkpoints
   - On preemption the VM will run the shutdown script to upload the latest checkpoint to gs://$BUCKET/checkpoints/
   - After training, copy results: gsutil cp gs://$BUCKET/checkpoints/* ./checkpoints/

Notes and recommendations

- Use preemptible GPUs for cost savings; ensure training checkpoints frequently (Ultralytics saves last.pt regularly).
- Verify that the training image includes a CUDA-compatible torch matching the VM drivers.
- Consider using a managed service (Vertex AI) for scalable/robust training.
