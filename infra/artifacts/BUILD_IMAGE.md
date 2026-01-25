Building and pushing the training image to Artifact Registry

Assumes `gcloud` is installed and configured for the target GCP project and region.

1. Configure variables
```bash
   export PROJECT=your-gcp-project-id
   export REGION=us-central1
   export REPO=bdmtx-training-repo
   export IMAGE_TAG=latest
```

2. Create the Artifact Registry repository (if not created by Terraform):
```bash
   gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION --description="bdmtx training images"
```

3. Configure docker auth for Artifact Registry:
```bash
   gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

4. Build the Docker image locally:
```bash
   docker build -t ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG} -f infra/Dockerfile.train .
```

5. Push the image:

```bash
   docker push ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG}
```

6. (Optional) If using Tofu variables, set `training_image` to the pushed image path in terraform.tfvars.

Notes:
- For production, use a CI pipeline to build and push images with proper signing and vulnerability scanning.
