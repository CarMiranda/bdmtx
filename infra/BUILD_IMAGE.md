Building and pushing the training image to Artifact Registry

Assumes gcloud is installed and configured for the target GCP project and region.

1. Configure variables

   export PROJECT=your-gcp-project-id
   export REGION=us-central1
   export REPO=bdmtx-training-repo
   export IMAGE_TAG=latest

2. Create the Artifact Registry repository (if not created by Terraform):

   gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION --description="bdmtx training images"

3. Configure docker auth for Artifact Registry:

   gcloud auth configure-docker ${REGION}-docker.pkg.dev

4. Build the Docker image locally:

   docker build -t ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG} -f infra/Dockerfile.train .

5. Push the image:

   docker push ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG}

6. (Optional) If using Terraform variables, set training_image to the pushed image path in terraform.tfvars.

Notes:
- The Dockerfile pulls requirements.txt from repo root and packages the workspace into the image.
- For production, use a CI pipeline to build and push images with proper signing and vulnerability scanning.
