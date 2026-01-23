GCP Terraform for bdmtx training

This folder contains a minimal OpenTofu scaffold to provision resources for training on GCP following common deep learning best practices:

- Dedicated GCS bucket for datasets and artifacts
- Service account for training workloads
- Compute instance with GPU accelerator and startup script
- Firewall rule allowing SSH

Quickstart

1. Install OpenTofu and authenticate with gcloud:
   gcloud auth application-default login

2. Copy terraform.tfvars.example -> terraform.tfvars and edit values.

3. Initialize and apply:
  ```bash
    tofu init
    tofu apply
  ```

Notes & best practices

- In production prefer using instance templates + managed instance groups or AI Platform (Vertex AI) for scalable training.
- Do NOT store credentials in code. Use Workload Identity or provide least-privilege service accounts.
- Replace the startup script with a vetted driver and container runtime installation for your target image.
- Consider using preemptible (spot) GPUs for cost savings and checkpointing during training.
