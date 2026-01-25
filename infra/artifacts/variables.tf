variable "project" {
  description = "GCP project id"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "bucket_name" {
  description = "GCS bucket for datasets and artifacts"
  type        = string
}

variable "service_account_name" {
  description = "Name for the training service account"
  type        = string
  default     = "bdmtx-trainer-sa"
}

variable "machine_type" {
  description = "Machine type for training instance"
  type        = string
  default     = "n1-standard-8"
}

variable "gpu_type" {
  description = "GPU accelerator type (e.g., nvidia-tesla-t4)"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 1
}

variable "training_image" {
  description = "Container image to run training (gcr or docker hub)"
  type        = string
  default     = ""
}
