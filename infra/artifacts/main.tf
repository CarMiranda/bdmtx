resource "google_storage_bucket" "dataset" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = true
  uniform_bucket_level_access = true
}

resource "google_service_account" "trainer" {
  account_id   = var.service_account_name
  display_name = "bdmtx training service account"
}

resource "google_storage_bucket_iam_member" "sa_storage_admin" {
  bucket = google_storage_bucket.dataset.name
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.trainer.email}"
}
resource "google_artifact_registry_repository" "training_images" {
  provider = google
  location  = var.region
  repository_id = "bdmtx-training-repo"
  format = "DOCKER"
  description = "Container images for bdmtx training"
}

resource "google_artifact_registry_repository_iam_member" "writer" {
  repository = google_artifact_registry_repository.training_images.name
  location   = google_artifact_registry_repository.training_images.location
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.trainer.email}"
}
