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
