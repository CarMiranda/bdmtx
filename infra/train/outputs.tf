output "bucket_name" {
  value = google_storage_bucket.dataset.name
}

output "trainer_instance" {
  value = google_compute_instance.trainer.name
}

output "service_account_email" {
  value = google_service_account.trainer.email
}
