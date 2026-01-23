locals {
  default_network = "default"
}

resource "google_compute_network" "default" {
  name = local.default_network
}

resource "google_compute_subnetwork" "default" {
  name          = "bdmtx-subnet"
  ip_cidr_range = "10.10.0.0/16"
  region        = var.region
  network       = google_compute_network.default.name
}

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

resource "google_compute_firewall" "allow_ssh" {
  name    = "bdmtx-allow-ssh"
  network = google_compute_network.default.name
  allow {
    protocol = "tcp"
    ports    = ["22"]
  }
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["bdmtx-training"]
}

resource "google_compute_instance" "trainer" {
  name         = "bdmtx-trainer"
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
      size  = 200
    }
  }

  network_interface {
    network = google_compute_network.default.name
    access_config {}
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
  }

  metadata_startup_script = file("startup.sh")

  service_account {
    email  = google_service_account.trainer.email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  tags = ["bdmtx-training"]

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  lifecycle {
    create_before_destroy = true
  }
}
