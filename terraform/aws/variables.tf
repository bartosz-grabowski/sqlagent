variable "aws_region" {
  description = "AWS region used for the SQLAgent deployment."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix used for AWS resource names and tags."
  type        = string
  default     = "sqlagent"
}

variable "project_repo_url" {
  description = "Git repository that the EC2 instance should clone during bootstrap."
  type        = string
  default     = "https://github.com/bartosz-grabowski/sqlagent.git"
}

variable "project_ref" {
  description = "Git branch or tag to deploy on the EC2 instance."
  type        = string
  default     = "main"
}

variable "instance_type" {
  description = "EC2 instance type for the single-host deployment."
  type        = string
  default     = "t3.small"
}

variable "root_volume_size_gb" {
  description = "Size of the EC2 root volume in GiB."
  type        = number
  default     = 24
}

variable "swap_size_mb" {
  description = "Swap file size in MiB."
  type        = number
  default     = 2048
}

variable "app_port" {
  description = "Public HTTP port exposed by the SQLAgent API."
  type        = number
  default     = 8000
}

variable "app_ingress_cidr_blocks" {
  description = "CIDR blocks allowed to reach the SQLAgent HTTP API."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "ssh_key_name" {
  description = "Optional EC2 key pair name. Leave null to use SSM Session Manager only."
  type        = string
  default     = null
}

variable "ssh_ingress_cidr_blocks" {
  description = "CIDR blocks allowed to reach SSH when ssh_key_name is set."
  type        = list(string)
  default     = []
}

variable "db_name" {
  description = "MySQL database name created inside the container."
  type        = string
  default     = "sqlagent_db"
}

variable "db_user" {
  description = "MySQL user used by the application."
  type        = string
  default     = "root"
}

variable "db_dump_path" {
  description = "Local path to the SQL dump file uploaded to S3 during terraform apply."
  type        = string
}

variable "db_root_password" {
  description = "MySQL root password stored in SSM Parameter Store and mounted into Docker."
  type        = string
  sensitive   = true
}

variable "ollama_model" {
  description = "Ollama model to pull on the AWS instance."
  type        = string
  default     = "qwen2.5:0.5b-instruct-q5_0"
}

variable "docker_compose_version" {
  description = "Docker Compose CLI plugin version installed by cloud-init."
  type        = string
  default     = "2.27.0"
}

variable "tags" {
  description = "Additional tags added to created AWS resources."
  type        = map(string)
  default     = {}
}
