data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
}

resource "random_id" "suffix" {
  byte_length = 4
}

locals {
  resource_name              = "${var.project_name}-${random_id.suffix.hex}"
  db_dump_object_key         = "bootstrap/db.sql"
  db_password_parameter_name = "/${local.resource_name}/db-root-password"
  should_open_ssh            = var.ssh_key_name != null && length(var.ssh_ingress_cidr_blocks) > 0
  merged_tags = merge(
    {
      Name      = local.resource_name
      Project   = var.project_name
      ManagedBy = "terraform"
    },
    var.tags,
  )
}

resource "aws_vpc" "sqlagent" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = local.merged_tags
}

resource "aws_internet_gateway" "sqlagent" {
  vpc_id = aws_vpc.sqlagent.id

  tags = local.merged_tags
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.sqlagent.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = merge(local.merged_tags, { Name = "${local.resource_name}-public-subnet" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.sqlagent.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.sqlagent.id
  }

  tags = merge(local.merged_tags, { Name = "${local.resource_name}-public-rt" })
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "sqlagent" {
  name        = "${local.resource_name}-sg"
  description = "SQLAgent API access"
  vpc_id      = aws_vpc.sqlagent.id

  ingress {
    description = "SQLAgent HTTP API"
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = var.app_ingress_cidr_blocks
  }

  dynamic "ingress" {
    for_each = local.should_open_ssh ? [1] : []

    content {
      description = "SSH"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = var.ssh_ingress_cidr_blocks
    }
  }

  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.merged_tags
}

resource "aws_s3_bucket" "bootstrap" {
  bucket        = "${local.resource_name}-bootstrap"
  force_destroy = true

  tags = local.merged_tags
}

resource "aws_s3_bucket_public_access_block" "bootstrap" {
  bucket = aws_s3_bucket.bootstrap.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "bootstrap" {
  bucket = aws_s3_bucket.bootstrap.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_object" "db_dump" {
  bucket = aws_s3_bucket.bootstrap.id
  key    = local.db_dump_object_key
  source = var.db_dump_path
  etag   = filemd5(var.db_dump_path)
}

resource "aws_ssm_parameter" "db_root_password" {
  name  = local.db_password_parameter_name
  type  = "SecureString"
  value = var.db_root_password

  tags = local.merged_tags
}

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sqlagent" {
  name               = "${local.resource_name}-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json

  tags = local.merged_tags
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.sqlagent.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "bootstrap_access" {
  statement {
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.bootstrap.arn}/${local.db_dump_object_key}",
    ]
  }

  statement {
    actions = ["ssm:GetParameter"]
    resources = [
      aws_ssm_parameter.db_root_password.arn,
    ]
  }
}

resource "aws_iam_role_policy" "bootstrap_access" {
  name   = "${local.resource_name}-bootstrap"
  role   = aws_iam_role.sqlagent.id
  policy = data.aws_iam_policy_document.bootstrap_access.json
}

resource "aws_iam_instance_profile" "sqlagent" {
  name = "${local.resource_name}-profile"
  role = aws_iam_role.sqlagent.name
}

resource "aws_instance" "sqlagent" {
  ami                         = data.aws_ssm_parameter.al2023_ami.value
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.sqlagent.id]
  iam_instance_profile        = aws_iam_instance_profile.sqlagent.name
  key_name                    = var.ssh_key_name
  user_data_replace_on_change = true

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    app_port               = var.app_port
    bootstrap_bucket       = aws_s3_bucket.bootstrap.bucket
    db_dump_object_key     = aws_s3_object.db_dump.key
    db_name                = var.db_name
    db_password_parameter  = aws_ssm_parameter.db_root_password.name
    docker_compose_version = var.docker_compose_version
    ollama_model           = var.ollama_model
    project_ref            = var.project_ref
    project_repo_url       = var.project_repo_url
    swap_size_mb           = var.swap_size_mb
    mysql_user             = var.db_user
  })

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_size_gb
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required"
  }

  tags = local.merged_tags
}
