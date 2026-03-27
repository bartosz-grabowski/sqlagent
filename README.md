# SQLAgent

SQLAgent is a lightweight learning project that answers natural-language questions about a MySQL database by using an LLM to inspect schema, generate SQL, execute the query, and summarize the result.

The project now supports two deployment modes:

- Local Docker deployment with a locally hosted Ollama model.
- Optional AWS deployment with Terraform, where Ollama runs on an EC2 instance.

## Disclaimer

This project is intended for learning purposes only.
It is not designed or guaranteed for production use.
Any deployment or usage is done at your own risk.

## Features

- Natural-language to SQL querying against a MySQL database.
- Docker-based local development workflow.
- Configurable Ollama endpoint, so the app can target either local or cloud-hosted Ollama.
- Optional AWS infrastructure managed with Terraform.
- Health endpoint at `GET /health` for deployment checks.

## Requirements

### Local mode

- Docker
- Docker Compose

### AWS mode

- Terraform 1.6+
- An AWS account
- AWS CLI credentials configured locally
- A Git-accessible copy of this repository

## Local Deployment

### 1. Add your database dump

For a quick test, copy the included example dump:

```bash
cp db/db.sql.example db/db.sql
```

If you want to use your own data instead, place your SQL dump file at `./db/db.sql`.

### 2. Set the MySQL password

Store the MySQL root password in `./db/passwd.txt`.

Make sure the file contains only the password and no trailing newline:

```bash
echo -n "your_password_here" > db/passwd.txt
```

### 3. Optionally choose a local model

The default local model is `gpt-oss:20b`.

```bash
export OLLAMA_MODEL=model_identifier
```

### 4. Start the local stack

```bash
docker compose up
```

This starts:

- `ollama` for local model hosting
- `db` for MySQL
- `agent` for the FastAPI application

If the model is not downloaded yet, wait for the Ollama container to finish pulling it before sending queries.

### 5. Query the agent

```bash
curl -G --data-urlencode "q=Your query" http://localhost:8000
```

### 6. Check service health

```bash
curl http://localhost:8000/health
```

## AWS Deployment

The AWS path is intentionally simple and cost-focused:

- one EC2 instance
- Docker Compose running `ollama`, `db`, and `agent`
- Terraform-managed VPC, subnet, security group, IAM role, S3 bootstrap bucket, and SSM parameter

This keeps the app easy to understand, but it is still not a production-grade architecture.

### Important cost note

Running Ollama in AWS is much heavier than calling a managed API. The default cloud deployment therefore uses a much smaller model than local mode:

- Local default: `gpt-oss:20b`
- AWS default: `qwen2.5:0.5b-instruct-q5_0`

Even with that change, AWS free-tier compatibility depends on your AWS account type, your region, your storage usage, and how long the instance runs. Treat the provided Terraform defaults as "lowest practical cost", not "guaranteed free".

### 1. Prepare Terraform variables

Move into the Terraform directory:

```bash
cd terraform/aws
```

Create a local variables file from the example:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Update at least these values:

- `project_ref`
- `db_dump_path`
- `db_root_password`
- `app_ingress_cidr_blocks`

For a quick infrastructure smoke test, you can point `db_dump_path` at `../../db/db.sql.example`.

### 2. Review the defaults

The Terraform stack will:

- upload your local SQL dump to a private S3 bucket
- store the MySQL password in SSM Parameter Store
- provision an EC2 instance
- install Docker and Docker Compose on the instance
- clone this repository on the instance
- start the AWS-specific Compose stack from `compose.aws.yaml`

### 3. Deploy

```bash
terraform init
terraform plan
terraform apply
```

After `apply`, Terraform outputs the public API URL and an AWS Systems Manager command you can use to open a shell on the instance.

### 4. Destroy when you are done

```bash
terraform destroy
```

## Configuration

### LLM configuration

- `OLLAMA_MODEL`: model name to use
- `OLLAMA_BASE_URL`: full Ollama base URL such as `http://ollama:11434`

### Database configuration

- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_DATABASE`
- `MYSQL_USER`
- `MYSQL_ROOT_PASSWORD`
- `MYSQL_ROOT_PASSWORD_FILE`

The application prefers `MYSQL_ROOT_PASSWORD` if it is set, and otherwise reads from `MYSQL_ROOT_PASSWORD_FILE`.

## Project Structure

```text
.
├── compose.yaml
├── compose.aws.yaml
├── db/
├── ollama/
├── src/sqlagent/
├── terraform/aws/
├── tests/
├── Dockerfile
├── README.md
├── pyproject.toml
└── uv.lock
```

## License

This project is distributed under the [MIT License](./LICENSE).
