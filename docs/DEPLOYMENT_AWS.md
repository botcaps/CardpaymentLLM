# AWS Deployment Guide — Card Scheme Orchestrator (CSO)

This guide covers two deployment paths on AWS and a complete reference for
swapping the LLM provider without touching any code.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Deployment Option A — EC2 (simplest)](#2-deployment-option-a--ec2-simplest)
3. [Deployment Option B — ECS + ECR (Docker, production-grade)](#3-deployment-option-b--ecs--ecr-docker-production-grade)
4. [Deployment Option C — AWS App Runner (easiest Docker)](#4-deployment-option-c--aws-app-runner-easiest-docker)
5. [Storing Secrets Safely in AWS](#5-storing-secrets-safely-in-aws)
6. [Replacing the LLM API — Complete Reference](#6-replacing-the-llm-api--complete-reference)
7. [Verifying the Deployment](#7-verifying-the-deployment)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Tool | Install / Check |
|------|----------------|
| AWS CLI v2 | `aws --version` → install from aws.amazon.com/cli |
| Docker Desktop | `docker --version` |
| Python 3.10–3.12 | `python --version` |
| AWS account | Console: console.aws.amazon.com |

Configure the AWS CLI once:

```bash
aws configure
# Enter: Access Key ID, Secret Access Key, region (e.g. us-east-1), output: json
```

---

## 2. Deployment Option A — EC2 (simplest)

Best for: quick demos, development, or when you don't need Docker.

### Step 1 — Launch an EC2 instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Choose **Ubuntu 22.04 LTS** (free tier: t2.micro, production: t3.medium or larger)
3. In **Security Group**, add an **Inbound Rule**:
   - Type: Custom TCP | Port: **8501** | Source: 0.0.0.0/0
4. Create or select a key pair (.pem file) — download and keep it safe
5. Launch the instance

### Step 2 — Connect to the instance

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### Step 3 — Install dependencies on the instance

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
git clone <your-repo-url> cso_final
cd cso_final
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4 — Set environment variables

```bash
cp .env.example .env
nano .env
# Fill in your LLM_PROVIDER and API key (see Section 6 for all options)
```

### Step 5 — Run the app

```bash
# Keep it alive after you log out:
nohup streamlit run dashboard.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  > streamlit.log 2>&1 &

echo "App running at http://<EC2_PUBLIC_IP>:8501"
```

To stop it later: `pkill -f streamlit`

---

## 3. Deployment Option B — ECS + ECR (Docker, production-grade)

Best for: stable production deployments with auto-scaling and health checks.

### Step 1 — Build and push the Docker image to ECR

```bash
# Set your AWS account ID and region
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1
REPO_NAME=cso-pipeline

# Create the ECR repository (one-time)
aws ecr create-repository --repository-name $REPO_NAME --region $AWS_REGION

# Log Docker into ECR
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS \
    --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build the image (from the project root where Dockerfile lives)
cd /path/to/cso_final
docker build -t $REPO_NAME .

# Tag and push
docker tag $REPO_NAME:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest

docker push \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest
```

### Step 2 — Store secrets in AWS Secrets Manager

Never put API keys into the task definition. Store them here instead:

```bash
# Example for Anthropic. Repeat for whichever provider you use.
aws secretsmanager create-secret \
  --name cso/anthropic-api-key \
  --secret-string "sk-ant-YOUR_KEY_HERE"

aws secretsmanager create-secret \
  --name cso/langsmith-api-key \
  --secret-string "YOUR_LANGSMITH_KEY"
```

### Step 3 — Create an ECS Cluster

```bash
aws ecs create-cluster --cluster-name cso-cluster
```

Or via Console: **ECS → Clusters → Create Cluster → Networking only (Fargate)**

### Step 4 — Create the ECS Task Definition

Create a file `task-definition.json`:

```json
{
  "family": "cso-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "cso-app",
      "image": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/cso-pipeline:latest",
      "portMappings": [
        { "containerPort": 8501, "protocol": "tcp" }
      ],
      "environment": [
        { "name": "LLM_PROVIDER",           "value": "anthropic" },
        { "name": "LANGCHAIN_TRACING_V2",   "value": "true" },
        { "name": "LANGCHAIN_PROJECT",      "value": "cso-capstone" }
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:cso/anthropic-api-key"
        },
        {
          "name": "LANGCHAIN_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:cso/langsmith-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cso-pipeline",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3
      }
    }
  ]
}
```

Register the task definition:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### Step 5 — Create a CloudWatch log group

```bash
aws logs create-log-group --log-group-name /ecs/cso-pipeline
```

### Step 6 — Run the ECS service

```bash
# Get your default VPC and subnet IDs
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)

SUBNET_IDS=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[*].SubnetId" --output text | tr '\t' ',')

# Create a security group allowing port 8501
SG_ID=$(aws ec2 create-security-group \
  --group-name cso-sg \
  --description "CSO Pipeline SG" \
  --vpc-id $VPC_ID \
  --query GroupId --output text)

aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 8501 --cidr 0.0.0.0/0

# Create the service
aws ecs create-service \
  --cluster cso-cluster \
  --service-name cso-service \
  --task-definition cso-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration \
    "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}"
```

### Step 7 — Find the public IP

```bash
# Get the task ARN
TASK_ARN=$(aws ecs list-tasks --cluster cso-cluster \
  --service-name cso-service --query "taskArns[0]" --output text)

# Get the ENI
ENI=$(aws ecs describe-tasks --cluster cso-cluster --tasks $TASK_ARN \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
  --output text)

# Get the public IP
aws ec2 describe-network-interfaces \
  --network-interface-ids $ENI \
  --query "NetworkInterfaces[0].Association.PublicIp" --output text
```

App is live at `http://<PUBLIC_IP>:8501`

---

## 4. Deployment Option C — AWS App Runner (easiest Docker)

Best for: zero-ops Docker deployment without managing clusters.

### Step 1 — Push image to ECR (same as Option B, Steps 1)

### Step 2 — Create App Runner service via Console

1. Go to **AWS Console → App Runner → Create Service**
2. **Source**: Container registry → Amazon ECR → select your `cso-pipeline` image
3. **Port**: 8501
4. **Environment variables** — add each one:
   - `LLM_PROVIDER` = `anthropic` (or your chosen provider)
   - `ANTHROPIC_API_KEY` = your key (or use Secrets Manager ARN)
   - `LANGCHAIN_TRACING_V2` = `true`
   - `LANGCHAIN_API_KEY` = your LangSmith key
   - `LANGCHAIN_PROJECT` = `cso-capstone`
5. **Health check path**: `/_stcore/health`
6. Click **Create & deploy**

App Runner gives you an HTTPS URL automatically (no IP needed).

---

## 5. Storing Secrets Safely in AWS

Never commit `.env` or put raw keys in task definitions.

### Option A — AWS Secrets Manager (recommended)

```bash
# Store a secret
aws secretsmanager create-secret \
  --name cso/openai-api-key \
  --secret-string "sk-YOUR_KEY"

# Update an existing secret
aws secretsmanager update-secret \
  --secret-id cso/openai-api-key \
  --secret-string "sk-NEW_KEY"

# Retrieve for debugging
aws secretsmanager get-secret-value \
  --secret-id cso/openai-api-key --query SecretString --output text
```

Reference in ECS task definition `secrets` block (see Step 4 of Option B).

### Option B — AWS Systems Manager Parameter Store (cheaper)

```bash
aws ssm put-parameter \
  --name "/cso/anthropic_api_key" \
  --value "sk-ant-YOUR_KEY" \
  --type SecureString

# Reference in task definition:
# "valueFrom": "arn:aws:ssm:us-east-1:<ACCOUNT_ID>:parameter/cso/anthropic_api_key"
```

---

## 6. Replacing the LLM API — Complete Reference

The entire LLM abstraction lives in one file: **`llm_clients.py`**.
All agents and the orchestrator call `get_chat_model(tier)` or `gemini_client()`
which both read from environment variables — **no code changes needed to
switch providers**.

### The one env var that controls everything

```
LLM_PROVIDER=<provider>
```

| Value | Provider | Required env vars |
|-------|----------|-------------------|
| `anthropic` | Claude (Haiku / Sonnet) | `ANTHROPIC_API_KEY` |
| `openai` | GPT-4o / GPT-4o-mini | `OPENAI_API_KEY` |
| `azure_openai` | Azure-hosted GPT-4o | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |
| `google` | Gemini 2.5 Flash | `GEMINI_API_KEY` |
| `groq` | Llama 3.3 70B (fast/cheap) | `GROQ_API_KEY` |
| `ollama` | Local Llama (no key) | none (Ollama must be running) |
| `mock` | Deterministic stub | none (no API calls) |

### How to switch provider — example: Gemini → OpenAI

**In `.env` (local / EC2):**

```bash
# Before
LLM_PROVIDER=google
GEMINI_API_KEY=AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# After
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

**In ECS task definition `environment` + `secrets` blocks:**

```json
"environment": [
  { "name": "LLM_PROVIDER", "value": "openai" }
],
"secrets": [
  {
    "name": "OPENAI_API_KEY",
    "valueFrom": "arn:aws:secretsmanager:us-east-1:<ACCOUNT>:secret:cso/openai-api-key"
  }
]
```

Then redeploy: `aws ecs update-service --cluster cso-cluster --service cso-service --force-new-deployment`

### Optional: override individual model tiers

Each agent uses one of three tiers (`fast`, `smart`, `judge`). You can
override the model for any tier without changing provider:

```bash
OVERRIDE_FAST_MODEL=claude-haiku-4-5-20251001   # used by auth & fraud agents
OVERRIDE_SMART_MODEL=claude-sonnet-4-6           # used by cost agent & planner
OVERRIDE_JUDGE_MODEL=claude-haiku-4-5-20251001   # used by LLM-as-judge evaluator
```

For Azure OpenAI these must be **deployment names** (not OpenAI model IDs):

```bash
OVERRIDE_FAST_MODEL=my-gpt4o-mini-deployment
OVERRIDE_SMART_MODEL=my-gpt4o-deployment
```

### Where LLM calls happen in the codebase

| File | Line(s) | What it does | Tier used |
|------|---------|-------------|-----------|
| `orchestrator/graph.py` | 95–96 | `get_llm(tier)` — helper used by all graph nodes | various |
| `orchestrator/graph.py` | 351 | `plan_candidates` — LLM planner picks schemes | `smart` |
| `orchestrator/graph.py` | 407 | `reflect_on_ranking` — self-critique node | `smart` |
| `orchestrator/graph.py` | 563 | `generate_explanation` — plain-English output | `fast` |
| `orchestrator/graph.py` | 597 | `critique_auth_score` — Reflexion self-correction | `fast` |
| `orchestrator/graph.py` | 661, 756 | Fraud & compliance reasoning nodes | `fast` / `smart` |
| `agents/auth_score/agent.py` | 114 | `gemini_client()` — legacy direct Gemini path | fast (Haiku/Flash) |
| `agents/cost/agent.py` | 104 | `gemini_client()` — legacy direct Gemini path | smart (Sonnet/Pro) |
| `evaluation/evaluators/tool_use_accuracy.py` | 205 | LLM-as-judge evaluator | `fast` |

> **Note on `gemini_client()` in the two agent files:**
> `agents/auth_score/agent.py` and `agents/cost/agent.py` contain a legacy
> code path that calls `gemini_client()` directly (requires `GEMINI_API_KEY`).
> This path is only reached when `cfg.use_llm == True` and the agent falls
> through to the live branch. To use a different provider with these agents,
> either set `LLM_MODE=mock` to skip them (demo mode) or update the live
> branch to call `get_chat_model()` instead of `gemini_client()`.

### Switching to AWS Bedrock (future option)

AWS Bedrock hosts Claude, Llama, and other models natively. To use it:

1. Enable the model in **AWS Console → Bedrock → Model access**
2. Add `langchain-aws` to `requirements.txt`
3. In `llm_clients.py`, add a new provider entry and use
   `ChatBedrockConverse` from `langchain_aws`
4. Set `LLM_PROVIDER=bedrock` + AWS credentials (IAM role on EC2/ECS works
   without an API key)

---

## 7. Verifying the Deployment

### Health check

```bash
curl http://<YOUR_IP_OR_URL>:8501/_stcore/health
# Expected: {"status": "ok"}
```

### Smoke test (mock mode — no API key needed)

```bash
# On the server, with venv active:
LLM_MODE=mock python -c "
from orchestrator.orchestrate import orchestrate
from data.samples import get_samples
txn = get_samples()[0]
result = orchestrate(txn)
print('Decision:', result.get('decision'))
print('Smoke test PASSED')
"
```

### Run the test suite

```bash
LLM_MODE=mock pytest tests/ -v
```

### Check LangSmith traces

If `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` is set, every run
appears in your LangSmith project at smith.langchain.com under `cso-capstone`.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `RuntimeError: ANTHROPIC_API_KEY is not set` | Missing env var | Set `ANTHROPIC_API_KEY` in `.env` or Secrets Manager |
| `LLM_PROVIDER=... is not recognised` | Typo in provider name | Valid values: `openai`, `azure_openai`, `anthropic`, `google`, `groq`, `ollama`, `mock` |
| `AZURE_OPENAI_ENDPOINT is not set` | Azure path incomplete | Set `AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/` |
| Port 8501 not reachable | Security group | Add inbound TCP 8501 rule to the EC2/ECS security group |
| Container exits immediately | Missing secret or bad image | Check CloudWatch logs: `aws logs tail /ecs/cso-pipeline --follow` |
| `gemini_client` error with non-Google provider | Legacy agent path | Set `LLM_MODE=mock` or update agents to use `get_chat_model()` |
| Streamlit shows blank page | App still starting | Wait 30s, reload; or check `streamlit.log` on EC2 |
| ECS task keeps stopping | Memory limit too low | Increase `memory` in task definition to 4096 (4 GB) |
