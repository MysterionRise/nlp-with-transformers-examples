# Deployment Guide - NLP Transformers Examples

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Download NER model
python -m spacy download en_core_web_sm

# Run launcher
python launch_ui.py

# Or run specific UI
python ui/qa_system.py
```

### Docker Deployment

#### Build Image

```bash
# Build Docker image
docker build -t nlp-transformers-examples:latest .

# Or with specific tag
docker build -t nlp-transformers-examples:v1.0 .
```

#### Run Container

```bash
# Run single container with all UIs
docker run -d \
  --name nlp-app \
  -p 7860-7869:7860-7869 \
  -v huggingface_cache:/home/nlpuser/.cache/huggingface \
  nlp-transformers-examples:latest

# With environment variables
docker run -d \
  --name nlp-app \
  -p 7860-7869:7860-7869 \
  -e NLP_DEBUG=true \
  -e NLP_LOG_LEVEL=INFO \
  -v huggingface_cache:/home/nlpuser/.cache/huggingface \
  nlp-transformers-examples:latest

# Run specific UI
docker run -d \
  --name nlp-qa \
  -p 7865:7865 \
  nlp-transformers-examples:latest \
  python ui/qa_system.py
```

### Docker Compose Deployment

#### Single Command Deployment

```bash
# Launch all UIs
docker-compose up -d

# View logs
docker-compose logs -f ui-launcher

# Stop services
docker-compose down

# View individual UI logs
docker-compose logs ui-launcher
docker-compose logs sentiment
```

#### Launch Individual Services

```bash
# Launch only UI launcher
docker-compose up -d ui-launcher

# Launch individual UIs
docker-compose --profile individual up -d sentiment
docker-compose --profile individual up -d qa
docker-compose --profile individual up -d generation
```

#### Access UIs

After deployment, access UIs at:
- Sentiment: http://localhost:7860
- Similarity: http://localhost:7861
- NER: http://localhost:7862
- Summarization: http://localhost:7863
- Performance: http://localhost:7864
- **QA: http://localhost:7865** (NEW)
- **Generation: http://localhost:7866** (NEW)
- **Zero-Shot: http://localhost:7867** (NEW)
- **Translation: http://localhost:7868** (NEW)
- **Vision: http://localhost:7869** (NEW)

## Kubernetes Deployment

### Create Deployment YAML

```yaml
# nlp-transformers-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlp-transformers
  labels:
    app: nlp-transformers
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nlp-transformers
  template:
    metadata:
      labels:
        app: nlp-transformers
    spec:
      containers:
      - name: nlp-transformers
        image: nlp-transformers-examples:latest
        ports:
        - containerPort: 7860
        - containerPort: 7865
        - containerPort: 7866
        - containerPort: 7867
        - containerPort: 7868
        - containerPort: 7869
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: NLP_DEBUG
          value: "false"
        - name: PYTHONUNBUFFERED
          value: "1"
        volumeMounts:
        - name: cache-volume
          mountPath: /home/nlpuser/.cache/huggingface
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: huggingface-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nlp-transformers-service
spec:
  selector:
    app: nlp-transformers
  type: LoadBalancer
  ports:
  - port: 7860
    targetPort: 7860
    name: sentiment
  - port: 7865
    targetPort: 7865
    name: qa
  - port: 7866
    targetPort: 7866
    name: generation
  - port: 7867
    targetPort: 7867
    name: zero-shot
  - port: 7868
    targetPort: 7868
    name: translation
  - port: 7869
    targetPort: 7869
    name: vision
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: huggingface-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
```

### Deploy to Kubernetes

```bash
# Create ConfigMap for sample data
kubectl create configmap nlp-data --from-file=data/

# Apply deployment
kubectl apply -f nlp-transformers-deployment.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# Access service
kubectl port-forward service/nlp-transformers-service 7860:7860
kubectl port-forward service/nlp-transformers-service 7865:7865
```

## Cloud Deployment

### AWS ECS

```bash
# Create ECR repository
aws ecr create-repository --repository-name nlp-transformers

# Tag image
docker tag nlp-transformers-examples:latest \
  <aws-account-id>.dkr.ecr.<region>.amazonaws.com/nlp-transformers:latest

# Push to ECR
docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/nlp-transformers:latest

# Create ECS task definition (see AWS documentation)
# Deploy using ECS console or CLI
```

### Google Cloud Run

```bash
# Configure Docker for GCP
gcloud auth configure-docker gcr.io

# Tag image
docker tag nlp-transformers-examples:latest \
  gcr.io/<project-id>/nlp-transformers:latest

# Push to GCR
docker push gcr.io/<project-id>/nlp-transformers:latest

# Deploy to Cloud Run
gcloud run deploy nlp-transformers \
  --image gcr.io/<project-id>/nlp-transformers:latest \
  --platform managed \
  --region us-central1 \
  --port 7860 \
  --memory 4Gi
```

### Azure Container Instances

```bash
# Create Azure Container Registry
az acr create --resource-group <rg> --name nlptransformers --sku Basic

# Push image to ACR
az acr build --registry nlptransformers --image nlp-transformers:latest .

# Deploy to ACI
az container create \
  --resource-group <rg> \
  --name nlp-transformers \
  --image nlptransformers.azurecr.io/nlp-transformers:latest \
  --ports 7860 7865 7866 7867 7868 7869 \
  --memory 4
```

## Performance Optimization

### GPU Support

```dockerfile
# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Then follow regular installation
```

```bash
# Run container with GPU
docker run --gpus all \
  -p 7860-7869:7860-7869 \
  nlp-transformers-examples:latest
```

### Memory Optimization

```bash
# Set memory limit
docker run -m 8g \
  -p 7860-7869:7860-7869 \
  nlp-transformers-examples:latest
```

### Model Caching

```bash
# Pre-warm cache by loading models
docker run -it \
  --name model-cache-warmer \
  -v huggingface_cache:/home/nlpuser/.cache/huggingface \
  nlp-transformers-examples:latest \
  python -c "
from ui.sentiment_playground import load_model
load_model('Twitter RoBERTa (Multilingual)')
"
```

## Monitoring

### Health Checks

```bash
# Check individual UI health
curl http://localhost:7860/api/predict

# Check container health
docker ps  # HEALTHCHECK status shown
```

### Logging

```bash
# View Docker logs
docker logs nlp-app -f

# View Docker Compose logs
docker-compose logs -f ui-launcher

# Kubernetes logs
kubectl logs deployment/nlp-transformers -f
```

### Metrics

```bash
# Docker stats
docker stats nlp-app

# Kubernetes metrics
kubectl top nodes
kubectl top pods
```

## Troubleshooting

### Model Loading Errors

```bash
# Check available memory
docker run --rm nlp-transformers-examples:latest df -h

# Reduce model size
docker run -e NLP_MAX_CACHED_MODELS=2 \
  nlp-transformers-examples:latest
```

### Port Conflicts

```bash
# Map to different host ports
docker run -p 8860:7860 -p 8865:7865 \
  nlp-transformers-examples:latest

# Check port usage
lsof -i :7860
sudo netstat -tlnp | grep 7860
```

### Out of Memory

```bash
# Increase container memory
docker run -m 16g \
  nlp-transformers-examples:latest

# Use smaller models
# Edit config/models.yaml to use DistilBERT, DistilGPT-2, etc.
```

## Best Practices

1. **Use Health Checks:** Enable health checks for production
2. **Persistent Storage:** Mount volume for model cache
3. **Resource Limits:** Set memory and CPU limits
4. **Non-root User:** Container uses nlpuser (non-root)
5. **Multi-stage Build:** Optimized image size using multi-stage Dockerfile
6. **Environment Variables:** Use env vars for configuration
7. **Log Rotation:** Configure log rotation for long-running containers
8. **Security:** Run as non-root, use read-only filesystems where possible
9. **Caching:** Pre-warm model cache before production deployment
10. **Load Balancing:** Use load balancer for horizontal scaling

## Scaling Considerations

### Horizontal Scaling

- Run multiple container instances
- Use load balancer (HAProxy, Nginx, AWS ALB)
- Share model cache using persistent volume
- Environment: Kubernetes, Docker Swarm, ECS

### Vertical Scaling

- Increase container memory (for larger models)
- Add GPU support (for faster inference)
- Optimize model selection (use DistilBERT instead of BERT)

## Security

- Run containers as non-root user
- Use read-only filesystems
- Implement rate limiting
- Use API authentication
- Enable HTTPS/TLS
- Keep dependencies updated
- Scan images for vulnerabilities

```bash
# Scan image for vulnerabilities
docker scan nlp-transformers-examples:latest

# Use Docker Content Trust
export DOCKER_CONTENT_TRUST=1
docker push nlp-transformers-examples:latest
```

## Updates

To deploy new versions:

```bash
# Rebuild image
docker build -t nlp-transformers-examples:v1.1 .

# Update Docker Compose
docker-compose down
docker-compose up -d

# Update Kubernetes
kubectl set image deployment/nlp-transformers \
  nlp-transformers=nlp-transformers-examples:v1.1
```

---

**For more information, see:**
- [IMPLEMENTATION_COMPLETE.md](./IMPLEMENTATION_COMPLETE.md) - Feature overview
- [CLAUDE.md](./CLAUDE.md) - Architecture and configuration
- [README.md](./README.md) - Project overview
