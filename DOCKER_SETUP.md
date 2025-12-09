# Docker Setup Guide - RAG Finance System

This guide covers containerized deployment of the RAG Finance System using Docker and Docker Compose.

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- OpenAI API key

## 🚀 Quick Start

### 1. Set Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional
PINECONE_API_KEY=your-pinecone-api-key-here
VECTOR_STORE_MODE=chroma
MAX_CORRECTIONS=2
LOG_LEVEL=INFO
```

### 2. Start All Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f rag-api
```

### 3. Verify Deployment

```bash
# Check service health
docker-compose ps

# Test API health endpoint
curl http://localhost:8000/health

# Test query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the revenue in Q4 2024?"}'
```

## 🌐 Access Services

Once running, access the services at:

| Service | URL | Description |
|---------|-----|-------------|
| **RAG API** | http://localhost:8000 | FastAPI application |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Jaeger UI** | http://localhost:16686 | Distributed tracing dashboard |
| **Prometheus** | http://localhost:9090 | Metrics and monitoring |
| **Grafana** | http://localhost:3000 | Visualization dashboards |

### Grafana Default Credentials
- **Username:** admin
- **Password:** admin

## 🛠️ Development Setup

For local development with hot reloading:

```bash
# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Rebuild after dependency changes
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Development features:
- ✅ Hot reloading on code changes
- ✅ Source code mounted as volume
- ✅ Debug logging enabled
- ✅ Local data directory mounted

## 📦 Docker Architecture

### Multi-Stage Dockerfile

The `Dockerfile` uses a multi-stage build pattern for optimal image size:

**Stage 1: Builder**
- Installs build dependencies (gcc, g++)
- Creates Python virtual environment
- Installs all Python dependencies

**Stage 2: Runtime**
- Minimal base image (python:3.12-slim)
- Copies only virtual environment (no build tools)
- Runs as non-root user (uid 1000)
- Includes health check command

**Image Size:** ~800MB (vs ~1.5GB without multi-stage)

### Services Architecture

```
┌─────────────────────────────────────────────────────┐
│                    rag-network                       │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ rag-api  │───▶│  Jaeger  │    │Prometheus│     │
│  │  :8000   │    │  :16686  │◀───│  :9090   │     │
│  └──────────┘    └──────────┘    └──────────┘     │
│       │                                │            │
│       │                                ▼            │
│       │                          ┌──────────┐      │
│       └─────────────────────────▶│ Grafana  │      │
│                                   │  :3000   │      │
│                                   └──────────┘      │
└─────────────────────────────────────────────────────┘
```

## 🔧 Service Configuration

### RAG API Service

```yaml
Environment Variables:
  - OPENAI_API_KEY: Required for LLM calls
  - VECTOR_STORE_MODE: chroma (local) or pinecone (production)
  - MAX_CORRECTIONS: Self-correction retry limit (default: 2)
  - OTEL_EXPORTER_OTLP_ENDPOINT: Jaeger endpoint for traces

Volumes:
  - chroma-data: Persistent vector database storage

Health Check:
  - Endpoint: /health
  - Interval: 30s
  - Timeout: 10s
```

### Jaeger Service

```yaml
Image: jaegertracing/all-in-one:1.52
Ports:
  - 16686: Jaeger UI
  - 4318: OTLP HTTP receiver (for traces)
  - 4317: OTLP gRPC receiver
Features:
  - All-in-one deployment (collector, query, UI)
  - In-memory storage (for production, use Cassandra/Elasticsearch)
```

### Prometheus Service

```yaml
Image: prom/prometheus:v2.48.1
Configuration: config/prometheus/prometheus.yml
Volumes:
  - prometheus-data: Metrics storage (30-day retention)
Scrape Targets:
  - prometheus:9090 (self-monitoring)
  - jaeger:14269 (Jaeger metrics)
  - rag-api:8000/metrics (application metrics)
```

### Grafana Service

```yaml
Image: grafana/grafana:10.2.3
Features:
  - Anonymous access enabled (viewer role)
  - Pre-configured datasources (Prometheus, Jaeger)
  - Pre-loaded dashboards
Volumes:
  - grafana-data: Persistent dashboards and settings
```

## 📊 Persistent Data

Docker volumes are created for persistent storage:

```bash
# List volumes
docker volume ls | grep rag

# Inspect volume
docker volume inspect rag-chroma-data

# Backup volume
docker run --rm -v rag-chroma-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/chroma-backup.tar.gz /data

# Restore volume
docker run --rm -v rag-chroma-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/chroma-backup.tar.gz -C /
```

### Volume Locations

| Volume | Purpose | Size (Approx) |
|--------|---------|---------------|
| `rag-chroma-data` | Vector database | 100MB - 10GB |
| `rag-prometheus-data` | Metrics (30 days) | 1GB - 5GB |
| `rag-grafana-data` | Dashboards/settings | 10MB - 100MB |

## 🔍 Monitoring and Observability

### Viewing Traces (Jaeger)

1. Open http://localhost:16686
2. Select "rag-finance-api" service
3. Click "Find Traces"
4. View detailed trace spans for each query

### Querying Metrics (Prometheus)

1. Open http://localhost:9090
2. Example queries:
   ```promql
   # Query latency (95th percentile)
   histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
   
   # Request rate
   rate(http_requests_total[5m])
   
   # Error rate
   rate(http_requests_total{status="500"}[5m])
   ```

### Viewing Dashboards (Grafana)

1. Open http://localhost:3000
2. Navigate to "Dashboards"
3. Select "RAG Finance Overview"
4. View real-time metrics and traces

## 🧪 Testing the Deployment

### Health Checks

```bash
# API health
curl http://localhost:8000/health | jq

# Metrics endpoint
curl http://localhost:8000/metrics | jq
```

### Sample Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the company revenue in Q4 2024?",
    "include_sources": true
  }' | jq
```

### Load Testing

```bash
# Install hey (HTTP load generator)
# macOS: brew install hey
# Linux: go install github.com/rakyll/hey@latest

# Run 100 requests with 10 concurrent workers
hey -n 100 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d '{"query":"What was the revenue?"}' \
  http://localhost:8000/query
```

## 🛑 Stopping and Cleanup

### Stop Services

```bash
# Stop all services (keep volumes)
docker-compose down

# Stop and remove volumes (destructive!)
docker-compose down -v

# Stop and remove images
docker-compose down --rmi all
```

### Clean Logs

```bash
# View log sizes
docker-compose logs --tail=0 2>&1 | wc -l

# Truncate logs
docker-compose logs --tail=0
```

## 🚨 Troubleshooting

### Issue: API service fails to start

**Solution:**
```bash
# Check logs
docker-compose logs rag-api

# Common issues:
# 1. Missing OPENAI_API_KEY in .env
# 2. Port 8000 already in use
# 3. Insufficient memory (increase Docker memory limit)
```

### Issue: Cannot connect to Jaeger

**Solution:**
```bash
# Verify Jaeger is running
docker-compose ps jaeger

# Check Jaeger logs
docker-compose logs jaeger

# Verify network connectivity
docker-compose exec rag-api ping jaeger
```

### Issue: Prometheus not scraping metrics

**Solution:**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq

# Verify prometheus.yml configuration
docker-compose exec prometheus cat /etc/prometheus/prometheus.yml

# Reload Prometheus configuration
curl -X POST http://localhost:9090/-/reload
```

### Issue: Out of disk space

**Solution:**
```bash
# Check Docker disk usage
docker system df

# Clean up unused resources
docker system prune -a --volumes

# Remove old images
docker image prune -a
```

## 🔐 Production Deployment

For production deployments, consider:

### Security Hardening

1. **Change default passwords:**
   ```yaml
   # docker-compose.yml
   grafana:
     environment:
       - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
   ```

2. **Disable anonymous access:**
   ```yaml
   grafana:
     environment:
       - GF_AUTH_ANONYMOUS_ENABLED=false
   ```

3. **Use secrets management:**
   ```bash
   # Use Docker secrets instead of environment variables
   docker secret create openai_key ./openai_key.txt
   ```

### Scalability

1. **Use external vector store:**
   ```yaml
   rag-api:
     environment:
       - VECTOR_STORE_MODE=pinecone
       - PINECONE_API_KEY=${PINECONE_API_KEY}
   ```

2. **Scale API service:**
   ```bash
   docker-compose up -d --scale rag-api=3
   ```

3. **Add load balancer:**
   ```yaml
   nginx:
     image: nginx:alpine
     ports:
       - "80:80"
     volumes:
       - ./nginx.conf:/etc/nginx/nginx.conf
     depends_on:
       - rag-api
   ```

### Monitoring

1. **External Prometheus:**
   - Use managed Prometheus (e.g., Grafana Cloud)
   - Configure remote write

2. **Persistent tracing:**
   - Replace Jaeger all-in-one with Jaeger + Elasticsearch
   - Or use managed tracing (e.g., Datadog, New Relic)

3. **Alerting:**
   ```yaml
   # config/prometheus/rules/alerts.yml
   groups:
     - name: api_alerts
       rules:
         - alert: HighErrorRate
           expr: rate(http_requests_total{status="500"}[5m]) > 0.05
           for: 5m
           labels:
             severity: critical
   ```

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

## 🤝 Contributing

For issues or improvements to the Docker setup:
1. Check existing issues
2. Test changes locally
3. Update this documentation
4. Submit a pull request

---

**Built with ❤️ for production-grade RAG systems**

