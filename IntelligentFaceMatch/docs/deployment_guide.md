
# Deployment Guide - Face Recognition System

## Deployment on Replit

1. Configure deployment settings in .replit:
   ```
   [deployment]
   deploymentTarget = "autoscale"
   run = ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"]
   ```

2. Set up environment variables in Replit Secrets:
   - DATABASE_URL
   - REDIS_URL
   - API_KEYS
   - MONITORING_TOKEN

3. Deploy using Replit's deployment feature

## Scaling Configuration

### Load Balancing
Configure in config.json:
```json
{
  "scaling": {
    "load_balancing": {
      "strategy": "round_robin",
      "health_check_interval": 30
    }
  }
}
```

### Database Sharding
Set up sharding in config.json:
```json
{
  "database": {
    "sharding": {
      "enabled": true,
      "strategy": "uniform",
      "shard_count": 3
    }
  }
}
```

## Monitoring
- Prometheus metrics at /metrics
- System health dashboard
- Performance alerts
