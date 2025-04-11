
# Configuration Guide - Face Recognition System

## Environment Variables

### Core Settings
```env
DATABASE_URL=postgresql://user:password@localhost:5432/face_db
REDIS_URL=redis://localhost:6379/0
API_KEYS=key1,key2,key3
```

### Feature Flags
```env
ENABLE_EMOTION_DETECTION=false
ENABLE_AGE_GENDER_ESTIMATION=false
ENABLE_RE_IDENTIFICATION=true
ENABLE_ACTIVE_LEARNING=false
```

### Performance Settings
```env
ENABLE_WASM=false
SHARDING_ENABLED=false
SHARDING_STRATEGY=geographic
NUM_SHARDS=1
LOAD_BALANCING_STRATEGY=round_robin
```

## Configuration File (config.json)

```json
{
  "recognition": {
    "model": "insightface",
    "threshold": 0.6,
    "max_results": 5
  },
  "database": {
    "vector_db": "faiss",
    "sharding_strategy": "uniform"
  },
  "cameras": {
    "frame_rate": 15,
    "recognition_interval": 1.0
  }
}
```

## Scaling Configuration

### Load Balancing
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

## Monitoring Setup
```env
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
```
