# Advanced Face Recognition System

A scalable, distributed face recognition system with real-time updates, advanced recognition features, and comprehensive monitoring capabilities.

## System Architecture

The system is built on a distributed architecture with the following key components:

### Core Components

1. **gRPC Service Layer** (`face_recognition_service.py`)
   - Handles core face processing operations
   - Provides high-performance binary communication
   - Supports all face recognition operations (registration, recognition, etc.)

2. **Database Layer** (`database.py`, `models.py`, `database_setup.py`)
   - Uses PostgreSQL for metadata and logs
   - Utilizes FAISS vector database for efficient face embedding storage and similarity search
   - Provides optimized face search with configurable thresholds

3. **Real-time Updates** (`realtime/` directory)
   - WebSocket integration with Socket.IO
   - Event-driven architecture for instant updates
   - Typed events for face registration, recognition, and deletion

4. **Camera Management** (`camera.py`)
   - Multi-camera support (webcams, IP cameras)
   - Frame processing with recognition overlays
   - Thread-safe implementation

5. **Scaling Infrastructure** (`scaling/` directory)
   - Load balancing with multiple algorithms
   - Database sharding strategies
   - Horizontal scaling support

### Advanced Features

1. **Face Quality Analysis** (`face_quality.py`)
   - Measures image quality through multiple metrics:
     - Sharpness
     - Brightness
     - Contrast
     - Face size
     - Pose deviation
     - Eye openness

2. **Re-identification** (`advanced_features/re_identification.py`)
   - Cross-camera person tracking
   - Temporal consistency in identification

3. **Emotion Detection** (`advanced_features/emotion_detector.py`)
   - Facial expression analysis
   - Emotion classification

4. **Active Learning** (`advanced_features/active_learning.py`)
   - Self-improving recognition models
   - Feedback loop for accuracy enhancement

5. **Monitoring System** (`monitoring/metrics.py`)
   - Performance metrics collection
   - Prometheus integration
   - System health tracking

## API Documentation

### gRPC API

The system exposes a powerful gRPC API for efficient binary communication:

| Operation | Description | Parameters | Returns |
|-----------|-------------|------------|---------|
| RegisterFace | Add a new face to the system | Image data, name, metadata | Face ID, quality score |
| RecognizeFaces | Identify faces in an image | Image data, threshold, max results | Matches with confidence scores |
| ListFaces | Retrieve registered faces | Pagination parameters | List of faces with metadata |
| DeleteFace | Remove a face from the system | Face ID | Success status |
| UpdateFaceMetadata | Modify face metadata | Face ID, metadata | Success status |
| GetFaceQuality | Analyze face image quality | Image data | Quality metrics |
| VerifyLiveness | Detect spoofing attempts | Image data, session ID | Liveness score |

### HTTP REST API

For web applications and simpler integrations, a RESTful API is provided:

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/api/faces` | GET | Get all registered faces | offset, limit |
| `/api/faces/register` | POST | Register a new face | name, image |
| `/api/faces/delete/<face_id>` | DELETE | Delete a face | face_id |
| `/api/cameras` | GET | List all cameras | - |
| `/api/cameras` | POST | Add a new camera | name, url/device_id |
| `/api/cameras/<camera_id>` | DELETE | Remove a camera | camera_id |
| `/api/test/recognize` | GET | Simulate recognition event | - |
| `/api/events/dispatch` | POST | Dispatch custom event | event_type, data |

## WebSocket Events

Real-time updates are delivered through Socket.IO events:

| Event | Description | Data |
|-------|-------------|------|
| `face_added` | New face registered | face_id, name, quality_score, added_at |
| `face_deleted` | Face removed from system | face_id, name, deleted_at |
| `face_recognized` | Face identified in camera | face_id, name, camera_id, camera_name, confidence, timestamp |
| `camera_connected` | Camera comes online | camera_id, camera_name, timestamp |
| `camera_disconnected` | Camera goes offline | camera_id, camera_name, timestamp |

### Client Usage

```javascript
// Connect to Socket.IO server
const socket = io();

// Subscribe to events
socket.emit('subscribe', {
  events: ['face_added', 'face_deleted', 'face_recognized']
});

// Handle events
socket.on('face_recognized', (data) => {
  console.log(`Recognized ${data.name} with confidence ${data.confidence}`);
  // Update UI...
});
```

## Scaling Capabilities

### Load Balancing

The system supports multiple load balancing strategies:

- **Round Robin**: Distribute requests evenly
- **Least Connections**: Send to least busy node
- **Least Response Time**: Send to fastest responding node
- **IP Hash**: Consistent routing based on client IP
- **Weighted Round Robin**: Distribution by node capacity

### Database Sharding

Data can be distributed across multiple database nodes using:

- **Hash-based Sharding**: Distribute by key hash
- **Range-based Sharding**: Partition by ID ranges
- **Geographic Sharding**: Distribute by physical location
- **Camera-based Sharding**: Group by camera sources

## Monitoring

The system includes comprehensive monitoring with:

- **Prometheus Metrics**: Request rates, latencies, error rates
- **Health Checks**: Component status monitoring
- **Performance Tracking**: Resource utilization metrics
- **Alert Integration**: Configurable alerting thresholds

## Web Interface

The system includes a web dashboard for:

- Face database management
- Camera monitoring
- Real-time recognition events
- System configuration
- Performance metrics visualization

## Installation & Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure database:
   ```
   python database_setup.py
   ```

3. Start the system:
   ```
   python main.py
   ```

## Configuration

System behavior can be customized through `config.json`:

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

## Security Considerations

- All API endpoints require authentication
- WebSocket connections use secure tokens
- Face data is encrypted at rest
- Strict access controls for administrative functions

## Performance Optimization

- Batched processing for multiple faces
- GPU acceleration where available
- Connection pooling for database operations
- Image caching to reduce redundant processing