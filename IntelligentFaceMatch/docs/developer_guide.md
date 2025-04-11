
# Developer Guide - Face Recognition System

## Architecture Overview
- gRPC Service Layer (`face_recognition_service.py`)
- Database Layer (`database.py`, `models.py`)
- Real-time Updates (WebSocket/Socket.IO)
- Camera Management System
- Scaling Infrastructure

## Core Components

### Face Recognition Service
```python
# face_recognition_service.py
# Handles face detection, encoding, and recognition
```

### Database Integration
- PostgreSQL for metadata
- FAISS vector database for face embeddings
- Connection pooling and sharding support

### WebSocket Events
- face_added
- face_deleted
- face_recognized
- camera_connected
- camera_disconnected

## Development Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/face_db
   REDIS_URL=redis://localhost:6379/0
   ```

3. Start development server:
   ```bash
   python main.py
   ```

## API Documentation

### gRPC API
- RegisterFace
- RecognizeFaces
- ListFaces
- DeleteFace
- UpdateFaceMetadata

### REST API
- GET /api/faces
- POST /api/faces/register
- DELETE /api/faces/{face_id}
- GET /api/cameras
- POST /api/cameras

## Best Practices
1. Use type hints
2. Follow PEP 8
3. Write unit tests
4. Document API changes
5. Monitor performance metrics
