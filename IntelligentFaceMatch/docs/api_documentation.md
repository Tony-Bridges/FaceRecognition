
# API Documentation - Face Recognition System

## REST API Endpoints

### Face Management
- `GET /api/faces`
  - Get all registered faces
  - Query params: offset, limit
  - Returns: List of faces with metadata

- `POST /api/faces/register`
  - Register a new face
  - Body: multipart/form-data with image and name
  - Returns: Face ID and quality score

- `DELETE /api/faces/{face_id}`
  - Delete a registered face
  - Returns: Success status

### Camera Management
- `GET /api/cameras`
  - List all configured cameras
  - Returns: List of camera info

- `POST /api/cameras`
  - Add a new camera
  - Body: Camera configuration (URL/device ID)
  - Returns: Camera ID

### Real-time Events
WebSocket events via Socket.IO:
```javascript
// Connect to Socket.IO
const socket = io();

// Subscribe to events
socket.emit('subscribe', {
  events: ['face_added', 'face_deleted', 'face_recognized']
});

// Handle events
socket.on('face_recognized', (data) => {
  console.log(`Recognized ${data.name}`);
});
```

## gRPC Service
Proto definition:
```protobuf
service FaceRecognition {
  rpc RegisterFace(RegisterRequest) returns (RegisterResponse);
  rpc RecognizeFaces(RecognizeRequest) returns (RecognizeResponse);
  rpc ListFaces(ListRequest) returns (ListResponse);
  rpc DeleteFace(DeleteRequest) returns (DeleteResponse);
}
```
