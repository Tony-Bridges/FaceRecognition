import os
import logging
import uvicorn
import grpc
import uuid
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request
import base64
import io
from datetime import datetime
import cv2
import numpy as np

# Import components
from database import Database
from camera import CameraManager
from stream import StreamManager
from api import ApiClient

# Models for API
class FaceData(BaseModel):
    face_id: str
    name: str
    quality_score: Optional[float] = None
    metadata: Optional[Dict] = None

class RecognitionResult(BaseModel):
    face_id: str
    name: str
    confidence: float
    rect: Optional[List[int]] = None

class CameraInfo(BaseModel):
    camera_id: str
    name: str
    url: Optional[str] = None
    device_id: Optional[int] = None
    enabled: bool = True
    config: Optional[Dict] = None

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Global components (will be set in the start_fastapi function)
db = None
camera_manager = None
stream_manager = None
api_client = None

# Dependency to get API client
def get_api_client():
    return api_client

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the dashboard page"""
    cameras = camera_manager.get_camera_info() if camera_manager else []
    faces, total = db.get_all_faces(limit=100) if db else ([], 0)
    face_data = [face.to_dict() for face in faces]
    
    return templates.TemplateResponse(
        "dashboard.html", 
        {
            "request": request, 
            "cameras": cameras,
            "faces": face_data,
            "total_faces": total
        }
    )

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Render the face registration page"""
    cameras = camera_manager.get_camera_info() if camera_manager else []
    return templates.TemplateResponse(
        "register.html", 
        {
            "request": request, 
            "cameras": cameras
        }
    )

# API endpoints
@app.get("/api/faces", response_model=Dict)
async def get_faces(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    api: ApiClient = Depends(get_api_client)
):
    """Get all registered faces"""
    try:
        response = await api.list_faces(offset, limit)
        return {
            "faces": [
                {
                    "face_id": face.face_id,
                    "name": face.name,
                    "quality_score": face.quality_score,
                    "metadata": {
                        "registered_by": face.metadata.registered_by,
                        "device_id": face.metadata.device_id,
                        "registration_date": face.metadata.registration_date,
                        "last_accessed": face.metadata.last_accessed
                    }
                }
                for face in response.faces
            ],
            "total": response.total
        }
    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

@app.post("/api/faces/register", response_model=Dict)
async def register_face(
    name: str = Form(...),
    image: UploadFile = File(...),
    verify_liveness: bool = Form(False),
    api: ApiClient = Depends(get_api_client)
):
    """Register a new face"""
    try:
        # Read image file
        contents = await image.read()
        
        # Register face
        response = await api.register_face(contents, name, verify_liveness)
        
        return {
            "success": response.success,
            "face_id": response.face_id,
            "message": response.message,
            "quality_score": response.quality_score
        }
    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

@app.post("/api/faces/recognize", response_model=Dict)
async def recognize_faces(
    image: UploadFile = File(...),
    threshold: float = Form(0.6),
    max_results: int = Form(5),
    api: ApiClient = Depends(get_api_client)
):
    """Recognize faces in an image"""
    try:
        # Read image file
        contents = await image.read()
        
        # Recognize faces
        response = await api.recognize_faces(contents, threshold, max_results)
        
        return {
            "matches": [
                {
                    "face_id": match.face.face_id,
                    "name": match.face.name,
                    "confidence": match.confidence,
                    "rect": [
                        match.face.rect.x,
                        match.face.rect.y,
                        match.face.rect.width,
                        match.face.rect.height
                    ] if match.face.rect else None
                }
                for match in response.matches
            ]
        }
    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

@app.delete("/api/faces/{face_id}", response_model=Dict)
async def delete_face(
    face_id: str,
    api: ApiClient = Depends(get_api_client)
):
    """Delete a face from the database"""
    try:
        response = await api.delete_face(face_id)
        return {
            "success": response.success,
            "message": response.message
        }
    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

@app.put("/api/faces/{face_id}/metadata", response_model=Dict)
async def update_face_metadata(
    face_id: str,
    metadata: Dict,
    api: ApiClient = Depends(get_api_client)
):
    """Update face metadata"""
    try:
        response = await api.update_face_metadata(face_id, metadata)
        return {
            "success": response.success,
            "message": response.message
        }
    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

@app.post("/api/faces/quality", response_model=Dict)
async def analyze_face_quality(
    image: UploadFile = File(...),
    api: ApiClient = Depends(get_api_client)
):
    """Analyze face quality"""
    try:
        # Read image file
        contents = await image.read()
        
        # Get quality score
        response = await api.get_face_quality(contents)
        
        return {
            "quality_score": response.quality_score,
            "quality_factors": response.quality_factors
        }
    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

@app.post("/api/faces/verify-liveness", response_model=Dict)
async def verify_liveness(
    image: UploadFile = File(...),
    session_id: str = Form(...),
    api: ApiClient = Depends(get_api_client)
):
    """Verify face liveness"""
    try:
        # Read image file
        contents = await image.read()
        
        # Verify liveness
        response = await api.verify_liveness(contents, session_id)
        
        return {
            "is_live": response.is_live,
            "confidence": response.confidence,
            "message": response.message
        }
    except grpc.RpcError as e:
        raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")

# Camera endpoints
@app.get("/api/cameras", response_model=List[Dict])
async def get_cameras():
    """Get all cameras"""
    if not camera_manager:
        return []
    
    return camera_manager.get_camera_info()

@app.post("/api/cameras", response_model=Dict)
async def add_camera(camera: CameraInfo):
    """Add a new camera"""
    if not camera_manager:
        raise HTTPException(status_code=500, detail="Camera manager not initialized")
    
    try:
        source = camera.url if camera.url else (int(camera.device_id) if camera.device_id is not None else 0)
        camera_id = camera_manager.add_camera(
            camera_id=camera.camera_id,
            source=source,
            name=camera.name,
            config=camera.config,
            start=camera.enabled
        )
        
        # Save to database
        if db:
            db.add_or_update_camera(
                camera_id=camera_id,
                name=camera.name,
                url=camera.url,
                device_id=camera.device_id,
                enabled=camera.enabled,
                config=camera.config
            )
        
        return {"camera_id": camera_id, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/cameras/{camera_id}", response_model=Dict)
async def remove_camera(camera_id: str):
    """Remove a camera"""
    if not camera_manager:
        raise HTTPException(status_code=500, detail="Camera manager not initialized")
    
    success = camera_manager.remove_camera(camera_id)
    
    # Remove from database
    if success and db:
        db.delete_camera(camera_id)
    
    return {"success": success}

@app.post("/api/cameras/{camera_id}/start", response_model=Dict)
async def start_camera(camera_id: str):
    """Start a camera"""
    if not camera_manager:
        raise HTTPException(status_code=500, detail="Camera manager not initialized")
    
    success = camera_manager.start_camera(camera_id)
    return {"success": success}

@app.post("/api/cameras/{camera_id}/stop", response_model=Dict)
async def stop_camera(camera_id: str):
    """Stop a camera"""
    if not camera_manager:
        raise HTTPException(status_code=500, detail="Camera manager not initialized")
    
    success = camera_manager.stop_camera(camera_id)
    return {"success": success}

# Streaming endpoints
@app.get("/video/{camera_id}")
async def video_feed(camera_id: str):
    """Stream video from a camera"""
    if not camera_manager or not stream_manager:
        raise HTTPException(status_code=500, detail="Camera or Stream manager not initialized")
    
    camera = camera_manager.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    # Generate a unique client ID
    client_id = str(uuid.uuid4())
    
    # Subscribe client to camera stream
    stream_manager.subscribe_client(client_id, camera_id)
    
    # Create MJPEG streaming response
    return StreamingResponse(
        content=stream_mjpeg(client_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

async def stream_mjpeg(client_id: str):
    """Stream MJPEG frames to client"""
    try:
        while True:
            # Get frame from stream manager
            frame_data = stream_manager.get_client_frame(client_id)
            
            if frame_data:
                yield frame_data
            else:
                # No frame available, wait a bit
                await asyncio.sleep(0.05)
    except Exception as e:
        logging.error(f"Error streaming to client {client_id}: {e}")
    finally:
        # Unsubscribe client when connection ends
        if stream_manager:
            stream_manager.unsubscribe_client(client_id)

def start_fastapi(database, cam_manager, strm_manager, host="0.0.0.0", port=5000):
    """Start the FastAPI application"""
    global db, camera_manager, stream_manager, api_client
    
    # Set global components
    db = database
    camera_manager = cam_manager
    stream_manager = strm_manager
    
    # Create API client
    grpc_host = os.environ.get("GRPC_HOST", "localhost")
    grpc_port = int(os.environ.get("GRPC_PORT", 8000))
    api_client = ApiClient(f"{grpc_host}:{grpc_port}")
    
    # Start the server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # This is just for development/testing
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=5000)
