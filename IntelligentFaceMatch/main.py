import os
import logging
import uuid
import random
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, jsonify, abort
import database_setup
from database_setup import db
from models import FaceRecord, RecognitionLog, CameraConfig

# Import WebSocket components
from realtime.socketio_server import SocketIOServer
from realtime.events import EventDispatcher, EventTypes, get_event_dispatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Setup Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get("SESSION_SECRET", "face_recognition_secret_key")

# Initialize database
database_setup.init_app(app)

# Initialize event dispatcher and WebSocket server
event_dispatcher = get_event_dispatcher()
socketio = SocketIOServer(app=app, event_dispatcher=event_dispatcher)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', 
                          error_code=404, 
                          error_message="Page Not Found",
                          error_description="The page you're looking for doesn't exist."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', 
                          error_code=500, 
                          error_message="Server Error",
                          error_description="Something went wrong on our end. Please try again later."), 500

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    # Get faces from database
    faces = []
    try:
        face_records = FaceRecord.query.all()
        faces = [face.to_dict() for face in face_records]
    except Exception as e:
        logging.error(f"Error getting faces: {e}")
    
    # Get cameras from database
    cameras = []
    try:
        camera_records = CameraConfig.query.all()
        cameras = [camera.to_dict() for camera in camera_records]
        
        # Add running status (always false for mock implementation)
        for camera in cameras:
            camera['running'] = False
            camera['frame_count'] = 0
    except Exception as e:
        logging.error(f"Error getting cameras: {e}")
    
    return render_template('dashboard.html', 
                          cameras=cameras, 
                          faces=faces, 
                          total_faces=len(faces))

@app.route('/register')
def register():
    """Face registration page"""
    # Get cameras from database
    cameras = []
    try:
        camera_records = CameraConfig.query.all()
        cameras = [camera.to_dict() for camera in camera_records]
        
        # Add running status
        for camera in cameras:
            camera['running'] = False
    except Exception as e:
        logging.error(f"Error getting cameras: {e}")
        
    return render_template('register.html', cameras=cameras)

# API Endpoints
@app.route('/api/faces', methods=['GET'])
def get_faces():
    """Get all registered faces"""
    try:
        # Get pagination parameters
        offset = request.args.get('offset', 0, type=int)
        limit = request.args.get('limit', 100, type=int)
        
        # Query the database
        faces_query = FaceRecord.query.order_by(FaceRecord.created_at.desc())
        total = faces_query.count()
        faces = faces_query.offset(offset).limit(limit).all()
        
        # Convert to dictionaries
        face_list = [face.to_dict() for face in faces]
        
        return jsonify({
            "faces": face_list,
            "total": total
        })
    except Exception as e:
        logging.error(f"Error getting faces: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/faces/register', methods=['POST'])
def register_face():
    """Register a new face"""
    try:
        # For now, just log the request 
        name = request.form.get('name')
        if not name:
            return jsonify({"success": False, "message": "Name is required"}), 400
            
        # Create a mock face record
        face_id = f"face_{uuid.uuid4().hex[:8]}"
        quality_score = 0.85  # Mock quality score
        
        # Create meta data
        meta_data = {
            "registered_by": "web",
            "registration_date": datetime.now().isoformat(),
        }
        
        # Use helper function to register face and send event
        _register_face_with_event(name, face_id, quality_score, meta_data)
        
        return jsonify({
            "success": True,
            "face_id": face_id,
            "message": "Face registered successfully",
            "quality_score": quality_score
        })
    except Exception as e:
        logging.error(f"Error registering face: {e}")
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get all cameras"""
    try:
        cameras = CameraConfig.query.all()
        camera_list = [camera.to_dict() for camera in cameras]
        
        # Add running status (always false for now)
        for camera in camera_list:
            camera['running'] = False
            camera['frame_count'] = 0
        
        return jsonify(camera_list)
    except Exception as e:
        logging.error(f"Error getting cameras: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cameras', methods=['POST'])
def add_camera():
    """Add a new camera"""
    try:
        data = request.json
        if not data or not data.get('name'):
            return jsonify({"success": False, "message": "Camera name is required"}), 400
        
        # Generate camera ID if not provided
        camera_id = data.get('camera_id', f"cam_{uuid.uuid4().hex[:8]}")
        
        # Create camera config
        camera = CameraConfig(
            camera_id=camera_id,
            name=data.get('name'),
            url=data.get('url'),
            device_id=data.get('device_id'),
            enabled=1 if data.get('enabled', True) else 0,
            config=data.get('config', {})
        )
        
        db.session.add(camera)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "camera_id": camera_id
        })
    except Exception as e:
        logging.error(f"Error adding camera: {e}")
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/cameras/<camera_id>', methods=['DELETE'])
def remove_camera(camera_id):
    """Remove a camera"""
    try:
        camera = CameraConfig.query.filter_by(camera_id=camera_id).first()
        if not camera:
            return jsonify({"success": False, "message": "Camera not found"}), 404
        
        db.session.delete(camera)
        db.session.commit()
        
        return jsonify({"success": True})
    except Exception as e:
        logging.error(f"Error removing camera: {e}")
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

# Placeholder route for video feed (returns a static image for now)
@app.route('/video/<camera_id>')
def video_feed(camera_id):
    """Stream video from camera (placeholder)"""
    # In a full implementation, this would stream MJPEG frames
    # For now, we'll redirect to a static image
    return redirect(url_for('static', filename='images/placeholder-camera.svg'))

# API route for dispatching events (for testing)
@app.route('/api/events/dispatch', methods=['POST'])
def dispatch_event():
    """Dispatch an event (for testing)"""
    try:
        data = request.json
        if not data or not data.get('event_type'):
            return jsonify({"success": False, "message": "Event type is required"}), 400
            
        event_type = data.get('event_type')
        event_data = data.get('data', {})
        
        # Dispatch the event
        event_dispatcher.dispatch(event_type, event_data)
        
        return jsonify({
            "success": True,
            "message": f"Event {event_type} dispatched"
        })
    except Exception as e:
        logging.error(f"Error dispatching event: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# Test endpoint to simulate a face recognition event
@app.route('/api/test/recognize', methods=['GET'])
def test_recognition():
    """Generate a test face recognition event"""
    try:
        # Get random face from database if possible
        face = None
        try:
            face_count = FaceRecord.query.count()
            if face_count > 0:
                # Get a random face
                random_offset = random.randint(0, face_count - 1)
                face = FaceRecord.query.offset(random_offset).first()
        except Exception as e:
            logging.error(f"Error getting random face: {e}")
        
        # If no face in DB, use mock data
        if not face:
            face_id = f"face_{uuid.uuid4().hex[:8]}"
            name = "Test Person"
            quality_score = 0.85
        else:
            face_id = face.face_id
            name = face.name
            quality_score = face.quality_score or 0.85
            
        # Get a random camera if possible
        camera_id = None
        camera_name = "Unknown Camera"
        try:
            camera_count = CameraConfig.query.count()
            if camera_count > 0:
                # Get a random camera
                random_offset = random.randint(0, camera_count - 1)
                camera = CameraConfig.query.offset(random_offset).first()
                camera_id = camera.camera_id
                camera_name = camera.name
        except Exception as e:
            logging.error(f"Error getting random camera: {e}")
            
        if not camera_id:
            camera_id = f"cam_{uuid.uuid4().hex[:8]}"
            
        # Create the event data
        timestamp = datetime.now().isoformat()
        confidence = round(random.uniform(0.65, 0.98), 2)
        
        event_data = {
            "face_id": face_id,
            "name": name,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "confidence": confidence,
            "quality_score": quality_score,
            "timestamp": timestamp
        }
        
        # Log a recognition event in the database
        try:
            log = RecognitionLog(
                log_id=f"log_{uuid.uuid4().hex[:8]}",
                face_id=face_id,
                camera_id=camera_id,
                confidence=confidence,
                details={
                    "timestamp": timestamp,
                    "camera_name": camera_name
                }
            )
            db.session.add(log)
            db.session.commit()
        except Exception as e:
            logging.error(f"Error logging recognition: {e}")
            db.session.rollback()
            
        # Dispatch the event
        event_dispatcher.dispatch(EventTypes.FACE_RECOGNIZED, event_data)
        
        return jsonify({
            "success": True,
            "message": f"Recognition event for {name} dispatched",
            "data": event_data
        })
    except Exception as e:
        logging.error(f"Error generating test recognition: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# Integrate with face registration to send real-time updates
@app.route('/api/faces/delete/<face_id>', methods=['DELETE'])
def delete_face(face_id):
    """Delete a face by ID"""
    try:
        face = FaceRecord.query.filter_by(face_id=face_id).first()
        if not face:
            return jsonify({"success": False, "message": "Face not found"}), 404
        
        # Store the face data before deletion for the event
        face_data = face.to_dict()
        
        # Delete the face
        db.session.delete(face)
        db.session.commit()
        
        # Dispatch event
        event_dispatcher.dispatch(EventTypes.FACE_DELETED, {
            "face_id": face_id,
            "name": face_data.get("name"),
            "deleted_at": datetime.now().isoformat()
        })
        
        return jsonify({"success": True})
    except Exception as e:
        logging.error(f"Error deleting face: {e}")
        db.session.rollback()
        return jsonify({"success": False, "message": str(e)}), 500

# Update the registration endpoint to send events
def _register_face_with_event(name, face_id, quality_score, meta_data=None):
    """Register a face and dispatch event"""
    # Create record in database
    face = FaceRecord(
        face_id=face_id,
        name=name,
        quality_score=quality_score,
        meta_data=meta_data or {}
    )
    
    db.session.add(face)
    db.session.commit()
    
    # Dispatch event
    event_dispatcher.dispatch(EventTypes.FACE_ADDED, {
        "face_id": face_id,
        "name": name,
        "quality_score": quality_score,
        "added_at": datetime.now().isoformat()
    })
    
    return face

# Main entry point for development/testing
if __name__ == '__main__':
    # Run with SocketIO instead of regular Flask server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
