
# Installation Guide - Face Recognition System

## System Requirements
- Python 3.11+
- PostgreSQL
- Redis
- GPU support (optional)

## Dependencies
All required packages are listed in pyproject.toml:
- FastAPI
- Flask-SocketIO
- gRPC
- OpenCV
- FAISS
- SQLAlchemy
- Prometheus Client

## Setup Steps

1. Clone the repository on Replit

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure database:
   ```bash
   python database_setup.py
   ```

4. Initialize FAISS index:
   ```bash
   python -c "from database import init_faiss; init_faiss()"
   ```

5. Start the system:
   ```bash
   python main.py
   ```

## Configuration
Edit config.json to customize:
- Recognition thresholds
- Camera settings
- Database connections
- Scaling options

## Verification
1. Access web interface at port 5000
2. Test camera connections
3. Verify real-time updates
4. Check monitoring metrics
