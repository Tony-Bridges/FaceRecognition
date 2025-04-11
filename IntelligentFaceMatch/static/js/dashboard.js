// Dashboard JavaScript - Handles camera feeds, face display, and real-time monitoring

// Charts for analytics
let recognitionChart;
let cameraActivityChart;

// Cache DOM elements
const cameraFeeds = document.getElementById('cameraFeeds');
const logsTableBody = document.getElementById('logsTableBody');
const facesList = document.getElementById('facesList');
const registeredFacesCount = document.getElementById('registeredFacesCount');
const activeCamerasCount = document.getElementById('activeCamerasCount');

// Modal elements
const addCameraModal = new bootstrap.Modal(document.getElementById('addCameraModal'));
const cameraSettingsModal = new bootstrap.Modal(document.getElementById('cameraSettingsModal'));
const faceDetailsModal = new bootstrap.Modal(document.getElementById('faceDetailsModal'));

// Global data
let cameras = [];
let faces = [];
let recognitionLogs = [];

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    // Initialize charts
    initCharts();
    
    // Load initial data
    loadFaces();
    loadCameras();
    
    // Set up polling for logs and updates
    setInterval(updateRecognitionLogs, 5000);
    setInterval(updateCameraStatus, 10000);
    
    // Setup event listeners
    setupEventListeners();
});

// Initialize dashboard charts
function initCharts() {
    // Recognition rate chart
    const recognitionCtx = document.getElementById('recognitionChart').getContext('2d');
    recognitionChart = new Chart(recognitionCtx, {
        type: 'line',
        data: {
            labels: getTimeLabels(),
            datasets: [{
                label: 'Recognition Events',
                data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                borderColor: 'rgba(75, 192, 192, 1)',
                tension: 0.3,
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
    
    // Camera activity chart
    const cameraCtx = document.getElementById('cameraActivityChart').getContext('2d');
    cameraActivityChart = new Chart(cameraCtx, {
        type: 'bar',
        data: {
            labels: ['Camera 1', 'Camera 2'],
            datasets: [{
                label: 'Frames Processed',
                data: [0, 0],
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Generate time labels for chart (past 10 minutes)
function getTimeLabels() {
    const labels = [];
    for (let i = 9; i >= 0; i--) {
        const date = new Date(Date.now() - i * 60000);
        labels.push(date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    }
    return labels;
}

// Load registered faces from the API
async function loadFaces() {
    try {
        const response = await fetch('/api/faces');
        const data = await response.json();
        
        faces = data.faces;
        updateFacesUI();
        
        // Update total count
        registeredFacesCount.textContent = data.total;
    } catch (error) {
        console.error('Error loading faces:', error);
    }
}

// Load cameras from the API
async function loadCameras() {
    try {
        const response = await fetch('/api/cameras');
        cameras = await response.json();
        
        // Update active camera count
        const activeCameras = cameras.filter(camera => camera.running).length;
        activeCameraCount.textContent = activeCameras;
        
        // Update camera activity chart
        updateCameraActivityChart();
    } catch (error) {
        console.error('Error loading cameras:', error);
    }
}

// Update recognition logs
async function updateRecognitionLogs() {
    try {
        // In a real app, we would fetch new logs from an API endpoint
        // For now, we'll simulate a few test entries
        
        // Use a growing list for the chart
        const currentCount = recognitionChart.data.datasets[0].data;
        currentCount.shift();
        currentCount.push(Math.floor(Math.random() * 5)); // Random value for demonstration
        
        // Update chart
        recognitionChart.data.labels = getTimeLabels();
        recognitionChart.update();
        
        // For a real implementation, we would update the logs table with real data from the API
    } catch (error) {
        console.error('Error updating logs:', error);
    }
}

// Update camera status
async function updateCameraStatus() {
    try {
        const response = await fetch('/api/cameras');
        cameras = await response.json();
        
        // Update active camera count
        const activeCameras = cameras.filter(camera => camera.running).length;
        activeCamerasCount.textContent = activeCameras;
        
        // Update camera activity chart
        updateCameraActivityChart();
    } catch (error) {
        console.error('Error updating camera status:', error);
    }
}

// Update camera activity chart
function updateCameraActivityChart() {
    if (!cameras.length) return;
    
    // Update chart with camera names and frame counts
    cameraActivityChart.data.labels = cameras.map(camera => camera.name);
    cameraActivityChart.data.datasets[0].data = cameras.map(camera => camera.frame_count % 100); // Modulo to keep values reasonable
    cameraActivityChart.update();
}

// Update faces display in the UI
function updateFacesUI() {
    if (!faces.length) {
        facesList.innerHTML = `
            <div class="col-12 text-center py-4">
                <p class="text-muted">No faces registered yet.</p>
                <a href="/register" class="btn btn-sm btn-primary">
                    <i class="fas fa-user-plus me-1"></i> Register a Face
                </a>
            </div>
        `;
        return;
    }
    
    let html = '';
    faces.forEach(face => {
        html += `
            <div class="col-6 mb-3">
                <div class="card bg-dark border-secondary face-card" data-id="${face.face_id}">
                    <div class="card-body text-center p-2">
                        <div class="face-icon mb-2">
                            <i class="fas fa-user-circle fa-2x"></i>
                        </div>
                        <h6 class="mb-0">${face.name}</h6>
                        <small class="text-muted">
                            Q: ${(face.quality_score || 0).toFixed(2)}
                        </small>
                        <button class="btn btn-sm btn-outline-danger mt-2 delete-face" data-id="${face.face_id}">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    });
    
    facesList.innerHTML = html;
    
    // Add event listeners to the face cards
    document.querySelectorAll('.face-card').forEach(card => {
        card.addEventListener('click', (e) => {
            if (!e.target.classList.contains('delete-face')) {
                const faceId = card.getAttribute('data-id');
                showFaceDetails(faceId);
            }
        });
    });
    
    // Add event listeners to delete buttons
    document.querySelectorAll('.delete-face').forEach(button => {
        button.addEventListener('click', (e) => {
            e.stopPropagation();
            const faceId = button.getAttribute('data-id');
            deleteFace(faceId);
        });
    });
}

// Show face details in modal
function showFaceDetails(faceId) {
    const face = faces.find(f => f.face_id === faceId);
    if (!face) return;
    
    // Populate modal fields
    document.getElementById('faceIdField').value = face.face_id;
    document.getElementById('faceNameField').value = face.name;
    
    // Set quality score
    const qualityScore = face.quality_score || 0;
    const qualityBar = document.getElementById('faceQualityBar');
    qualityBar.style.width = `${qualityScore * 100}%`;
    qualityBar.className = 'progress-bar';
    
    if (qualityScore > 0.7) {
        qualityBar.classList.add('bg-success');
    } else if (qualityScore > 0.4) {
        qualityBar.classList.add('bg-warning');
    } else {
        qualityBar.classList.add('bg-danger');
    }
    
    document.getElementById('faceQualityText').textContent = `Quality: ${qualityScore.toFixed(2)}`;
    
    // Set dates
    if (face.metadata) {
        document.getElementById('faceRegDateField').value = face.metadata.registration_date || 'Unknown';
        document.getElementById('faceLastAccessedField').value = face.metadata.last_accessed || 'Never';
        
        // Set metadata
        const metadata = { ...face.metadata };
        // Remove standard fields to leave only additional info
        delete metadata.registered_by;
        delete metadata.device_id;
        delete metadata.registration_date;
        delete metadata.last_accessed;
        
        document.getElementById('faceMetadataField').value = JSON.stringify(metadata, null, 2);
    } else {
        document.getElementById('faceRegDateField').value = 'Unknown';
        document.getElementById('faceLastAccessedField').value = 'Never';
        document.getElementById('faceMetadataField').value = '{}';
    }
    
    // Set modal title
    document.getElementById('faceDetailsTitle').textContent = `Face Details: ${face.name}`;
    
    // Show the modal
    faceDetailsModal.show();
}

// Delete a face
async function deleteFace(faceId) {
    if (!confirm('Are you sure you want to delete this face?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/faces/${faceId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Remove from local array
            faces = faces.filter(face => face.face_id !== faceId);
            
            // Update UI
            updateFacesUI();
            
            // Update counter
            registeredFacesCount.textContent = parseInt(registeredFacesCount.textContent) - 1;
            
            // Close modal if open
            faceDetailsModal.hide();
        } else {
            alert('Failed to delete face: ' + result.message);
        }
    } catch (error) {
        console.error('Error deleting face:', error);
        alert('Error deleting face. Please try again.');
    }
}

// Update face metadata
async function updateFaceMetadata() {
    const faceId = document.getElementById('faceIdField').value;
    const name = document.getElementById('faceNameField').value;
    let metadata = {};
    
    try {
        metadata = JSON.parse(document.getElementById('faceMetadataField').value);
    } catch (e) {
        alert('Invalid JSON in metadata field. Please check the format.');
        return;
    }
    
    try {
        // Update name (would require a separate API call in a real app)
        // For now, we'll just update the metadata
        
        const response = await fetch(`/api/faces/${faceId}/metadata`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(metadata)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update local data
            const face = faces.find(f => f.face_id === faceId);
            if (face) {
                face.name = name;
                face.metadata = { ...face.metadata, ...metadata };
            }
            
            // Update UI
            updateFacesUI();
            
            // Close modal
            faceDetailsModal.hide();
        } else {
            alert('Failed to update face: ' + result.message);
        }
    } catch (error) {
        console.error('Error updating face:', error);
        alert('Error updating face. Please try again.');
    }
}

// Set up all event listeners
function setupEventListeners() {
    // Camera type selector in add camera form
    document.getElementById('cameraType').addEventListener('change', function() {
        const deviceIdField = document.getElementById('deviceIdField');
        const urlField = document.getElementById('urlField');
        
        if (this.value === 'webcam') {
            deviceIdField.classList.remove('d-none');
            urlField.classList.add('d-none');
        } else {
            deviceIdField.classList.add('d-none');
            urlField.classList.remove('d-none');
        }
    });
    
    // Save camera button
    document.getElementById('saveCamera').addEventListener('click', addCamera);
    
    // Update camera button
    document.getElementById('updateCamera').addEventListener('click', updateCamera);
    
    // Delete camera button
    document.getElementById('deleteCamera').addEventListener('click', deleteCamera);
    
    // Update face button
    document.getElementById('updateFaceBtn').addEventListener('click', updateFaceMetadata);
    
    // Delete face button in modal
    document.getElementById('deleteFaceBtn').addEventListener('click', function() {
        const faceId = document.getElementById('faceIdField').value;
        deleteFace(faceId);
    });
    
    // Camera settings modal
    document.getElementById('cameraSettingsModal').addEventListener('show.bs.modal', function(event) {
        const button = event.relatedTarget;
        const cameraId = button.getAttribute('data-id');
        const camera = cameras.find(c => c.camera_id === cameraId);
        
        if (camera) {
            document.getElementById('editCameraId').value = camera.camera_id;
            document.getElementById('editCameraName').value = camera.name;
            document.getElementById('editDeviceId').value = camera.device_id || '';
            document.getElementById('editStreamUrl').value = camera.url || '';
            document.getElementById('editEnableCamera').checked = camera.running;
        }
    });
    
    // Toggle camera buttons
    document.querySelectorAll('.toggle-camera').forEach(button => {
        button.addEventListener('click', function() {
            const cameraId = this.getAttribute('data-id');
            const isRunning = this.getAttribute('data-running') === 'true';
            
            if (isRunning) {
                stopCamera(cameraId, this);
            } else {
                startCamera(cameraId, this);
            }
        });
    });
}

// Add a new camera
async function addCamera() {
    const name = document.getElementById('cameraName').value.trim();
    if (!name) {
        alert('Please enter a camera name');
        return;
    }
    
    const type = document.getElementById('cameraType').value;
    let deviceId = null;
    let url = null;
    
    if (type === 'webcam') {
        deviceId = parseInt(document.getElementById('deviceId').value);
    } else {
        url = document.getElementById('streamUrl').value.trim();
        if (!url) {
            alert('Please enter a valid stream URL');
            return;
        }
    }
    
    const enabled = document.getElementById('enableCamera').checked;
    
    try {
        const cameraId = 'cam_' + Date.now().toString(36); // Generate a simple ID
        
        const response = await fetch('/api/cameras', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                camera_id: cameraId,
                name: name,
                device_id: deviceId,
                url: url,
                enabled: enabled,
                config: {}
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Close modal
            addCameraModal.hide();
            
            // Reload cameras
            loadCameras();
            
            // Reload page to show new camera
            window.location.reload();
        } else {
            alert('Failed to add camera: ' + result.message);
        }
    } catch (error) {
        console.error('Error adding camera:', error);
        alert('Error adding camera. Please try again.');
    }
}

// Update a camera
async function updateCamera() {
    const cameraId = document.getElementById('editCameraId').value;
    const name = document.getElementById('editCameraName').value.trim();
    const deviceId = document.getElementById('editDeviceId').value ? parseInt(document.getElementById('editDeviceId').value) : null;
    const url = document.getElementById('editStreamUrl').value.trim();
    const enabled = document.getElementById('editEnableCamera').checked;
    
    if (!name) {
        alert('Please enter a camera name');
        return;
    }
    
    try {
        const response = await fetch('/api/cameras', {
            method: 'POST', // Using POST for update as well
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                camera_id: cameraId,
                name: name,
                device_id: deviceId,
                url: url,
                enabled: enabled,
                config: {}
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Close modal
            cameraSettingsModal.hide();
            
            // Reload cameras
            loadCameras();
            
            // Reload page to show changes
            window.location.reload();
        } else {
            alert('Failed to update camera: ' + result.message);
        }
    } catch (error) {
        console.error('Error updating camera:', error);
        alert('Error updating camera. Please try again.');
    }
}

// Delete a camera
async function deleteCamera() {
    const cameraId = document.getElementById('editCameraId').value;
    
    if (!confirm('Are you sure you want to delete this camera?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/cameras/${cameraId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Close modal
            cameraSettingsModal.hide();
            
            // Reload cameras
            loadCameras();
            
            // Reload page to show changes
            window.location.reload();
        } else {
            alert('Failed to delete camera: ' + result.message);
        }
    } catch (error) {
        console.error('Error deleting camera:', error);
        alert('Error deleting camera. Please try again.');
    }
}

// Start a camera
async function startCamera(cameraId, buttonElement) {
    try {
        const response = await fetch(`/api/cameras/${cameraId}/start`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update button
            buttonElement.innerHTML = '<i class="fas fa-pause"></i>';
            buttonElement.setAttribute('data-running', 'true');
            
            // Update camera status
            const camera = cameras.find(c => c.camera_id === cameraId);
            if (camera) {
                camera.running = true;
            }
            
            // Update active count
            const activeCameras = cameras.filter(c => c.running).length;
            activeCamerasCount.textContent = activeCameras;
        } else {
            alert('Failed to start camera');
        }
    } catch (error) {
        console.error('Error starting camera:', error);
        alert('Error starting camera. Please try again.');
    }
}

// Stop a camera
async function stopCamera(cameraId, buttonElement) {
    try {
        const response = await fetch(`/api/cameras/${cameraId}/stop`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update button
            buttonElement.innerHTML = '<i class="fas fa-play"></i>';
            buttonElement.setAttribute('data-running', 'false');
            
            // Update camera status
            const camera = cameras.find(c => c.camera_id === cameraId);
            if (camera) {
                camera.running = false;
            }
            
            // Update active count
            const activeCameras = cameras.filter(c => c.running).length;
            activeCamerasCount.textContent = activeCameras;
        } else {
            alert('Failed to stop camera');
        }
    } catch (error) {
        console.error('Error stopping camera:', error);
        alert('Error stopping camera. Please try again.');
    }
}
