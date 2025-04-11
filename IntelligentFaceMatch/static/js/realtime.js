/**
 * Real-time updates using Socket.IO for face recognition dashboard
 */

// Socket.IO connection and event handling
let socket;

/**
 * Initialize Socket.IO connection
 */
function initializeSocket() {
    // Connect to the Socket.IO server
    socket = io.connect(window.location.origin, {
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: Infinity
    });
    
    // Connection events
    socket.on('connect', function() {
        console.log('Socket.IO connected.');
        showNotification('Connected', 'Real-time updates enabled', 'success');
        
        // Subscribe to face events
        socket.emit('subscribe', {
            events: ['face_added', 'face_deleted', 'face_recognized']
        });
    });
    
    socket.on('disconnect', function() {
        console.log('Socket.IO disconnected.');
        showNotification('Disconnected', 'Real-time updates disabled', 'warning');
    });
    
    socket.on('connect_error', function(error) {
        console.error('Connection error:', error);
        showNotification('Connection Error', 'Could not connect to real-time service', 'error');
    });
    
    // Face events
    socket.on('face_added', handleFaceAdded);
    socket.on('face_deleted', handleFaceDeleted);
    socket.on('face_recognized', handleFaceRecognized);
    
    // Camera events
    socket.on('camera_connected', handleCameraConnected);
    socket.on('camera_disconnected', handleCameraDisconnected);
    
    // History event
    socket.on('history', handleHistory);
}

/**
 * Handle 'face_added' event
 * @param {Object} event - Event data 
 */
function handleFaceAdded(event) {
    console.log('Face added:', event);
    
    // Update face counter
    if (window.faceCount) {
        const currentCount = parseInt(window.faceCount.innerText) || 0;
        window.faceCount.innerText = currentCount + 1;
    }
    
    // Show notification
    const { data } = event;
    showNotification('Face Added', `${data.name} was added to the database`, 'success');
    
    // Create face card (if we're on the dashboard page)
    const facesList = document.getElementById('faces-list');
    if (facesList) {
        // Create new face card
        createFaceCard(facesList, data);
    }
}

/**
 * Handle 'face_deleted' event
 * @param {Object} event - Event data 
 */
function handleFaceDeleted(event) {
    console.log('Face deleted:', event);
    
    // Update face counter
    if (window.faceCount) {
        const currentCount = parseInt(window.faceCount.innerText) || 0;
        if (currentCount > 0) {
            window.faceCount.innerText = currentCount - 1;
        }
    }
    
    // Show notification
    const { data } = event;
    showNotification('Face Deleted', `${data.name} was removed from the database`, 'info');
    
    // Remove face card (if we're on the dashboard page)
    if (data.face_id) {
        const faceCard = document.querySelector(`.face-card[data-face-id="${data.face_id}"]`);
        if (faceCard) {
            // Animate removal
            faceCard.classList.add('removing');
            
            // Remove from DOM after animation
            setTimeout(() => {
                const container = faceCard.closest('.col-md-6');
                if (container && container.parentNode) {
                    container.parentNode.removeChild(container);
                }
            }, 500);
        }
    }
}

/**
 * Handle 'face_recognized' event
 * @param {Object} event - Event data 
 */
function handleFaceRecognized(event) {
    console.log('Face recognized:', event);
    
    // Show notification
    const { data } = event;
    
    // Format confidence as percentage
    const confidence = data.confidence 
        ? `${Math.round(data.confidence * 100)}%` 
        : 'Unknown';
    
    showNotification('Face Recognized', `${data.name} was recognized (${confidence})`, 'info');
    
    // Add to recognition log
    const recognitionLog = document.getElementById('recognition-log');
    if (recognitionLog) {
        // Remove placeholder if present
        const placeholder = recognitionLog.querySelector('.text-center.text-muted');
        if (placeholder) {
            recognitionLog.removeChild(placeholder);
        }
        
        // Add log entry
        const logEntry = document.createElement('div');
        logEntry.className = 'recognition-log-entry new-entry';
        
        // Format timestamp
        const timestamp = data.timestamp 
            ? new Date(data.timestamp).toLocaleTimeString() 
            : new Date().toLocaleTimeString();
        
        // Get camera name if available
        let cameraName = 'Unknown Camera';
        if (data.camera_id) {
            const cameraElement = document.querySelector(`.camera-card[data-camera-id="${data.camera_id}"]`);
            if (cameraElement) {
                const nameEl = cameraElement.querySelector('h5.mb-0');
                if (nameEl) {
                    cameraName = nameEl.textContent.trim();
                }
            }
        }
        
        logEntry.innerHTML = `
            <div class="log-entry-header">
                <span class="log-entry-time">${timestamp}</span>
                <span class="log-entry-confidence">${confidence}</span>
            </div>
            <div class="log-entry-body">
                <div class="log-entry-name">${data.name}</div>
                <div class="log-entry-camera">${cameraName}</div>
            </div>
        `;
        
        // Add to log
        recognitionLog.insertBefore(logEntry, recognitionLog.firstChild);
        
        // Remove animation class after animation completes
        setTimeout(() => {
            logEntry.classList.remove('new-entry');
        }, 2000);
        
        // Limit number of log entries
        const maxEntries = 50;
        const entries = recognitionLog.querySelectorAll('.recognition-log-entry');
        if (entries.length > maxEntries) {
            for (let i = maxEntries; i < entries.length; i++) {
                recognitionLog.removeChild(entries[i]);
            }
        }
    }
}

/**
 * Handle 'camera_connected' event
 * @param {Object} event - Event data 
 */
function handleCameraConnected(event) {
    console.log('Camera connected:', event);
    
    // Show notification
    const { data } = event;
    showNotification('Camera Connected', `${data.name} is now streaming`, 'success');
    
    // Update camera card status
    if (data.camera_id) {
        const cameraCard = document.querySelector(`.camera-card[data-camera-id="${data.camera_id}"]`);
        if (cameraCard) {
            const statusCircle = cameraCard.querySelector('.status-circle');
            if (statusCircle) {
                statusCircle.classList.remove('status-inactive');
                statusCircle.classList.add('status-active');
            }
        }
    }
}

/**
 * Handle 'camera_disconnected' event
 * @param {Object} event - Event data 
 */
function handleCameraDisconnected(event) {
    console.log('Camera disconnected:', event);
    
    // Show notification
    const { data } = event;
    showNotification('Camera Disconnected', `${data.name} stopped streaming`, 'warning');
    
    // Update camera card status
    if (data.camera_id) {
        const cameraCard = document.querySelector(`.camera-card[data-camera-id="${data.camera_id}"]`);
        if (cameraCard) {
            const statusCircle = cameraCard.querySelector('.status-circle');
            if (statusCircle) {
                statusCircle.classList.remove('status-active');
                statusCircle.classList.add('status-inactive');
            }
        }
    }
}

/**
 * Handle 'history' event
 * @param {Object} event - Event data
 */
function handleHistory(event) {
    console.log('History received:', event);
    
    // Process only if we have valid data
    if (!event || !event.event_type || !event.events || !Array.isArray(event.events)) {
        return;
    }
    
    // Handle specific event types
    switch (event.event_type) {
        case 'face_recognized':
            processRecognitionHistory(event.events);
            break;
            
        default:
            console.log(`Unhandled history type: ${event.event_type}`);
    }
}

/**
 * Process recognition history data
 * @param {Array} events - History events 
 */
function processRecognitionHistory(events) {
    const recognitionLog = document.getElementById('recognition-log');
    if (!recognitionLog || !events || !events.length) return;
    
    // Remove placeholder if present
    const placeholder = recognitionLog.querySelector('.text-center.text-muted');
    if (placeholder) {
        recognitionLog.removeChild(placeholder);
    }
    
    // Sort by timestamp (newest first)
    events.sort((a, b) => {
        const timeA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
        const timeB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
        return timeB - timeA;
    });
    
    // Add entries (up to 50)
    const maxEntries = 50;
    events.slice(0, maxEntries).forEach(data => {
        // Format data for display
        const confidence = data.confidence 
            ? `${Math.round(data.confidence * 100)}%` 
            : 'Unknown';
            
        const timestamp = data.timestamp 
            ? new Date(data.timestamp).toLocaleTimeString() 
            : '';
            
        // Create log entry
        const logEntry = document.createElement('div');
        logEntry.className = 'recognition-log-entry';
        
        logEntry.innerHTML = `
            <div class="log-entry-header">
                <span class="log-entry-time">${timestamp}</span>
                <span class="log-entry-confidence">${confidence}</span>
            </div>
            <div class="log-entry-body">
                <div class="log-entry-name">${data.name || 'Unknown'}</div>
                <div class="log-entry-camera">${data.camera_name || 'Unknown Camera'}</div>
            </div>
        `;
        
        // Add to log
        recognitionLog.appendChild(logEntry);
    });
}

/**
 * Create a face card and append it to the container
 * @param {HTMLElement} container - Container for face cards
 * @param {Object} data - Face data 
 */
function createFaceCard(container, data) {
    // Create wrapper column
    const col = document.createElement('div');
    col.className = 'col-md-6 col-lg-4 mb-4';
    
    // Format quality score
    let qualityScore = 'N/A';
    if (data.quality_score) {
        qualityScore = parseFloat(data.quality_score).toFixed(2);
    }
    
    // Format date
    const dateAdded = data.added_at 
        ? data.added_at.split('T')[0] 
        : 'Unknown';
    
    // Create card HTML
    col.innerHTML = `
        <div class="card bg-dark face-card new-card" data-face-id="${data.face_id}">
            <div class="face-image">
                <div class="face-avatar">
                    <i class="fas fa-user"></i>
                </div>
            </div>
            <div class="card-body">
                <h5 class="card-title face-name">${data.name}</h5>
                <p class="card-text">
                    <small class="text-muted">ID: ${data.face_id}</small>
                </p>
                <p class="card-text">
                    <small class="text-muted">
                        <i class="fas fa-star me-1"></i>Quality: ${qualityScore}
                    </small>
                </p>
                <p class="card-text">
                    <small class="text-muted">
                        <i class="fas fa-calendar me-1"></i>Added: ${dateAdded}
                    </small>
                </p>
            </div>
            <div class="face-actions">
                <button class="btn-delete" data-face-id="${data.face_id}" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `;
    
    // Add delete handler
    const deleteBtn = col.querySelector('.btn-delete');
    if (deleteBtn) {
        deleteBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            const faceId = this.getAttribute('data-face-id');
            if (faceId) {
                const faceName = col.querySelector('.face-name').innerText;
                if (confirm(`Are you sure you want to delete ${faceName}?`)) {
                    deleteFace(faceId);
                }
            }
        });
    }
    
    // Add to container
    if (container.firstChild) {
        container.insertBefore(col, container.firstChild);
    } else {
        container.appendChild(col);
    }
    
    // Remove the "no faces" message if present
    const noFacesMsg = container.querySelector('.alert');
    if (noFacesMsg) {
        container.removeChild(noFacesMsg);
    }
    
    // Remove animation class after animation completes
    setTimeout(() => {
        const card = col.querySelector('.face-card');
        if (card) {
            card.classList.remove('new-card');
        }
    }, 2000);
}

/**
 * Delete a face
 * @param {string} faceId - ID of the face to delete
 */
function deleteFace(faceId) {
    fetch(`/api/faces/delete/${faceId}`, {
        method: 'DELETE',
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Don't need to do anything here, the WebSocket event will handle UI updates
            console.log(`Face ${faceId} deleted successfully.`);
        } else {
            showNotification('Error', data.message || 'Failed to delete face', 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error', 'An error occurred while deleting the face', 'error');
    });
}

/**
 * Show notification to the user
 * @param {string} title - Notification title
 * @param {string} message - Notification message
 * @param {string} type - Notification type (success, info, warning, error) 
 */
function showNotification(title, message, type = 'info') {
    // Check if Toastify is available
    if (typeof Toastify === 'function') {
        // Default styles based on type
        const defaults = {
            success: { background: 'linear-gradient(to right, #43a047, #2e7d32)' },
            info: { background: 'linear-gradient(to right, #039be5, #0277bd)' },
            warning: { background: 'linear-gradient(to right, #ff9800, #ef6c00)' },
            error: { background: 'linear-gradient(to right, #e53935, #c62828)' }
        };
        
        // Create and show toast
        Toastify({
            text: `<b>${title}</b><br>${message}`,
            duration: 3000,
            close: true,
            gravity: 'top',
            position: 'right',
            stopOnFocus: true,
            escapeMarkup: false,
            ...defaults[type]
        }).showToast();
    } else {
        // Fallback to alert if Toastify is not available
        console.log(`${title}: ${message} (${type})`);
    }
}