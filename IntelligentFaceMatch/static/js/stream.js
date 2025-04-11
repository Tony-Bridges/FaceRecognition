// Stream JavaScript - Handles camera streaming and face registration

// DOM Elements
const cameraSelect = document.getElementById('cameraSelect');
const cameraFeed = document.getElementById('cameraFeed');
const captureBtn = document.getElementById('captureBtn');
const uploadBtn = document.getElementById('uploadBtn');
const imageUpload = document.getElementById('imageUpload');
const facePreview = document.getElementById('facePreview');
const noPreview = document.getElementById('noPreview');
const personName = document.getElementById('personName');
const registerBtn = document.getElementById('registerBtn');
const verifyLiveness = document.getElementById('verifyLiveness');

// Quality elements
const qualityScore = document.getElementById('qualityScore');
const qualityScoreBar = document.getElementById('qualityScoreBar');
const qualityMessage = document.getElementById('qualityMessage');

// Step elements
const steps = [
    document.getElementById('step1'),
    document.getElementById('step2'),
    document.getElementById('step3'),
    document.getElementById('step4'),
    document.getElementById('step5')
];

// Quality bars
const qualityBars = {
    sharpness: document.getElementById('sharpnessBar'),
    brightness: document.getElementById('brightnessBar'),
    contrast: document.getElementById('contrastBar'),
    faceSize: document.getElementById('faceSizeBar'),
    pose: document.getElementById('poseBar'),
    eye: document.getElementById('eyeBar')
};

// Quality values
const qualityValues = {
    sharpness: document.getElementById('sharpnessValue'),
    brightness: document.getElementById('brightnessValue'),
    contrast: document.getElementById('contrastValue'),
    faceSize: document.getElementById('faceSizeValue'),
    pose: document.getElementById('poseValue'),
    eye: document.getElementById('eyeValue')
};

// Global variables
let selectedCamera = null;
let capturedImage = null;
let imageQuality = null;
let streamInterval = null;
let livenessSessionId = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Setup event listeners
    setupEventListeners();
    
    // Initialize camera selector
    initializeCameraSelect();
});

// Initialize camera select dropdown
function initializeCameraSelect() {
    cameraSelect.addEventListener('change', function() {
        selectedCamera = this.value;
        
        if (selectedCamera) {
            captureBtn.disabled = false;
            startCameraStream();
            updateStepStatus(0, 'completed');
            updateStepStatus(1, 'active');
        } else {
            captureBtn.disabled = true;
            stopCameraStream();
            updateStepStatus(0, 'active');
            updateStepStatus(1, '');
        }
    });
}

// Start camera stream
function startCameraStream() {
    // Stop any existing stream
    stopCameraStream();
    
    // Start new stream
    cameraFeed.src = `/video/${selectedCamera}`;
    cameraFeed.style.display = 'block';
}

// Stop camera stream
function stopCameraStream() {
    if (streamInterval) {
        clearInterval(streamInterval);
        streamInterval = null;
    }
    
    cameraFeed.src = '';
}

// Capture image from camera
function captureImageFromCamera() {
    // In a production app, we would capture the image from the camera feed
    // Here, we use the current frame from the video feed
    
    // Since we can't directly access the video frame, 
    // we'll simulate by using the current state of the video element
    capturedImage = cameraFeed.src;
    
    // Show the preview
    facePreview.src = capturedImage;
    facePreview.style.display = 'block';
    noPreview.style.display = 'none';
    
    // Enable the name field
    personName.disabled = false;
    
    // Update the step status
    updateStepStatus(1, 'completed');
    updateStepStatus(2, 'active');
    
    // Analyze face quality
    analyzeFaceQuality();
}

// Handle file upload
function handleFileUpload(file) {
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        // Set the image as the preview
        facePreview.src = e.target.result;
        facePreview.style.display = 'block';
        noPreview.style.display = 'none';
        
        // Store the image data
        capturedImage = e.target.result;
        
        // Enable the name field
        personName.disabled = false;
        
        // Update step status
        updateStepStatus(0, 'completed');
        updateStepStatus(1, 'completed');
        updateStepStatus(2, 'active');
        
        // Analyze face quality
        analyzeFaceQuality();
    };
    
    reader.readAsDataURL(file);
}

// Analyze face quality
async function analyzeFaceQuality() {
    if (!capturedImage) return;
    
    try {
        // Create a FormData object
        const formData = new FormData();
        
        // Convert base64 to blob if needed
        let imageBlob;
        
        if (capturedImage.startsWith('data:')) {
            // Convert data URL to Blob
            const byteString = atob(capturedImage.split(',')[1]);
            const mimeString = capturedImage.split(',')[0].split(':')[1].split(';')[0];
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            
            imageBlob = new Blob([ab], { type: mimeString });
        } else if (capturedImage.startsWith('/video/')) {
            // For demonstration, we'll create a simple blob
            // In a real app, you'd need to capture the actual frame from the video
            imageBlob = new Blob([], { type: 'image/jpeg' });
        } else {
            // Use as-is
            imageBlob = capturedImage;
        }
        
        formData.append('image', imageBlob);
        
        // Send to API
        const response = await fetch('/api/faces/quality', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // Store quality data
        imageQuality = result;
        
        // Update UI
        updateQualityUI(result);
        
        // Enable register button if name is filled
        checkRegisterReady();
        
        // Update step status
        updateStepStatus(3, 'completed');
        updateStepStatus(4, 'active');
    } catch (error) {
        console.error('Error analyzing face quality:', error);
        qualityMessage.textContent = 'Error analyzing face quality. Please try again.';
        qualityMessage.className = 'small text-danger';
    }
}

// Update quality UI
function updateQualityUI(qualityData) {
    const score = qualityData.quality_score;
    const factors = qualityData.quality_factors;
    
    // Update overall score
    qualityScore.textContent = score.toFixed(2);
    qualityScoreBar.style.width = `${score * 100}%`;
    
    // Set color based on score
    if (score >= 0.7) {
        qualityScoreBar.className = 'progress-bar bg-success';
        qualityMessage.textContent = 'Excellent quality! This face will be recognized reliably.';
        qualityMessage.className = 'small text-success';
    } else if (score >= 0.5) {
        qualityScoreBar.className = 'progress-bar bg-warning';
        qualityMessage.textContent = 'Acceptable quality. Consider retaking with better lighting or positioning.';
        qualityMessage.className = 'small text-warning';
    } else {
        qualityScoreBar.className = 'progress-bar bg-danger';
        qualityMessage.textContent = 'Poor quality. Please retake the image with better conditions.';
        qualityMessage.className = 'small text-danger';
    }
    
    // Update individual factors
    if (factors) {
        updateQualityFactor('sharpness', factors.sharpness);
        updateQualityFactor('brightness', factors.brightness);
        updateQualityFactor('contrast', factors.contrast);
        updateQualityFactor('faceSize', factors.face_size);
        updateQualityFactor('pose', factors.pose_deviation);
        updateQualityFactor('eye', factors.eye_openness);
    }
}

// Update individual quality factor
function updateQualityFactor(name, value) {
    if (value === undefined) return;
    
    const bar = qualityBars[name];
    const valueEl = qualityValues[name];
    
    if (bar && valueEl) {
        bar.style.width = `${value * 100}%`;
        valueEl.textContent = value.toFixed(2);
        
        // Set color based on value
        if (value >= 0.7) {
            bar.className = 'progress-bar bg-info';
        } else if (value >= 0.5) {
            bar.className = 'progress-bar bg-warning';
        } else {
            bar.className = 'progress-bar bg-danger';
        }
    }
}

// Check if registration is ready
function checkRegisterReady() {
    const name = personName.value.trim();
    
    if (name && capturedImage) {
        registerBtn.disabled = false;
    } else {
        registerBtn.disabled = true;
    }
}

// Register face
async function registerFace() {
    const name = personName.value.trim();
    
    if (!name || !capturedImage) {
        alert('Please provide a name and image');
        return;
    }
    
    // Disable register button
    registerBtn.disabled = true;
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('name', name);
        formData.append('verify_liveness', verifyLiveness.checked);
        
        // Convert image if needed
        let imageBlob;
        
        if (capturedImage.startsWith('data:')) {
            // Convert data URL to Blob
            const byteString = atob(capturedImage.split(',')[1]);
            const mimeString = capturedImage.split(',')[0].split(':')[1].split(';')[0];
            const ab = new ArrayBuffer(byteString.length);
            const ia = new Uint8Array(ab);
            
            for (let i = 0; i < byteString.length; i++) {
                ia[i] = byteString.charCodeAt(i);
            }
            
            imageBlob = new Blob([ab], { type: mimeString });
        } else if (capturedImage.startsWith('/video/')) {
            // For demonstration, we'll create a simple blob
            // In a real app, you'd need to capture the actual frame from the video
            imageBlob = new Blob([], { type: 'image/jpeg' });
        } else {
            // Use as-is
            imageBlob = capturedImage;
        }
        
        formData.append('image', imageBlob);
        
        // Send request
        const response = await fetch('/api/faces/register', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // Show result
        document.getElementById('resultsCard').style.display = 'block';
        
        if (result.success) {
            // Show success message
            document.getElementById('registrationSuccess').style.display = 'block';
            document.getElementById('registrationError').style.display = 'none';
            document.getElementById('successMessage').textContent = 'Face registered successfully!';
            
            // Show result details
            document.getElementById('resultDetails').style.display = 'block';
            document.getElementById('resultFaceId').textContent = result.face_id;
            document.getElementById('resultQuality').textContent = result.quality_score.toFixed(2);
            
            // Update step status
            updateStepStatus(4, 'completed');
        } else {
            // Show error message
            document.getElementById('registrationSuccess').style.display = 'none';
            document.getElementById('registrationError').style.display = 'block';
            document.getElementById('errorMessage').textContent = result.message || 'Error registering face';
            
            // Hide result details
            document.getElementById('resultDetails').style.display = 'none';
        }
    } catch (error) {
        console.error('Error registering face:', error);
        
        // Show error message
        document.getElementById('resultsCard').style.display = 'block';
        document.getElementById('registrationSuccess').style.display = 'none';
        document.getElementById('registrationError').style.display = 'block';
        document.getElementById('errorMessage').textContent = 'Error connecting to server';
        document.getElementById('resultDetails').style.display = 'none';
    } finally {
        // Re-enable register button
        registerBtn.disabled = false;
    }
}

// Update step status
function updateStepStatus(stepIndex, status) {
    // Clear all statuses
    if (steps[stepIndex]) {
        steps[stepIndex].className = 'step';
        
        if (status) {
            steps[stepIndex].classList.add(status);
        }
    }
}

// Reset registration form
function resetRegistration() {
    // Reset form fields
    personName.value = '';
    personName.disabled = true;
    facePreview.style.display = 'none';
    noPreview.style.display = 'block';
    verifyLiveness.checked = false;
    
    // Reset buttons
    registerBtn.disabled = true;
    
    // Reset captured image
    capturedImage = null;
    
    // Reset quality display
    qualityScore.textContent = '-';
    qualityScoreBar.style.width = '0%';
    qualityMessage.textContent = '';
    
    // Reset quality factors
    Object.keys(qualityBars).forEach(key => {
        qualityBars[key].style.width = '0%';
        qualityValues[key].textContent = '-';
    });
    
    // Reset steps
    steps.forEach((step, index) => {
        if (index === 0) {
            updateStepStatus(index, 'active');
        } else {
            updateStepStatus(index, '');
        }
    });
    
    // Hide results
    document.getElementById('resultsCard').style.display = 'none';
    document.getElementById('registrationSuccess').style.display = 'none';
    document.getElementById('registrationError').style.display = 'none';
    document.getElementById('resultDetails').style.display = 'none';
}

// Setup event listeners
function setupEventListeners() {
    // Capture button click
    captureBtn.addEventListener('click', captureImageFromCamera);
    
    // Upload button click
    uploadBtn.addEventListener('click', function() {
        imageUpload.click();
    });
    
    // Image upload change
    imageUpload.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            handleFileUpload(this.files[0]);
        }
    });
    
    // Person name input
    personName.addEventListener('input', function() {
        if (this.value.trim()) {
            updateStepStatus(2, 'completed');
            checkRegisterReady();
        } else {
            updateStepStatus(2, 'active');
            registerBtn.disabled = true;
        }
    });
    
    // Register button click
    registerBtn.addEventListener('click', registerFace);
    
    // Register another button
    document.getElementById('registerAnotherBtn').addEventListener('click', resetRegistration);
}
