# Real-time AI-based Exam Monitoring System

A Python-based system that monitors student attention during exams using computer vision and AI. It detects face direction and eye gaze to identify when a student is looking away from the screen.

## Features

- **Real-time Face Direction Detection**: Detects if the student's face is turned left, right, or centered
- **Eye Gaze Detection**: Monitors eye movement direction (left, right, or center)
- **Warning System**: Triggers a warning if the student looks away for more than 15 seconds
- **Visual Feedback**: Displays real-time status with color-coded indicators
- **Logging**: Automatically logs warning events with timestamps
- **Screenshot Capture**: Saves screenshots when warnings are triggered
- **Audio Alert**: Plays a beep sound when warning is activated (Windows)

## Requirements

- Python 3.7 or higher
- Webcam/Camera
- Windows OS (for winsound beep feature)

## Installation

### Step 1: Install Python

If you don't have Python installed:
1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Verify installation by opening Command Prompt and typing:
   ```bash
   python --version
   ```

### Step 2: Create Virtual Environment (Recommended)

1. Open Command Prompt or PowerShell in the project directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   # On Windows (Command Prompt)
   venv\Scripts\activate
   
   # On Windows (PowerShell)
   venv\Scripts\Activate.ps1
   ```

### Step 3: Install Dependencies

1. Make sure you're in the project directory
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `opencv-python` - For video capture and image processing
   - `mediapipe` - For face mesh detection
   - `numpy` - For numerical operations
   - `protobuf<5.0.0` - Required for MediaPipe compatibility

### Step 4: Verify Installation

Test if all packages are installed correctly:
```bash
python -c "import cv2; import mediapipe as mp; import numpy; print('MediaPipe version:', mp.__version__); print('Has solutions:', hasattr(mp, 'solutions')); print('All packages installed successfully!')"
```

**Important Note:** Make sure you're using the virtual environment Python. If you see `AttributeError: module 'mediapipe' has no attribute 'solutions'`, you may need to:
1. Activate your virtual environment: `exm_venv\Scripts\activate`
2. Reinstall MediaPipe: `pip install mediapipe==0.10.13`

## Usage

### Running the System

1. Make sure your webcam is connected and working
2. Run the monitoring system:
   ```bash
   python exam_monitoring.py
   ```
3. Position yourself in front of the camera
4. The system will start monitoring automatically

### Controls

- **Press 'q'** - Quit the application
- The system runs automatically once started

### Understanding the Display

The screen shows:
- **Face Direction**: LEFT / RIGHT / CENTER (green when centered, orange when away)
- **Eye Direction**: LEFT / RIGHT / CENTER (green when centered, orange when away)
- **Timer**: Shows elapsed time since looking away (0.0s / 15.0s)
- **Warning Status**: Displays "WARNING ACTIVE" when threshold is exceeded

### Warning System

- **Warning Trigger**: Activates when face OR eyes are looking away for **15 seconds continuously**
- **Visual Warning**: 
  - Red border around the frame
  - "WARNING: LOOKING AWAY" text at the top
- **Audio Warning**: Beep sound plays when warning activates
- **Automatic Actions**:
  - Screenshot saved to `logs/` folder
  - Warning logged to `logs/warnings.log`
  - Console message printed

### Timer Reset

The timer automatically resets when:
- Face returns to center position
- Eyes return to center position
- No face is detected

## Project Structure

```
exam_Monitering/
│
├── exam_monitoring.py    # Main application file
├── requirements.txt      # Python dependencies
├── README.md            # This file
│
└── logs/                # Generated folder (created automatically)
    ├── warnings.log     # Warning event log
    └── warning_*.jpg    # Screenshot images
```

## Configuration

You can modify these constants in `exam_monitoring.py`:

- `WARNING_TIME_SECONDS = 15` - Time threshold before warning (in seconds)
- `FACE_TURN_THRESHOLD = 0.02` - Sensitivity for face direction detection
- `EYE_GAZE_THRESHOLD = 0.015` - Sensitivity for eye gaze detection

## Troubleshooting

### Webcam Not Working

1. Check if webcam is connected
2. Close other applications using the webcam
3. Try changing the camera index in code (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)

### Poor Detection Accuracy

1. Ensure good lighting in the room
2. Position yourself directly in front of the camera
3. Keep your face clearly visible
4. Adjust thresholds in the code if needed

### Import Errors

If you get import errors:
```bash
pip install --upgrade opencv-python mediapipe numpy
```

### Performance Issues

1. Close other applications to free up resources
2. Reduce frame resolution in code (modify `cap.set(cv2.CAP_PROP_FRAME_WIDTH, ...)`)
3. Ensure you have a good webcam (720p or higher recommended)

## Technical Details

### Face Direction Detection

- Uses nose tip and cheek landmarks
- Calculates relative positions to determine orientation
- Normalized by frame width for consistency

### Eye Gaze Detection

- Uses iris landmarks from MediaPipe Face Mesh
- Compares iris position to eye center
- Averages both eyes for accuracy

### Timer Logic

- Tracks continuous time when face/eyes are away
- Resets immediately when returning to center
- Independent timers for face and eyes (uses maximum)

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed correctly
3. Ensure your webcam is working with other applications

## Future Enhancements

Possible improvements:
- Multiple student detection
- Network streaming for remote monitoring
- Database integration for logging
- Web dashboard for monitoring
- Mobile app integration

