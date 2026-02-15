"""
Real-time AI-based Exam Monitoring System
Detects face direction and eye gaze to monitor student attention during exams.
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
import winsound

# Import MediaPipe
import mediapipe as mp

# Initialize MediaPipe Face Mesh
# Access solutions after import to ensure proper initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Face Mesh landmarks indices (MediaPipe uses 468 landmarks)
# Key landmarks for face direction detection
NOSE_TIP = 4
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_EYE_INNER = 33
LEFT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 133

# Iris landmarks for eye gaze detection
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Thresholds for detection
FACE_TURN_THRESHOLD = 0.02  # Threshold for face direction detection
EYE_GAZE_THRESHOLD = 0.015   # Threshold for eye gaze detection
WARNING_TIME_SECONDS = 15    # Time in seconds before warning triggers


class ExamMonitor:
    """Main class for exam monitoring system."""
    
    def __init__(self):
        """Initialize the exam monitoring system."""
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Timer tracking
        self.face_away_start_time = None
        self.eyes_away_start_time = None
        self.current_face_direction = "CENTER"
        self.current_eye_direction = "CENTER"
        self.warning_active = False
        
        # Create logs directory if it doesn't exist
        self.logs_dir = "logs"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        # Screenshot counter
        self.screenshot_count = 0
    
    def get_face_direction(self, landmarks, frame_width, frame_height):
        """
        Detect face direction (LEFT, RIGHT, or CENTER) using nose tip and cheek positions.
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            str: "LEFT", "RIGHT", or "CENTER"
        """
        try:
            # Get normalized landmark positions
            nose_tip = landmarks.landmark[NOSE_TIP]
            left_cheek = landmarks.landmark[LEFT_CHEEK]
            right_cheek = landmarks.landmark[RIGHT_CHEEK]
            
            # Convert to pixel coordinates
            nose_x = nose_tip.x * frame_width
            left_cheek_x = left_cheek.x * frame_width
            right_cheek_x = right_cheek.x * frame_width
            
            # Calculate relative positions
            # If nose is closer to left cheek, face is turned right
            # If nose is closer to right cheek, face is turned left
            nose_to_left = abs(nose_x - left_cheek_x)
            nose_to_right = abs(nose_x - right_cheek_x)
            
            # Calculate difference ratio
            diff_ratio = (nose_to_left - nose_to_right) / frame_width
            
            # Determine direction based on threshold
            if diff_ratio > FACE_TURN_THRESHOLD:
                return "LEFT"
            elif diff_ratio < -FACE_TURN_THRESHOLD:
                return "RIGHT"
            else:
                return "CENTER"
                
        except Exception as e:
            print(f"Error in get_face_direction: {e}")
            return "CENTER"
    
    def get_eye_direction(self, landmarks, frame_width, frame_height):
        """
        Detect eye gaze direction (LEFT, RIGHT, or CENTER) using iris positions.
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            str: "LEFT", "RIGHT", or "CENTER"
        """
        try:
            # Get eye corner and iris positions
            left_eye_inner = landmarks.landmark[LEFT_EYE_INNER]
            left_eye_outer = landmarks.landmark[LEFT_EYE_OUTER]
            right_eye_inner = landmarks.landmark[RIGHT_EYE_INNER]
            right_eye_outer = landmarks.landmark[RIGHT_EYE_OUTER]
            
            # Get iris center positions (average of iris landmarks)
            left_iris_center = np.mean([
                [landmarks.landmark[i].x * frame_width, landmarks.landmark[i].y * frame_height]
                for i in LEFT_IRIS
            ], axis=0)
            
            right_iris_center = np.mean([
                [landmarks.landmark[i].x * frame_width, landmarks.landmark[i].y * frame_height]
                for i in RIGHT_IRIS
            ], axis=0)
            
            # Calculate eye center positions
            left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) / 2 * frame_width
            right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) / 2 * frame_width
            
            # Calculate iris offset from eye center
            left_iris_offset = left_iris_center[0] - left_eye_center_x
            right_iris_offset = right_iris_center[0] - right_eye_center_x
            
            # Average offset for both eyes
            avg_offset = (left_iris_offset + right_iris_offset) / 2
            
            # Normalize by eye width
            eye_width = abs(left_eye_outer.x - left_eye_inner.x) * frame_width
            normalized_offset = avg_offset / eye_width if eye_width > 0 else 0
            
            # Determine direction
            if normalized_offset > EYE_GAZE_THRESHOLD:
                return "RIGHT"
            elif normalized_offset < -EYE_GAZE_THRESHOLD:
                return "LEFT"
            else:
                return "CENTER"
                
        except Exception as e:
            print(f"Error in get_eye_direction: {e}")
            return "CENTER"
    
    def update_timer(self, face_direction, eye_direction):
        """
        Update timer based on current face and eye directions.
        
        Args:
            face_direction: Current face direction
            eye_direction: Current eye gaze direction
        """
        current_time = time.time()
        
        # Update face direction timer
        if face_direction in ["LEFT", "RIGHT"]:
            if self.face_away_start_time is None:
                self.face_away_start_time = current_time
        else:
            # Reset timer if looking back to center
            if self.face_away_start_time is not None:
                self.face_away_start_time = None
        
        # Update eye direction timer
        if eye_direction in ["LEFT", "RIGHT"]:
            if self.eyes_away_start_time is None:
                self.eyes_away_start_time = current_time
        else:
            # Reset timer if looking back to center
            if self.eyes_away_start_time is not None:
                self.eyes_away_start_time = None
    
    def get_elapsed_time(self):
        """
        Get elapsed time since face/eyes started looking away.
        
        Returns:
            float: Elapsed time in seconds, or 0 if not looking away
        """
        current_time = time.time()
        max_elapsed = 0
        
        if self.face_away_start_time is not None:
            elapsed = current_time - self.face_away_start_time
            max_elapsed = max(max_elapsed, elapsed)
        
        if self.eyes_away_start_time is not None:
            elapsed = current_time - self.eyes_away_start_time
            max_elapsed = max(max_elapsed, elapsed)
        
        return max_elapsed
    
    def check_warning(self):
        """
        Check if warning should be triggered based on elapsed time.
        
        Returns:
            bool: True if warning should be active
        """
        elapsed = self.get_elapsed_time()
        should_warn = elapsed >= WARNING_TIME_SECONDS
        
        # Trigger warning actions if just crossed threshold
        if should_warn and not self.warning_active:
            self.trigger_warning()
        
        self.warning_active = should_warn
        return should_warn
    
    def trigger_warning(self):
        """Trigger warning actions: beep, log, and screenshot."""
        try:
            # Play beep sound
            winsound.Beep(1000, 500)  # 1000 Hz for 500ms
            
            # Log to file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] WARNING: Student looking away - Face: {self.current_face_direction}, Eyes: {self.current_eye_direction}\n"
            
            log_file = os.path.join(self.logs_dir, "warnings.log")
            with open(log_file, "a") as f:
                f.write(log_message)
            
            # Print to console
            print(f"\n{'='*60}")
            print(f"WARNING: LOOKING AWAY")
            print(f"Time: {timestamp}")
            print(f"Face Direction: {self.current_face_direction}")
            print(f"Eye Direction: {self.current_eye_direction}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"Error triggering warning: {e}")
    
    def save_screenshot(self, frame):
        """Save screenshot when warning is triggered."""
        try:
            self.screenshot_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.logs_dir, f"warning_{timestamp}_{self.screenshot_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")
    
    def draw_warning(self, frame):
        """
        Draw warning text and red border on frame.
        
        Args:
            frame: Input frame to draw on
        """
        if self.warning_active:
            # Draw red border
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 10)
            
            # Draw warning text
            warning_text = "WARNING: LOOKING AWAY"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            
            # Get text size for centering
            (text_width, text_height), baseline = cv2.getTextSize(warning_text, font, font_scale, thickness)
            text_x = (frame.shape[1] - text_width) // 2
            text_y = 50
            
            # Draw text with black outline and red fill
            cv2.putText(frame, warning_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, warning_text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
    
    def draw_status(self, frame, face_direction, eye_direction, elapsed_time):
        """
        Draw real-time status information on frame.
        
        Args:
            frame: Input frame to draw on
            face_direction: Current face direction
            eye_direction: Current eye gaze direction
            elapsed_time: Elapsed time since looking away
        """
        # Status panel background
        panel_height = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y_offset = 35
        line_height = 30
        
        # Face direction
        face_color = (0, 255, 0) if face_direction == "CENTER" else (0, 165, 255)
        cv2.putText(frame, f"Face Direction: {face_direction}", 
                   (20, y_offset), font, font_scale, face_color, thickness)
        
        # Eye direction
        eye_color = (0, 255, 0) if eye_direction == "CENTER" else (0, 165, 255)
        cv2.putText(frame, f"Eye Direction: {eye_direction}", 
                   (20, y_offset + line_height), font, font_scale, eye_color, thickness)
        
        # Timer
        timer_color = (0, 255, 0) if elapsed_time < WARNING_TIME_SECONDS else (0, 0, 255)
        timer_text = f"Timer: {elapsed_time:.1f}s / {WARNING_TIME_SECONDS}s"
        cv2.putText(frame, timer_text, 
                   (20, y_offset + line_height * 2), font, font_scale, timer_color, thickness)
        
        # Warning status
        if self.warning_active:
            warning_status = "WARNING ACTIVE"
            cv2.putText(frame, warning_status, 
                       (20, y_offset + line_height * 3), font, font_scale, (0, 0, 255), thickness)
    
    def run(self):
        """Main loop to run the exam monitoring system."""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Exam Monitoring System Started")
        print("Press 'q' to quit")
        print("-" * 60)
        
        screenshot_saved = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with Face Mesh
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Get frame dimensions
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Detect directions
                    face_direction = self.get_face_direction(face_landmarks, frame_width, frame_height)
                    eye_direction = self.get_eye_direction(face_landmarks, frame_width, frame_height)
                    
                    # Update current directions
                    self.current_face_direction = face_direction
                    self.current_eye_direction = eye_direction
                    
                    # Update timer
                    self.update_timer(face_direction, eye_direction)
                    
                    # Check for warning
                    self.check_warning()
                    
                    # Get elapsed time
                    elapsed_time = self.get_elapsed_time()
                    
                    # Save screenshot on first warning trigger
                    if self.warning_active and not screenshot_saved:
                        self.save_screenshot(frame)
                        screenshot_saved = True
                    elif not self.warning_active:
                        screenshot_saved = False
                    
                    # Draw face mesh (optional, can be commented out for cleaner view)
                    # mp_drawing.draw_landmarks(
                    #     frame,
                    #     face_landmarks,
                    #     mp_face_mesh.FACEMESH_CONTOURS,
                    #     None,
                    #     mp_drawing_styles.get_default_face_mesh_contours_style()
                    # )
                else:
                    # No face detected
                    face_direction = "NO FACE"
                    eye_direction = "NO FACE"
                    elapsed_time = 0
                    self.face_away_start_time = None
                    self.eyes_away_start_time = None
                    self.warning_active = False
                    screenshot_saved = False
                
                # Draw status and warnings
                self.draw_status(frame, face_direction, eye_direction, elapsed_time)
                self.draw_warning(frame)
                
                # Display frame
                cv2.imshow('Exam Monitoring System', frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("System shutdown complete")


def main():
    """Main entry point."""
    try:
        monitor = ExamMonitor()
        monitor.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

