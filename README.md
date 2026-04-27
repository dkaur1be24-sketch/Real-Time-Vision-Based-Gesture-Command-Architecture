Gesture-Based Media Control System using Face Recognition and MediaPipe

Problem Statement
In modern computing systems, users still rely heavily on traditional input devices such as keyboards, mice, and touchscreens. These methods are not always efficient, especially in scenarios requiring hands-free interaction.

This project addresses the need for a natural, contactless, and intelligent human-computer interaction system that allows users to control media and system functions using hand gestures and facial expressions.

Importance:
Reduces dependency on physical devices
Enables accessibility for disabled users
Useful in smart environments (classrooms, IoT systems, automation)

Methodology:
Input (Webcam)
      ↓
Face Detection + Face Recognition
      ↓
Face Mesh (Smile Detection)
      ↓
Hand Tracking (MediaPipe Hands)
      ↓
Gesture Classification
      ↓
Action Execution


Stages:
   Input Source: Live webcam feed
   Preprocessing: Frame resizing, RGB conversion
Feature Extraction:
  Face landmarks (MediaPipe Face Mesh)
  Hand landmarks (MediaPipe Hands)
Inference:
  Smile detection using MAR
  Finger counting logic
Output:
  System control (volume, browser, exit)

5. Model Details
   
Models Used:
MediaPipe Face Mesh (468 landmarks model)
MediaPipe Hands (21 landmarks model)
face_recognition (dlib-based encoding model)

Input Format:
RGB frames from webcam (OpenCV)

Frameworks:
OpenCV
MediaPipe
face_recognition (dlib)
PyAutoGUI

Optimization:
Frame resizing for faster face recognition
Encoding interval optimization (every N frames)
Lightweight landmark-based detection instead of deep CNN

6. Training Details
Dataset Used:
  Pre-trained MediaPipe models (Google)
  face_recognition pre-trained encodings
Training Procedure:
  No custom training required
  Uses pre-trained landmark detection models
  Face encoding comparison for identity matching
Performance Graphs:
  Not applicable (no custom training)
  Real-time performance evaluated instead

7. Results / Output
System Output:
  Smile detection triggers YouTube video playback
  Open palm increases system volume
  Closed fist decreases system volume
  V-sign gesture exits system safely

Performance Metrics:
  Average FPS: 20–30 FPS (depending on hardware)
  Low latency real-time gesture response
  Face recognition delay reduced using frame skipping

  8. Setup Instructions
     Step 1: Install Dependencies
     pip install opencv-python mediapipe face-recognition pyautogui numpy

     Step 2: Run Project
     python facial_gestures.py
     
     Step 3: Controls
     Smile → Open YouTube
     Palm → Volume Up
     Fist → Volume Down
     V-sign → Exit system

  
     
