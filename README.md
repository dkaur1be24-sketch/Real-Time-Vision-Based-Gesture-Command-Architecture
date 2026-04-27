# Real-Time-Vision-Based-Gesture-Command-Architecture
A real-time AI-powered Human-Computer Interaction system that enables control of media and system actions using hand gestures and facial expressions through a webcam.

This project combines Computer Vision, Face Recognition, and MediaPipe tracking to create a touchless control interface.

Features

👤 Face-Based Personalization
Detects multiple faces in real time
Locks control to a single active user
Ignores strangers for security
Uses face recognition for identity matching

😊 Facial Gesture Control
Smile Detection (MediaPipe Face Mesh)
Hold a smile → Automatically opens YouTube video

Uses Mouth Aspect Ratio (MAR) for accurate smile detection

✋ Hand Gesture Controls
✋ Open Palm → Volume Up
✊ Closed Fist → Volume Down
✌ V-Sign → Exit System (Hold gesture)

🎥 Media Automation
Opens a predefined YouTube video on gesture trigger
Controls system volume using keyboard simulation
Can close browser tab automatically

🧠 Smart Interaction System
Debounce logic prevents accidental triggers
Gesture holding system for stability
Multi-face detection with active user tracking

🛠️ Tech Stack
Python 3.x
OpenCV
MediaPipe (Face Mesh + Hands)
face_recognition
PyAutoGUI
Webbrowser module

📷 System Workflow
Webcam captures real-time video
Face Recognition locks the primary user
MediaPipe detects:
Facial landmarks (smile detection)
Hand landmarks (gesture control)
Actions triggered:
Smile → Open YouTube
Palm → Volume Up
Fist → Volume Down
V-sign → Exit system

Key Concepts Used
Face Landmark Detection (MediaPipe Face Mesh)
Hand Tracking (MediaPipe Hands)
Face Recognition Encoding
Gesture Classification Logic
Real-time Video Processing
Human-Computer Interaction (HCI)

📌 Applications
Touchless system control
Smart classrooms
Accessibility tools for disabled users
AI-based media automation systems
Human-computer interaction research

⚠️ Limitations
Requires good lighting for accuracy
Performance depends on webcam quality
Face recognition may slow on low-end devices

🔮 Future Improvements
Add voice commands integration
Mobile phone gesture control support
Deep learning-based emotion detection
GUI dashboard for customization
Multi-user profile switching
