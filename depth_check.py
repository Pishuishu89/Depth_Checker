import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points: a (hip), b (knee), c (ankle)
    """
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B (midpoint)
    c = np.array(c)  # Point C

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Open webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB for Mediapipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image)

    # Convert the image back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if result.pose_landmarks:
        # Draw pose landmarks on the video frame
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks for depth analysis
        landmarks = result.pose_landmarks.landmark
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate the angle at the knee
        angle = calculate_angle(hip, knee, ankle)

        # Provide feedback on squat depth
        if angle < 90:
            feedback = "Squat Depth Achieved!"
            color = (0, 255, 0)  # Green
        else:
            feedback = "Go Lower!"
            color = (0, 0, 255)  # Red

        # Display the feedback on the frame
        cv2.putText(image, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"Angle: {int(angle)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the video feed
    cv2.imshow('DepthCheck', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
