import cv2
import mediapipe as mp
import numpy as np
import math
import json
from deepface import DeepFace


def analyze_video(video_path, output_json_path="full_analysis.json"):
    # Mediapipe setup
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face.FaceMesh(refine_landmarks=True)
    pose = mp_pose.Pose()

    # Landmark indices
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE = [263, 362]
    RIGHT_EYE = [133, 33]

    # Util functions
    def get_iris_center(landmarks, indices, w, h):
        x = [landmarks[i].x * w for i in indices]
        y = [landmarks[i].y * h for i in indices]
        return int(np.mean(x)), int(np.mean(y))

    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Capture video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    prev_nose = None
    results_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe face & pose detection
        face_results = face_mesh.process(rgb)
        pose_results = pose.process(rgb)

        eye_contact = False
        head_direction = "Unknown"
        posture_status = "Not Detected"
        smile_status = "Unknown"
        shoulder_diff = None

        # ========== Eye Contact Detection ==========
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark

            left_iris = get_iris_center(landmarks, LEFT_IRIS, w, h)
            right_iris = get_iris_center(landmarks, RIGHT_IRIS, w, h)

            l_outer = int(landmarks[LEFT_EYE[0]].x * w)
            l_inner = int(landmarks[LEFT_EYE[1]].x * w)
            r_inner = int(landmarks[RIGHT_EYE[0]].x * w)
            r_outer = int(landmarks[RIGHT_EYE[1]].x * w)

            left_range = abs(l_inner - l_outer)
            right_range = abs(r_outer - r_inner)
            l_min = min(l_outer, l_inner)
            r_min = min(r_outer, r_inner)

            left_iris_pos = (left_iris[0] - l_min) / left_range if left_range != 0 else 0.5
            right_iris_pos = (right_iris[0] - r_min) / right_range if right_range != 0 else 0.5

            if 0.40 < left_iris_pos < 0.60 and 0.40 < right_iris_pos < 0.60:
                eye_contact = True

            cv2.circle(frame, left_iris, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_iris, 3, (0, 255, 0), -1)

        # ========== Head Movement ==========
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            nose = landmarks[1]
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
            forehead = landmarks[10]

            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)
            left_x = int(left_cheek.x * w)
            right_x = int(right_cheek.x * w)
            forehead_y = int(forehead.y * h)

            # Horizontal and vertical tilt
            horizontal_angle = right_x - left_x
            vertical_tilt = nose_y - forehead_y

            # Movement (compare nose position with previous frame)
            movement = 0
            if prev_nose:
                movement = abs(prev_nose[0] - nose_x) + abs(prev_nose[1] - nose_y)

            # Orientation detection
            if movement < 4:
                head_direction = "Still"
            else:
                head_direction = "Moving"

            prev_nose = (nose_x, nose_y)

        # ========== Posture Detection ==========
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            left = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            shoulder_diff = abs(left[1] - right[1])

            if shoulder_diff < 10:
                posture_status = "Good Posture"
            elif left[1] > right[1]:
                posture_status = "Leaning Left"
            else:
                posture_status = "Leaning Right"

            cv2.line(frame, left, right, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ========== Smile Detection ==========
        if frame_count % 10 == 0:  # Run every 10th frame for speed
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                smile_status = "Smiling 😀" if emotion.lower() == "happy" else "Not Smiling 😐"
            except:
                smile_status = "Face Not Detected"

        # ========== Show Results on Frame ==========
        cv2.putText(frame, f"Eye Contact: {'Yes' if eye_contact else 'No'}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Head: {head_direction}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Posture: {posture_status}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Smile: {smile_status}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Full Interview Analysis", frame)

        # Save per-frame analysis
        results_list.append({
            "frame": frame_count,
            "eye_contact": eye_contact,
            "head_movement": head_direction,
            "posture": posture_status,
            "smile": smile_status,
            "shoulder_diff": shoulder_diff
        })

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save JSON summary
    with open(output_json_path, "w") as f:
        json.dump(results_list, f, indent=4)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ All analysis saved to 'full_analysis.json'")

    return output_json_path
