import cv2
import mediapipe as mp
import numpy as np
import gradio as gr
from PIL import Image

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load exercise images
exercise_images = {
    'Squats': 'squats.png',
    'Bicep Curls': 'bicep_curls.png',
    'Overhead Press': 'overhead_press.png',
    'Tricep Extensions': 'tricep_extensions.png',
    'Chest Press': 'chest_press.png',
    'Rows': 'rows.png',
    'Lunges': 'lunges.png'
}

# Angle calculation function
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

# Real-time feedback thresholds
form_feedback = {
    'Squats': {'min_angle': 80, 'max_angle': 160},
    'Bicep Curls': {'min_angle': 30, 'max_angle': 160},
    'Overhead Press': {'min_angle': 70, 'max_angle': 160},
    'Tricep Extensions': {'min_angle': 30, 'max_angle': 160},
    'Chest Press': {'min_angle': 90, 'max_angle': 160},
    'Rows': {'min_angle': 70, 'max_angle': 160},
    'Lunges': {'min_angle': 80, 'max_angle': 160}
}

# Exercise processing function
def process_video(exercise, reps_goal):
    cap = cv2.VideoCapture(0)
    counters, stages = {ex: 0 for ex in form_feedback}, {ex: None for ex in form_feedback}
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                feedback_text = ''
                wrong_form = False
                
                if exercise == 'Squats':
                    hip, knee, ankle = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                                        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                                        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle = calculate_angle(hip, knee, ankle)
                    
                    if angle < form_feedback[exercise]['min_angle']:
                        feedback_text = 'Go Lower'
                        wrong_form = True
                    elif angle > form_feedback[exercise]['max_angle']:
                        feedback_text = 'Stand Straight'
                        wrong_form = True
                    
                    if angle > 160:
                        stages[exercise] = "up"
                    if angle < 90 and stages[exercise] == 'up':
                        stages[exercise] = "down"
                        counters[exercise] += 1
                
                cv2.putText(image, f'{exercise}: {counters[exercise]}', (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                
                if wrong_form:
                    cv2.putText(image, feedback_text, (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                if counters[exercise] >= reps_goal:
                    cv2.putText(image, 'Workout Complete!', (200,240),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, cv2.LINE_AA)
                    break
                
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            
            except Exception as e:
                pass
            
            _, jpeg = cv2.imencode('.jpg', image)
            yield jpeg.tobytes()
    
    cap.release()

# Define Gradio UI
def gradio_interface(exercise, reps_goal):
    return process_video(exercise, reps_goal)

# Create Gradio Interface
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Radio(['Squats', 'Bicep Curls', 'Overhead Press',
                  'Tricep Extensions', 'Chest Press', 'Rows', 'Lunges'],
                 label='Select Exercise'),
        gr.Slider(minimum=1, maximum=100, value=10, step=1, label='Set Repetition Goal')
    ],
    outputs=gr.Video(label='Live Exercise Feedback'),
    title='OpenCV Gym Exercise Detection',
    description='Perform the selected exercise, and receive real-time feedback on your form.'
)

demo.launch()
