import sys
print(sys.version) 
import cv2
print(cv2.__version__)
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image  # For handling images

# Initialize MediaPipe and Streamlit
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

st.title('OpenCV Gym Exercise Detection')

# Sidebar for exercise selection and settings
st.sidebar.title('Exercise Selection and Settings')

# Exercise selection
exercise = st.sidebar.selectbox(
    'Choose Exercise',
    ('Squats', 'Bicep Curls', 'Overhead Press',
     'Tricep Extensions', 'Chest Press', 'Rows', 'Lunges')
)

# Display the exercise image
# Ensure the image files are in the same directory or provide the correct path
exercise_images = {
    'Squats': 'squats.png',
    'Bicep Curls': 'bicep_curls.png',
    'Overhead Press': 'overhead_press.png',
    'Tricep Extensions': 'tricep_extensions.png',
    'Chest Press': 'chest_press.png',
    'Rows': 'rows.png',
    'Lunges': 'lunges.png'
}

try:
    exercise_image = Image.open(exercise_images[exercise])
    st.sidebar.image(exercise_image, caption=f'How to perform {exercise}', use_column_width=True)
except FileNotFoundError:
    st.sidebar.write("Exercise image not found.")

# Custom repetition limit
REPS_LIMIT = st.sidebar.number_input('Set Repetition Goal', min_value=1, max_value=100, value=10, step=1)

# Reset button
reset = st.sidebar.button('Reset Exercise Session')

# Variables to control the app flow
if 'run' not in st.session_state:
    st.session_state.run = False

if reset:
    st.session_state.run = False  # Stop the current session
    st.session_state.counters = {
        'Squats': 0,
        'Bicep Curls': 0,
        'Overhead Press': 0,
        'Tricep Extensions': 0,
        'Chest Press': 0,
        'Rows': 0,
        'Lunges': 0
    }
    st.session_state.stages = {
        'Squats': None,
        'Bicep Curls': None,
        'Overhead Press': None,
        'Tricep Extensions': None,
        'Chest Press': None,
        'Rows': None,
        'Lunges': None
    }

start = st.checkbox('Start Exercise Session')

# Initialize counters and stages if not already done
if 'counters' not in st.session_state:
    st.session_state.counters = {
        'Squats': 0,
        'Bicep Curls': 0,
        'Overhead Press': 0,
        'Tricep Extensions': 0,
        'Chest Press': 0,
        'Rows': 0,
        'Lunges': 0
    }

if 'stages' not in st.session_state:
    st.session_state.stages = {
        'Squats': None,
        'Bicep Curls': None,
        'Overhead Press': None,
        'Tricep Extensions': None,
        'Chest Press': None,
        'Rows': None,
        'Lunges': None
    }

# Frame window for displaying the video
FRAME_WINDOW = st.image([])

# Define the angle calculation function
def calculate_angle(a, b, c):
    """Calculate the angle between three points (a, b, c)."""
    a = np.array(a)  # First coordinate
    b = np.array(b)  # Second coordinate (vertex)
    c = np.array(c)  # Third coordinate

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Real-time feedback thresholds for form correction
form_feedback = {
    'Squats': {'min_angle': 80, 'max_angle': 160},
    'Bicep Curls': {'min_angle': 30, 'max_angle': 160},
    'Overhead Press': {'min_angle': 70, 'max_angle': 160},
    'Tricep Extensions': {'min_angle': 30, 'max_angle': 160},
    'Chest Press': {'min_angle': 90, 'max_angle': 160},
    'Rows': {'min_angle': 70, 'max_angle': 160},
    'Lunges': {'min_angle': 80, 'max_angle': 160}
}

# Start video capture and pose estimation
if start:
    st.session_state.run = True
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to access the camera.")
                break

            # Recolor image to RGB for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make pose detection
            results = pose.process(image)

            # Recolor back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Variables for feedback
                wrong_form = False
                feedback_text = ''

                # Exercise detection logic based on user selection
                if exercise == 'Squats':
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Calculate angle
                    angle = calculate_angle(hip, knee, ankle)

                    # Visualize angle
                    cv2.putText(image, str(int(angle)),
                                tuple(np.multiply(knee, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    # Real-time form feedback
                    if angle < form_feedback[exercise]['min_angle']:
                        feedback_text = 'Go Lower'
                        wrong_form = True
                    elif angle > form_feedback[exercise]['max_angle']:
                        feedback_text = 'Stand Straight'
                        wrong_form = True

                    # Squat detection logic
                    if angle > 160:
                        st.session_state.stages[exercise] = "up"
                    if angle < 90 and st.session_state.stages[exercise] == 'up':
                        st.session_state.stages[exercise] = "down"
                        st.session_state.counters[exercise] += 1
                        print(f"Squat Count: {st.session_state.counters[exercise]}")

                elif exercise == 'Bicep Curls':
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Visualize angle
                    cv2.putText(image, str(int(angle)),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    # Real-time form feedback
                    if angle < form_feedback[exercise]['min_angle']:
                        feedback_text = 'Curl Up More'
                        wrong_form = True
                    elif angle > form_feedback[exercise]['max_angle']:
                        feedback_text = 'Extend Arm Fully'
                        wrong_form = True

                    # Bicep curl detection logic
                    if angle > 160:
                        st.session_state.stages[exercise] = "down"
                    if angle < 30 and st.session_state.stages[exercise] == 'down':
                        st.session_state.stages[exercise] = "up"
                        st.session_state.counters[exercise] += 1
                        print(f"Bicep Curl Count: {st.session_state.counters[exercise]}")

                # Implement similar logic for other exercises...
                # Overhead Press
                elif exercise == 'Overhead Press':
                    # Get coordinates
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # Calculate angle
                    angle = calculate_angle(elbow, shoulder, hip)

                    # Visualize angle
                    cv2.putText(image, str(int(angle)),
                                tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    # Real-time form feedback
                    if angle < form_feedback[exercise]['min_angle']:
                        feedback_text = 'Raise Arms Higher'
                        wrong_form = True
                    elif angle > form_feedback[exercise]['max_angle']:
                        feedback_text = 'Lower Arms'
                        wrong_form = True

                    # Overhead press detection logic
                    if angle < 70:
                        st.session_state.stages[exercise] = "down"
                    if angle > 160 and st.session_state.stages[exercise] == 'down':
                        st.session_state.stages[exercise] = "up"
                        st.session_state.counters[exercise] += 1
                        print(f"Overhead Press Count: {st.session_state.counters[exercise]}")

                # Tricep Extensions
                elif exercise == 'Tricep Extensions':
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Visualize angle
                    cv2.putText(image, str(int(angle)),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    # Real-time form feedback
                    if angle < form_feedback[exercise]['min_angle']:
                        feedback_text = 'Straighten Arm'
                        wrong_form = True
                    elif angle > form_feedback[exercise]['max_angle']:
                        feedback_text = 'Bend Elbow More'
                        wrong_form = True

                    # Tricep extension detection logic
                    if angle < 30:
                        st.session_state.stages[exercise] = "down"
                    if angle > 160 and st.session_state.stages[exercise] == 'down':
                        st.session_state.stages[exercise] = "up"
                        st.session_state.counters[exercise] += 1
                        print(f"Tricep Extension Count: {st.session_state.counters[exercise]}")

                # Chest Press
                elif exercise == 'Chest Press':
                    # Get coordinates
                    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate angle
                    angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)

                    # Visualize angle
                    cv2.putText(image, str(int(angle_left)),
                                tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    # Real-time form feedback
                    if angle_left < form_feedback[exercise]['min_angle']:
                        feedback_text = 'Extend Arms More'
                        wrong_form = True
                    elif angle_left > form_feedback[exercise]['max_angle']:
                        feedback_text = 'Bend Elbows'
                        wrong_form = True

                    # Chest press detection logic
                    if angle_left > 160:
                        st.session_state.stages[exercise] = "back"
                    if angle_left < 90 and st.session_state.stages[exercise] == 'back':
                        st.session_state.stages[exercise] = "forward"
                        st.session_state.counters[exercise] += 1
                        print(f"Chest Press Count: {st.session_state.counters[exercise]}")

                # Rows
                elif exercise == 'Rows':
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Visualize angle
                    cv2.putText(image, str(int(angle)),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    # Real-time form feedback
                    if angle < form_feedback[exercise]['min_angle']:
                        feedback_text = 'Pull Back More'
                        wrong_form = True
                    elif angle > form_feedback[exercise]['max_angle']:
                        feedback_text = 'Extend Arms'
                        wrong_form = True

                    # Row detection logic
                    if angle > 160:
                        st.session_state.stages[exercise] = "forward"
                    if angle < 70 and st.session_state.stages[exercise] == 'forward':
                        st.session_state.stages[exercise] = "back"
                        st.session_state.counters[exercise] += 1
                        print(f"Row Count: {st.session_state.counters[exercise]}")

                # Lunges
                elif exercise == 'Lunges':
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # Calculate angle
                    angle = calculate_angle(hip, knee, ankle)

                    # Visualize angle
                    cv2.putText(image, str(int(angle)),
                                tuple(np.multiply(knee, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    # Real-time form feedback
                    if angle < form_feedback[exercise]['min_angle']:
                        feedback_text = 'Go Lower'
                        wrong_form = True
                    elif angle > form_feedback[exercise]['max_angle']:
                        feedback_text = 'Stand Straight'
                        wrong_form = True

                    # Lunge detection logic
                    if angle > 160:
                        st.session_state.stages[exercise] = "up"
                    if angle < 90 and st.session_state.stages[exercise] == 'up':
                        st.session_state.stages[exercise] = "down"
                        st.session_state.counters[exercise] += 1
                        print(f"Lunge Count: {st.session_state.counters[exercise]}")

                # General display of the counter
                cv2.putText(image, f'{exercise}: {st.session_state.counters[exercise]}',
                            (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2,
                            cv2.LINE_AA)

                # Display real-time feedback
                if wrong_form:
                    cv2.putText(image, feedback_text,
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

                # Check if the repetition limit has been reached
                if st.session_state.counters[exercise] >= REPS_LIMIT:
                    cv2.putText(image, 'Workout Complete!',
                                (200,240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4,
                                cv2.LINE_AA)
                    st.session_state.run = False  # Stop the while loop

            except Exception as e:
                # Handle exceptions (e.g., landmarks not detected)
                print(e)
                pass

            # Render pose landmarks on the image
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # Display the image in Streamlit
            FRAME_WINDOW.image(image, channels='BGR')

        else:
            cap.release()
            st.write("Exercise session completed. Great job!")
            st.session_state.run = False
else:
    st.write("Click the checkbox to start the exercise session.")
