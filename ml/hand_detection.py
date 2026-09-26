"""
ASL Sign Language Translator - Step 1: Real-Time Hand Detection
-----------------------------------------------------------------
Uses your webcam + MediaPipe's Hand Landmarker task to detect a
hand and draw its 21 landmark points in real time. This is the
foundation for everything else in the project.

Needs models/hand_landmarker.task downloaded first -- if it's
missing, running this will print a command you can use to get it.
"""

import cv2
from mp_setup import create_landmarker, detect, draw_landmarks

landmarker = create_landmarker(num_hands=2)
cap = cv2.VideoCapture(0)

print("Starting hand detection... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Couldn't read from the webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = detect(landmarker, rgb_frame)

    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            draw_landmarks(frame, hand_landmarks)
            wrist = hand_landmarks[0]
            cv2.putText(frame, f"Hand {i + 1} wrist: ({wrist.x:.2f}, {wrist.y:.2f})",
                        (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("ASL Translator - Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()