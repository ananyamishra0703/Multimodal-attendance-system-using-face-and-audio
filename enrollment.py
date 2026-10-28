"""
enrollment.py
Interactive webcam + microphone enrollment flow for a new student: capture
several face samples and a few voice samples, then retrain both
recognition models on the updated enrolled set.
"""

import cv2
import time
import db
import face_recognition as fr
import voice_recognition as vr

NUM_FACE_SAMPLES = 20
NUM_VOICE_SAMPLES = 3
VOICE_SAMPLE_DURATION = 3.0


def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(
            "\nCould not access the webcam.\n"
            "On macOS: System Settings -> Privacy & Security -> Camera -> "
            "make sure your Terminal app is allowed.\n"
        )
        return None
    return cap


def capture_face_samples(student_id):
    print(f"\nCapturing {NUM_FACE_SAMPLES} face samples.")
    print("Look at the camera and slowly turn your head slightly left/right/up/down")
    print("so the system learns your face from a few angles.")
    print("A green box means a face is detected. Press 'q' to cancel.\n")

    cap = open_camera()
    if cap is None:
        return False

    collected = 0
    frame_counter = 0
    window = "Enrollment - Face Capture"

    try:
        while collected < NUM_FACE_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam.")
                break

            frame_counter += 1
            box = fr.detect_largest_face(frame)
            display = frame.copy()

            if box is not None:
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)
                # Capture a sample roughly every 5 frames a face is visible,
                # rather than every single frame, so the 20 samples span a
                # few seconds of slightly different head angles instead of
                # being near-identical consecutive frames.
                if frame_counter % 5 == 0:
                    face_img = fr.crop_and_normalize_face(frame, box)
                    fr.save_enrollment_sample(student_id, collected, face_img)
                    collected += 1

            cv2.putText(
                display, f"Samples: {collected}/{NUM_FACE_SAMPLES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            cv2.imshow(window, display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Cancelled.")
                return False
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Captured {collected} face samples.")
    return collected >= NUM_FACE_SAMPLES


def capture_voice_samples(student_id):
    print(f"\nCapturing {NUM_VOICE_SAMPLES} voice samples.")
    print(f"Each recording is {VOICE_SAMPLE_DURATION:.0f} seconds. Say your name and roll")
    print("number clearly each time (saying the same thing each time is fine).\n")

    for i in range(NUM_VOICE_SAMPLES):
        input(f"Press Enter, then start speaking immediately (sample {i + 1}/{NUM_VOICE_SAMPLES})...")
        print("Recording...")
        audio = vr.record_audio(duration_seconds=VOICE_SAMPLE_DURATION)
        vr.save_voice_sample(student_id, i, audio)
        print("Saved.\n")

    return True


def enroll_new_student():
    print("\n--- Enroll New Student ---")
    name = input("Full name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    roll_no = input("Roll number: ").strip()
    if not roll_no:
        print("Roll number cannot be empty.")
        return

    if db.get_student_by_roll(roll_no):
        print(f"A student with roll number '{roll_no}' is already enrolled.")
        return

    student_id = db.add_student(name, roll_no)
    print(f"Created student record (id={student_id}).")

    face_ok = capture_face_samples(student_id)
    if not face_ok:
        print("Face enrollment incomplete. You can re-run enrollment for this student later.")

    voice_ok = capture_voice_samples(student_id)

    print("\nRetraining recognition models on all enrolled students...")
    ok1, msg1 = fr.train_model()
    print("Face model:", msg1)
    ok2, msg2 = vr.train_models()
    print("Voice model:", msg2)

    if ok1 and ok2:
        print(f"\n{name} ({roll_no}) is fully enrolled and ready for attendance.\n")
    else:
        print("\nEnrollment finished with warnings — check messages above.\n")
