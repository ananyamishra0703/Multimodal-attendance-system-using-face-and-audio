"""
attendance.py
Live attendance session: shows a webcam preview with face recognition
running continuously, and on demand records a short voice sample to
confirm identity. A student is marked Present only when BOTH modalities
agree on who they are — decision-level fusion of the two biometric
signals, which is the actual "multimodal" part of this system (either
signal alone would be weaker evidence; requiring both to agree makes a
false accept considerably less likely than either used alone).
"""

import cv2
import db
import face_recognition as fr
import voice_recognition as vr

FACE_MAX_DISTANCE = 75.0
VOICE_MIN_MARGIN = 5.0
VOICE_MIN_ABSOLUTE_SCORE = -60.0
VOICE_SAMPLE_DURATION = 3.0


def run_attendance_session():
    print("\n--- Take Attendance ---")

    face_recognizer = fr.load_model()
    voice_models = vr.load_models()

    if face_recognizer is None or not voice_models:
        print("No trained models found yet. Enroll at least one student first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(
            "\nCould not access the webcam.\n"
            "On macOS: System Settings -> Privacy & Security -> Camera -> "
            "make sure your Terminal app is allowed.\n"
        )
        return

    print("\nLive camera preview started.")
    print("Press 'c' when your face is boxed in green to check in.")
    print("Press 'q' to end the attendance session.\n")

    window = "Attendance - Live"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam.")
                break

            box = fr.detect_largest_face(frame)
            display = frame.copy()

            if box is not None:
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)

            cv2.putText(
                display, "Press 'c' to check in, 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                _attempt_check_in(frame, face_recognizer, voice_models)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Attendance session ended.\n")


def _attempt_check_in(frame, face_recognizer, voice_models):
    box = fr.detect_largest_face(frame)
    if box is None:
        print(">> No face detected — position yourself in frame and try again.\n")
        return

    face_img = fr.crop_and_normalize_face(frame, box)
    face_label, face_distance = fr.recognize_face(face_recognizer, face_img, max_distance=FACE_MAX_DISTANCE)

    if face_label is None:
        print(f">> Face not recognized (distance={face_distance:.1f}). Not enrolled, or lighting/angle is poor.\n")
        return

    student = db.get_student(face_label)
    student_name = student["name"] if student else f"id={face_label}"
    print(f">> Face matched: {student_name} (distance={face_distance:.1f})")
    print(">> Now say your name and roll number out loud...")

    audio = vr.record_audio(duration_seconds=VOICE_SAMPLE_DURATION)
    voice_label, voice_score = vr.recognize_voice(
        voice_models, audio, min_margin=VOICE_MIN_MARGIN, min_absolute_score=VOICE_MIN_ABSOLUTE_SCORE
    )

    score_str = f"{voice_score:.1f}" if voice_score is not None else "n/a"

    if voice_label is None:
        print(f">> Voice not confidently recognized (score={score_str}).")
        db.record_attendance(face_label, face_distance, voice_score, status="Unverified")
        print(">> Logged as Unverified — face matched but voice did not confirm.\n")
        return

    if voice_label != face_label:
        voice_student = db.get_student(voice_label)
        voice_name = voice_student["name"] if voice_student else f"id={voice_label}"
        print(f">> Mismatch: face says {student_name}, voice says {voice_name}.")
        db.record_attendance(face_label, face_distance, voice_score, status="Unverified")
        print(">> Logged as Unverified due to modality disagreement.\n")
        return

    record_id, status = db.record_attendance(face_label, face_distance, voice_score, status="Present")
    if status == "already_marked":
        print(f">> {student_name} is already marked Present today.\n")
    else:
        print(f">> CONFIRMED: {student_name} marked Present (face + voice both matched).\n")
