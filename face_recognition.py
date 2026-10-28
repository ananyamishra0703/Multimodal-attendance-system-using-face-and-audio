"""
face_recognition.py
Face detection + recognition using OpenCV's built-in Haar Cascade detector
and LBPH (Local Binary Patterns Histograms) recognizer.

Why LBPH instead of the popular `face_recognition` / dlib library: dlib
requires compiling C++ code from source on most systems (needs CMake, a
C++ toolchain) and this has a history of failing on exactly the kind of
machine this project needs to run on reliably. LBPH ships inside
opencv-contrib-python, which installs from a prebuilt wheel with no
compilation step at all, at the cost of being a classical (non-deep-
learning) algorithm — plenty accurate for a small, controlled set of
enrolled students, which is exactly this project's use case.
"""

import os
import cv2
import numpy as np

FACE_SIZE = (200, 200)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "models", "face_model.yml"
)
FACES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faces")

# Bundled directly in the project (app/resources/) rather than loaded from
# cv2.data.haarcascades — some opencv-contrib-python-headless wheel builds
# don't actually ship the cascade XML files in that directory, so relying
# on it is unreliable across platforms/versions. Shipping our own copy
# means this works regardless of what a given install includes.
_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "resources", "haarcascade_frontalface_default.xml")
_cascade = cv2.CascadeClassifier(_CASCADE_PATH)
if _cascade.empty():
    raise RuntimeError(
        f"Failed to load face detector cascade from {_CASCADE_PATH}. "
        "The file may be missing or corrupted."
    )


def detect_largest_face(frame_bgr):
    """Returns (x, y, w, h) of the largest detected face in the frame, or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # Pick the largest face box (closest to camera / most prominent).
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return tuple(faces[0])


def crop_and_normalize_face(frame_bgr, box):
    x, y, w, h = box
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face = gray[y : y + h, x : x + w]
    face = cv2.resize(face, FACE_SIZE)
    face = cv2.equalizeHist(face)  # normalize lighting
    return face


def save_enrollment_sample(student_id, sample_index, face_img):
    student_dir = os.path.join(FACES_DIR, str(student_id))
    os.makedirs(student_dir, exist_ok=True)
    path = os.path.join(student_dir, f"{sample_index}.png")
    cv2.imwrite(path, face_img)
    return path


def _load_all_samples():
    """Loads every saved face sample across all enrolled students.
    Returns (images, labels) suitable for LBPH training."""
    images, labels = [], []
    if not os.path.isdir(FACES_DIR):
        return images, labels
    for student_id_str in os.listdir(FACES_DIR):
        student_dir = os.path.join(FACES_DIR, student_id_str)
        if not os.path.isdir(student_dir):
            continue
        try:
            student_id = int(student_id_str)
        except ValueError:
            continue
        for fname in os.listdir(student_dir):
            if not fname.lower().endswith(".png"):
                continue
            img = cv2.imread(os.path.join(student_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(student_id)
    return images, labels


def train_model():
    """Retrains the LBPH recognizer on every enrolled student's saved face
    samples and writes the model to disk. Call this after enrolling a new
    student (or new samples for an existing one)."""
    images, labels = _load_all_samples()
    if len(images) == 0:
        return False, "No enrolled face samples found."
    if len(set(labels)) < 1:
        return False, "Need at least one enrolled student."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    recognizer.write(MODEL_PATH)
    return True, f"Trained on {len(images)} samples across {len(set(labels))} student(s)."


def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    return recognizer


def recognize_face(recognizer, face_img, max_distance=75.0):
    """Returns (student_id, confidence) if a confident match is found,
    else (None, distance). LBPH's predict() returns a *distance* — lower
    means more similar, so we require it to be below max_distance to
    accept the match. This threshold is tunable; 75 is a reasonable
    starting point for a well-lit webcam at close range."""
    label, distance = recognizer.predict(face_img)
    if distance <= max_distance:
        return label, distance
    return None, distance
