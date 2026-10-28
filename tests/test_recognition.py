"""
tests/test_recognition.py

Validates the enroll -> train -> recognize pipeline for both face and
voice recognition, plus the attendance database logic, using synthetic
data (no real camera/microphone required to run this).

This does NOT test real-world accuracy against actual human faces and
voices — that can only be verified by running the live application
(`python3 app/main.py`) with a real camera/microphone, which is expected
to need some threshold tuning (see README.md, "Tuning recognition
accuracy"). What this DOES verify is that the underlying pipeline is
implemented correctly: a fresh sample of an enrolled identity is
recognized as that identity, two different enrolled identities are
correctly told apart, and unrecognized input (noise) is correctly
rejected rather than confidently misclassified.

Run with:
    python3 tests/test_recognition.py
"""

import sys
import os
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "app"))

import face_recognition as fr
import voice_recognition as vr
import db

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def reset_data():
    for sub in ["faces", "voices", "models"]:
        path = os.path.join(DATA_DIR, sub)
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def make_synthetic_face(seed, size=fr.FACE_SIZE):
    """Generates a distinct, reproducible synthetic 'face' pattern per
    student so LBPH has something consistent to learn and distinguish."""
    rng = np.random.RandomState(seed)
    base = rng.randint(80, 180, size=size, dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing="ij")
    pattern = (np.sin(xx / (3 + seed)) * 40 + np.cos(yy / (5 + seed)) * 40).astype(np.int16)
    img = np.clip(base.astype(np.int16) + pattern, 0, 255).astype(np.uint8)
    noise = rng.randint(-10, 10, size=size).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def make_synthetic_voice(seed, duration=2.0, sample_rate=vr.SAMPLE_RATE):
    """Generates a distinct synthetic 'voice' as a sum of sine waves at
    seed-dependent frequencies, mimicking how different speakers have
    different characteristic pitch/formant structure."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    base_freq = 100 + seed * 37
    signal = (
        0.5 * np.sin(2 * np.pi * base_freq * t)
        + 0.3 * np.sin(2 * np.pi * base_freq * 2.1 * t)
        + 0.2 * np.sin(2 * np.pi * base_freq * 3.3 * t)
    )
    noise = rng.normal(0, 0.02, size=signal.shape)
    return (signal + noise).astype(np.float32)


def test_face_recognition():
    print("=== Face recognition test ===")
    reset_data()

    for student_id, seed in [(1, 0), (2, 1)]:
        for i in range(15):
            img = make_synthetic_face(seed)
            fr.save_enrollment_sample(student_id, i, img)

    ok, msg = fr.train_model()
    print("train_model:", ok, msg)
    assert ok, "Face model training failed"

    recognizer = fr.load_model()
    assert recognizer is not None, "Failed to load trained face model"

    test_img_1 = make_synthetic_face(0)
    label, dist = fr.recognize_face(recognizer, test_img_1, max_distance=75.0)
    print(f"Recognize student 1 sample -> label={label}, distance={dist:.2f}")
    assert label == 1, f"Expected student 1, got {label}"

    test_img_2 = make_synthetic_face(1)
    label, dist = fr.recognize_face(recognizer, test_img_2, max_distance=75.0)
    print(f"Recognize student 2 sample -> label={label}, distance={dist:.2f}")
    assert label == 2, f"Expected student 2, got {label}"

    print("Face recognition test PASSED\n")


def test_voice_recognition():
    print("=== Voice recognition test ===")

    for student_id, seed in [(1, 0), (2, 1)]:
        for i in range(4):
            audio = make_synthetic_voice(seed, duration=2.0)
            vr.save_voice_sample(student_id, i, audio)

    ok, msg = vr.train_models()
    print("train_models:", ok, msg)
    assert ok, "Voice model training failed"

    models = vr.load_models()
    assert models is not None, "Failed to load trained voice models"

    test_audio_1 = make_synthetic_voice(0, duration=2.0)
    label, score = vr.recognize_voice(models, test_audio_1)
    print(f"Recognize student 1 voice -> label={label}, score={score}")
    assert label == 1, f"Expected student 1, got {label}"

    test_audio_2 = make_synthetic_voice(1, duration=2.0)
    label, score = vr.recognize_voice(models, test_audio_2)
    print(f"Recognize student 2 voice -> label={label}, score={score}")
    assert label == 2, f"Expected student 2, got {label}"

    rng = np.random.RandomState(99)
    noise_audio = rng.normal(0, 0.3, size=int(2.0 * vr.SAMPLE_RATE)).astype(np.float32)
    label, score = vr.recognize_voice(models, noise_audio)
    print(f"Recognize random noise -> label={label}, score={score} (expect None)")
    assert label is None, "Random noise should not be confidently matched to an enrolled student"

    print("Voice recognition test PASSED\n")


def test_db():
    print("=== Database test ===")

    if os.path.isfile(db.DB_PATH):
        os.remove(db.DB_PATH)
    db.init_db()

    sid1 = db.add_student("Test Student One", "R001")
    sid2 = db.add_student("Test Student Two", "R002")
    print("Added students:", sid1, sid2)

    rec_id, status = db.record_attendance(sid1, face_confidence=42.0, voice_confidence=-10.5, status="Present")
    print("Record attendance:", rec_id, status)
    assert status == "ok"

    rec_id2, status2 = db.record_attendance(sid1, face_confidence=40.0, voice_confidence=-9.0, status="Present")
    print("Duplicate attendance attempt (same day):", rec_id2, status2)
    assert status2 == "already_marked", "Should not allow marking the same student Present twice in one day"

    rows = db.list_attendance()
    assert len(rows) == 1
    students = db.list_students()
    assert len(students) == 2

    print("Database test PASSED\n")


if __name__ == "__main__":
    test_face_recognition()
    test_voice_recognition()
    test_db()
    print("ALL TESTS PASSED")
