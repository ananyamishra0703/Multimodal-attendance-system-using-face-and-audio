"""
voice_recognition.py
Speaker identification using MFCC (Mel-Frequency Cepstral Coefficient)
features and a per-student Gaussian Mixture Model — the classical
statistical approach to speaker recognition (the same family of technique
behind GMM-UBM systems used in production speaker-ID before deep neural
speaker embeddings became standard). Chosen here specifically because it
needs only numpy/scipy/scikit-learn/librosa, all of which install from
prebuilt wheels with no compilation — unlike most modern neural speaker-
embedding libraries, which typically pull in a full deep learning
framework.

How it works:
  Enrollment: record a few seconds of a student's voice, extract MFCC
  frames (a short-time spectral fingerprint of the voice), and fit a
  small Gaussian Mixture Model to the distribution of those frames. This
  model captures the characteristic "shape" of that person's voice.

  Recognition: record a new sample, extract its MFCC frames, and score
  them against every enrolled student's GMM (log-likelihood). The
  student whose model scores highest is the predicted speaker.
"""

import os
import pickle
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from sklearn.mixture import GaussianMixture

SAMPLE_RATE = 16000
N_MFCC = 13
GMM_COMPONENTS = 8

VOICES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "voices")
MODELS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "models", "voice_models.pkl"
)


def record_audio(duration_seconds=3.0, sample_rate=SAMPLE_RATE):
    """Records from the default microphone and returns a mono float32
    numpy array. Blocks until recording finishes."""
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def save_voice_sample(student_id, sample_index, audio, sample_rate=SAMPLE_RATE):
    student_dir = os.path.join(VOICES_DIR, str(student_id))
    os.makedirs(student_dir, exist_ok=True)
    path = os.path.join(student_dir, f"{sample_index}.wav")
    sf.write(path, audio, sample_rate)
    return path


def extract_mfcc(audio, sample_rate=SAMPLE_RATE):
    """Extracts MFCC frames from an audio signal, trimming silence first
    so enrollment/recognition isn't thrown off by leading/trailing quiet."""
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio_trimmed) < sample_rate * 0.3:
        # Too short after trimming (e.g. near-silent recording) — fall
        # back to the untrimmed audio so we still extract *something*.
        audio_trimmed = audio

    mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=sample_rate, n_mfcc=N_MFCC)
    # Transpose so each row is one time frame's MFCC vector — this is
    # the (n_frames, n_mfcc) shape scikit-learn's GMM expects.
    return mfcc.T


def _load_all_mfccs():
    """Loads every saved voice sample across all enrolled students and
    extracts their MFCC frames. Returns {student_id: mfcc_frames}."""
    per_student = {}
    if not os.path.isdir(VOICES_DIR):
        return per_student
    for student_id_str in os.listdir(VOICES_DIR):
        student_dir = os.path.join(VOICES_DIR, student_id_str)
        if not os.path.isdir(student_dir):
            continue
        try:
            student_id = int(student_id_str)
        except ValueError:
            continue
        all_frames = []
        for fname in os.listdir(student_dir):
            if not fname.lower().endswith(".wav"):
                continue
            audio, sr = sf.read(os.path.join(student_dir, fname))
            audio = audio.astype(np.float32)
            frames = extract_mfcc(audio, sr)
            all_frames.append(frames)
        if all_frames:
            per_student[student_id] = np.vstack(all_frames)
    return per_student


def train_models():
    """Fits one GaussianMixture model per enrolled student on their MFCC
    frames and writes all models to a single pickle file."""
    per_student = _load_all_mfccs()
    if not per_student:
        return False, "No enrolled voice samples found."

    models = {}
    for student_id, frames in per_student.items():
        n_components = min(GMM_COMPONENTS, max(1, len(frames) // 10))
        gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=42)
        gmm.fit(frames)
        models[student_id] = gmm

    os.makedirs(os.path.dirname(MODELS_PATH), exist_ok=True)
    with open(MODELS_PATH, "wb") as f:
        pickle.dump(models, f)
    return True, f"Trained voice models for {len(models)} student(s)."


def load_models():
    if not os.path.isfile(MODELS_PATH):
        return None
    with open(MODELS_PATH, "rb") as f:
        return pickle.load(f)


def recognize_voice(models, audio, sample_rate=SAMPLE_RATE, min_margin=5.0, min_absolute_score=-60.0):
    """Scores the recorded audio against every enrolled student's GMM and
    returns (student_id, best_score) if the match is confident, else
    (None, best_score).

    Two separate checks guard against a false match, and both matter:
      1. min_absolute_score — the best score itself must not be too low.
         A relative margin alone isn't enough: pure noise or silence
         scores badly against *every* enrolled model, but one model can
         still score "least badly" and win on margin alone. This floor
         catches that case by rejecting outright-poor fits regardless of
         how the candidates compare to each other.
      2. min_margin — the best score must be clearly ahead of the
         second-best, so two enrolled voices that are genuinely similar
         don't produce an overconfident, coin-flip match.

    The exact threshold values are dataset-dependent in real speaker
    recognition systems; these are reasonable starting points for short
    (2-4s) enrollment/test clips and should be tuned against real
    recordings during the demo rehearsal.
    """
    if not models:
        return None, None

    frames = extract_mfcc(audio, sample_rate)
    scores = {}
    for student_id, gmm in models.items():
        scores[student_id] = gmm.score(frames)  # average log-likelihood per frame

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best_score = ranked[0]

    if best_score < min_absolute_score:
        return None, best_score

    if len(ranked) == 1:
        return best_id, best_score

    second_score = ranked[1][1]
    if (best_score - second_score) >= min_margin:
        return best_id, best_score
    return None, best_score
