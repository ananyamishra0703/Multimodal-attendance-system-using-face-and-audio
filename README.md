# Multimodal Attendance System (Face + Voice)

**Developed by:** Ananya Mishra (24BAI1558)
**Course:** BCSE301P – Software Engineering Lab, VIT Chennai
**Faculty:** Dr. Elakya R

A desktop application that marks student attendance by recognizing both
their **face** (webcam) and **voice** (microphone), and only accepts the
match when both agree — genuine multimodal biometric identification, not
just a camera that snaps a photo.

## Why two modalities, and why this counts as "multimodal"

A system that only checks one biometric signal (face *or* voice) can be
fooled more easily, and is more sensitive to that one signal being poor
quality (bad lighting, background noise). This system requires **both**
to independently agree on the same identity before marking someone
Present — a basic form of *decision-level fusion*, a real concept from
multimodal biometrics. If only the face matches, or only the voice
matches, or they disagree, the attempt is logged as **Unverified**
instead of silently accepted.

## How recognition works

**Face recognition** — Haar Cascade face *detection* (finds where a face
is in the frame) followed by LBPH (Local Binary Patterns Histograms) face
*recognition* (identifies whose face it is, based on texture patterns
learned from several enrolled photos per student). This is a classical,
well-understood computer vision technique — not a deep neural network —
chosen deliberately: it ships inside `opencv-contrib-python` with
**zero compilation required**, unlike the commonly recommended
`face_recognition`/`dlib` library, which needs a C++ build toolchain and
often fails to install.

**Voice recognition** — MFCC (Mel-Frequency Cepstral Coefficients) feature
extraction followed by a Gaussian Mixture Model per enrolled student. This
is the classical statistical approach to speaker identification (the
same family of technique behind production GMM-UBM speaker-ID systems
before deep neural speaker embeddings became standard). Chosen for the
same reason as above: `librosa` + `scikit-learn` install from prebuilt
wheels with no compilation, unlike most modern neural speaker-embedding
libraries.

**Fusion** — a check-in is only marked **Present** if the face match and
the voice match identify the *same* enrolled student. Any disagreement,
or either modality failing to confidently recognize anyone, is logged as
**Unverified** so there's still a record of the attempt.

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.9+ |
| Face detection/recognition | OpenCV (Haar Cascade + LBPH) |
| Voice feature extraction | librosa (MFCC) |
| Voice classification | scikit-learn (Gaussian Mixture Model) |
| Audio I/O | sounddevice + soundfile |
| Storage | SQLite (Python's built-in `sqlite3` module) |
| Interface | Command-line menu + live OpenCV camera window |

Every dependency installs from a prebuilt wheel — there is no C++
compiler, CMake, or native build step involved anywhere in this project.

## Getting started

```bash
# clone the repository
git clone https://github.com/ananyamishra0703/multimodal-attendance-system.git
cd multimodal-attendance-system

# set up a virtual environment
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

# install dependencies and run
pip install -r requirements.txt
python3 app/main.py
```

The first run creates a `data/` folder (SQLite database, enrolled face
photos, voice recordings, trained model files) and an `exports/` folder
automatically — nothing to set up by hand.

### macOS camera & microphone permissions

The **first time** you run this, macOS may block camera/microphone
access silently rather than showing a permission prompt, because it's a
Terminal process asking, not an app with its own icon. If the camera
window never opens, or recording produces silence:

1. Open **System Settings → Privacy & Security → Camera** — make sure
   **Terminal** (or whichever app you're running Python from — iTerm,
   VS Code, etc.) is toggled **on**.
2. Do the same under **Privacy & Security → Microphone**.
3. If you don't see your terminal app listed yet, run the program once
   first (it may still fail) — macOS usually adds the app to that list
   after the first access attempt, then you can toggle it on and re-run.

## Using the system

Run `python3 app/main.py` and you'll get a menu:

```
1. Enroll new student
2. Take attendance
3. View attendance log
4. Export attendance to CSV
5. List enrolled students
6. Exit
```

**1. Enroll new student** — enter a name and roll number, then:
- A camera window opens. Look at it and slowly turn your head slightly
  left/right/up/down — the system captures 20 face samples automatically
  as it detects your face from different angles (takes a few seconds).
  Press `q` to cancel if needed.
- You'll then be asked to record your voice 3 times (3 seconds each) —
  say your name and roll number clearly each time.
- The system retrains both recognition models on all enrolled students
  automatically — this takes a few seconds.

**2. Take attendance** — opens a live camera window.
- When your face is boxed in **green**, press `c` to check in.
- The system identifies your face, then asks you to speak — say your
  name and roll number again.
- If both the face and voice match the same enrolled student, you're
  marked **Present**. If not, it's logged as **Unverified**.
- Press `q` to end the session.

**3-5.** View the attendance log, export it to CSV (saved in `exports/`),
or list everyone currently enrolled.

## Project structure

```
multimodal-attendance-system/
├── README.md
├── requirements.txt
├── app/
│   ├── main.py                 # CLI menu / entry point
│   ├── db.py                   # SQLite: students & attendance records
│   ├── face_recognition.py     # Haar Cascade detection + LBPH recognition
│   ├── voice_recognition.py    # MFCC extraction + GMM speaker models
│   ├── enrollment.py           # Enrollment flow (capture + train)
│   ├── attendance.py           # Live attendance session + fusion logic
│   └── resources/
│       └── haarcascade_frontalface_default.xml
├── testing/
│   ├── ananya_test.py          # Selenium WebDriver functional tests
│   └── AttendanceLoadTest.jmx  # Apache JMeter load test plan
├── diagrams/                   # StarUML models (ER, DFD, UML diagrams)
└── data/                       # created automatically on first run
    ├── attendance.db
    ├── faces/<student_id>/     # enrolled face photos
    ├── voices/<student_id>/    # enrolled voice recordings
    └── models/                 # trained LBPH + GMM model files
```

## Demo flow

1. **Enroll 2-3 students** — shows the live face capture and voice
   recording in action.
2. **Take attendance** for one of them — the green detection box appears,
   press `c`, speak when prompted, and the system confirms "Present."
3. **Rejection case** — someone *not enrolled* tries to check in, or an
   enrolled person's face matches but someone else speaks — the attempt
   is logged as "Unverified" instead of being wrongly accepted. This
   demonstrates that the fusion logic genuinely verifies both modalities.
4. **View the attendance log** and **export to CSV** to show the audit
   trail.

## Tuning recognition accuracy

Real faces and voices vary with lighting, camera, microphone, and room
noise, so two thresholds can be adjusted during deployment if matches
are too strict or too lenient:

- `FACE_MAX_DISTANCE` in `app/attendance.py` (default `75.0`) — LBPH
  returns a *distance*, so **lower is a better match**. Raise this number
  if correct faces are being rejected; lower it if wrong faces are being
  accepted.
- `VOICE_MIN_MARGIN` / `VOICE_MIN_ABSOLUTE_SCORE` in `app/attendance.py`
  — controls how confidently the voice match must "win" against other
  enrolled students and against outright noise. Loosen
  `VOICE_MIN_ABSOLUTE_SCORE` (make it more negative) if valid voices are
  being rejected as unrecognized.

Good lighting (facing a light source, not backlit) and a quiet room make
a large practical difference to both modalities.

## Testing

**Recognition pipeline validation.** The core recognition logic
(enrollment → training → recognition, for both face and voice) was first
validated with synthetic data — generated face-like image patterns and
multi-tone audio signals, each with a distinct, reproducible identity —
and then tested with real enrolled users. The tests confirm that the
system:
- Correctly recognizes a fresh sample from an enrolled identity.
- Correctly distinguishes between two different enrolled identities.
- Correctly rejects random noise rather than misidentifying it as an
  enrolled student.

**Testing tools** (files in the `testing/` folder):

| Tool | Type | File |
|---|---|---|
| Selenium WebDriver (Python) | Functional / black-box UI testing | `testing/ananya_test.py` |
| Apache JMeter 5.6.3 | Performance / load testing | `testing/AttendanceLoadTest.jmx` |

To run the Selenium tests:
```bash
pip install selenium
python3 testing/ananya_test.py
```

To run the JMeter load test, open `AttendanceLoadTest.jmx` in Apache
JMeter and click **Start**, or run it from the command line:
```bash
jmeter -n -t testing/AttendanceLoadTest.jmx -l results.jtl
```

## Limitations and future enhancements

Current scope: single enrolled class at a time (no multi-class/section
support), desktop CLI + OpenCV window rather than a web interface, no
anti-spoofing (a photo held up to the camera could fool face detection —
a known limitation of camera-based face recognition generally), and
recognition accuracy depends on reasonable lighting and a quiet room for
voice capture.

Planned enhancements:
- **Liveness detection** (e.g., blink detection) to prevent photo spoofing
- **Web/kiosk interface** with role-based access for faculty and admins
- **Deep-learning embeddings** (FaceNet-style face and speaker embeddings)
  for higher accuracy
- **Multi-class support** and a **PostgreSQL** backend
- **Cloud-synced attendance log** and analytics dashboard
