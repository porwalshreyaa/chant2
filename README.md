# Chant Counter

**A small, silent mantra counter that uses your webcam to count mouth openings.**

Chant Counter watches your lips—not your voice—to keep count while you focus on your practice. It shows the running count in a compact camera window and can group mouth openings into rounds for longer mantras.

## How it works

The script reads webcam frames, finds facial landmarks, and measures the gap across the inner lips relative to the width of the mouth.

On startup, it learns your baseline through a quick two-step calibration (about 12 seconds):
1. **Closed mouth:** Keep your lips closed while nodding or moving your head slightly.
2. **Chanting:** Chant normally 5–8 times.

It calculates a threshold based on your movements, counting only when the lip gap is clearly larger than anything seen with your mouth closed—so nodding or natural head motion won't trigger counts. A short cooldown helps prevent one opening from being counted more than once. The count appears in the preview and in the terminal.

This is a visual estimate, not speech recognition: it does not identify words, sounds, or whether a chant was actually spoken. Good lighting and a clear view of your face help it work reliably.

## Get started

Requires Python 3 and a working webcam.

```bash
git clone <repository-url>
cd chant2
python -m venv .venv
```

Activate the environment, then install the project dependencies:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python chant_counter.py
```

On first launch, the script downloads the MediaPipe face-landmark model (`face_landmarker.task`) into the project folder if it is not already there. Later runs reuse that file.

## Controls

Click the preview window to focus it, then use:

| Key | Action |
| --- | --- |
| `r` | Reset the count |
| `c` | Recalibrate (learn baseline numbers again) |
| `q` | Quit |

The script prints the final count when it exits.

## Customize

```bash
python chant_counter.py --per 8
```

| Option | Default | What it changes |
| --- | ---: | --- |
| `--camera` | `0` | Webcam device index; try `1` for another camera. |
| `--per` | `1` | Mouth openings per counted round. For an eight-syllable mantra, use `8`. |
| `--sensitivity` | `0.45` | Trigger point between closed and chanting (`0.0` to `1.0`). Lower values are more sensitive. |
| `--min-gap` | `0.12` | Minimum seconds between counted openings. Raise it to ignore rapid repeat detections. |
| `--closed-secs` | `5` | Duration in seconds to calibrate closed mouth. |
| `--chant-secs` | `7` | Duration in seconds to calibrate chanting. |

For example, combine options as needed:

```bash
python chant_counter.py --camera 1 --per 8 --sensitivity 0.4 --min-gap 0.2
```

## Technology

| Technology | What it does here | Why it is used |
| --- | --- | --- |
| **Python** | Runs the counter and coordinates camera input, landmark detection, and counting. | Keeps the script straightforward to run and modify. |
| **MediaPipe Tasks — Face Landmarker** | Estimates face and lip landmarks in each video frame. | Provides the lip positions needed to detect mouth openings without analyzing audio. |
| **OpenCV** (`opencv-python`) | Captures webcam frames, converts and resizes images, draws the preview and count, and reads keyboard input. | Supplies the real-time camera and simple desktop-window features. |
| **NumPy** | Computes statistical percentiles during the calibration step. | Accurately sets closed vs. open thresholds resistant to outlier movements. |

The other entries in `requirements.txt` are dependencies used by these libraries.

## Privacy

Camera frames are analyzed on your computer; this script does not record or upload video or audio. An internet connection is needed on first run only if the face-landmark model is not already present.
