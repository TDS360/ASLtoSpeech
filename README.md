# ASL Sign Language Translator

Real-time American Sign Language (ASL) fingerspelling translator
using a webcam, Python, OpenCV, and MediaPipe.

## Setup
1. Install Python 3.8+
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the hand landmark model (required -- the scripts won't
   run without it). From the project root folder, run:
   ```
   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models\hand_landmarker.task"
   ```
   (The `models` folder is created automatically if it doesn't exist yet.)
4. Confirm your webcam + MediaPipe setup works:
   ```
   cd src
   python hand_detection.py
   ```
5. Collect your own training data:
   ```
   python collect_data.py
   ```

## Project structure
- `src/hand_detection.py` — Step 1: real-time hand tracking
- `src/mp_setup.py` — shared MediaPipe detector setup + drawing helper
- `src/normalize.py` — shared landmark normalization function
- `src/collect_data.py` — Step 2: collects labeled hand-pose samples
  into `data/landmarks.csv`
- `src/train_model.py` — Step 3: trains a classifier (coming soon)
- `src/translator.py` — Step 4: real-time letter + word recognition
  app (coming soon)
- `docs/research_notes.md` — background research for the board
- `docs/project_log.md` — running build log

## Current status
- [x] Real-time hand detection working
- [ ] Data collected for all 24 static letters
- [ ] Classifier trained
- [x] Live translator built (needs real data + training to test)
- [ ] (Stretch) Raspberry Pi standalone version