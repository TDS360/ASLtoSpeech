
## Why landmarks instead of raw images
- Several existing student/hobby ASL projects extract 21 hand
  landmarks (63 numbers: x, y, z per point) with MediaPipe, then
  train a lightweight classifier (Random Forest, SVM, or KNN)
  instead of a heavy image-based CNN.
- This is much faster to train, needs far less data, and runs well
  on lower-powered hardware like a Raspberry Pi.
- Sources: DEV Community ASL/MediaPipe tutorial; asl-landmark-
  classifier (GitHub); SignBridge-ML (GitHub).

## Why J and Z are excluded (for now)
- J and Z involve hand motion, not a single held pose, so they don't
  fit a single-frame landmark classifier.
- Similar projects handle only the 24 static letters (A-I, K-Y) for
  this reason, treating J/Z as a later addition that tracks motion
  across several frames instead of one snapshot.
- Source: SignBridge-ML (GitHub).

## Known hard letters — plan extra data around these
- A, S, and O look very similar (closed-fist shapes) and are
  frequently confused with each other.
- M and N are also commonly confused, especially with less careful
  signing.
- Plan: collect extra samples for A/S/O and M/N, and consider a
  "top 2 guesses" display instead of forcing one answer.
- Sources: ACM comparison of sign language alphabet recognition
  techniques; robotic ASL hand accuracy study (arXiv).

## Lighting and camera conditions
- One MediaPipe study found accuracy dropped from 85.9% (good
  lighting, clean background) to 71.6% (poor lighting, cluttered
  background).
- Bright, even lighting significantly improves detection accuracy;
  poor lighting causes jitter and false detections.
- Plan: collect training data and run the live demo in the same
  well-lit spot.
- Sources: MediaPipe Hands on-device tracking study; "5 Things I
  Wish I Knew Before Using MediaPipe" (DEV Community).

## Why we normalize landmarks
- Extracting landmarks already avoids most background/lighting
  problems compared to raw pixels, but hand position and distance
  from the camera are a separate issue landmark extraction alone
  doesn't solve.
- Our approach: subtract the wrist position from every landmark
  (removes position dependence), then divide by the wrist-to-
  middle-finger-base distance (removes distance/scale dependence).
- Source: "Improving Hand Pose Recognition using Localization and
  Zoom Normalizations over MediaPipe Landmarks" (ASTOUND project).

## Classifier choice
- Random Forest, SVM, and KNN are the most common choices for this
  kind of landmark classification in similar projects — worth
  training more than one and comparing accuracy on our own data.
- Sources: asl-landmark-classifier (GitHub); DEV Community
  ASL/MediaPipe tutorial.

## Data collection plan
- Aim for 100+ samples per letter, collected from our own webcam
  under consistent lighting.
- Extra samples for the known-hard letters: A, S, O, M, N.

## Why we detect 2 hands but only classify 1
- ASL fingerspelling is deliberately a one-handed system -- this is
  different from British Sign Language (BSL) and Auslan, which use
  a two-handed manual alphabet. ASL was designed so a signer's other
  hand stays free.
- Because of this, our translator detects up to 2 hands on screen
  (so it doesn't get confused if a second hand appears), but always
  runs letter recognition on just the first detected hand -- matching
  how the language actually works, not a limitation we settled for.
- Source: British Sign Language fingerspelling alphabet guide;
  ASL Bloom (ASL vs. BSL comparison).