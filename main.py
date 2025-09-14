import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import time
from collections import deque

# ---------------- Pygame Setup (Music Player) ----------------
pygame.mixer.init()
music_folder = os.path.join(os.path.dirname(__file__), "songfor")
playlist = [os.path.join(music_folder, f) for f in os.listdir(music_folder) if f.endswith(".mp3")]
current_song = 0

if playlist:
    pygame.mixer.music.load(playlist[current_song])
    pygame.mixer.music.play()
    pygame.mixer.music.pause()  # start paused
else:
    print("⚠️  No songs found in:", music_folder)

# ---------------- Mediapipe Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.65)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- Swipe detection params ----------------
POSITION_BUFFER_SEC = 0.6   # how many seconds to look back
SWIPE_DIST_PX = 120        # minimum horizontal pixel displacement for a swipe
SWIPE_VEL_PXPS = 300      # minimum velocity (px / sec) to qualify as swipe
SWIPE_COOLDOWN = 0.8      # seconds after a swipe to ignore new swipes

pos_buffer = deque()      # stores (timestamp, palm_x)
last_swipe_time = 0

# ---------------- Main Loop ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # mirror view
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_present = False

    if results.multi_hand_landmarks:
        hand_present = True
        for handLms in results.multi_hand_landmarks:
            # collect landmarks in pixel coords
            lm_px = []
            for lm in handLms.landmark:
                lm_px.append((int(lm.x * w), int(lm.y * h)))
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # --- palm center: average of wrist & MCPs (0,5,9,13,17) for stability ---
            ids_for_center = [0, 5, 9, 13, 17]
            palm_x = int(np.mean([lm_px[i][0] for i in ids_for_center]))
            palm_y = int(np.mean([lm_px[i][1] for i in ids_for_center]))

            # draw palm center
            cv2.circle(img, (palm_x, palm_y), 8, (0, 255, 255), cv2.FILLED)

            # append current palm x with timestamp and remove old samples
            now = time.time()
            pos_buffer.append((now, palm_x))
            # purge older than window
            while pos_buffer and now - pos_buffer[0][0] > POSITION_BUFFER_SEC:
                pos_buffer.popleft()

            # --- compute displacement & velocity over the buffer ---
            if len(pos_buffer) >= 2:
                t0, x0 = pos_buffer[0]
                tn, xn = pos_buffer[-1]
                dx = xn - x0
                dt = tn - t0 if (tn - t0) > 0 else 0.0001
                vel = dx / dt

                # debug overlay (optional)
                cv2.putText(img, f"dx={dx:.0f} vel={vel:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                # swipe detection (require both distance and velocity)
                if (abs(dx) > SWIPE_DIST_PX and abs(vel) > SWIPE_VEL_PXPS
                        and now - last_swipe_time > SWIPE_COOLDOWN and playlist):
                    if dx > 0:
                        # swipe RIGHT -> Next
                        current_song = (current_song + 1) % len(playlist)
                        pygame.mixer.music.load(playlist[current_song])
                        pygame.mixer.music.play()
                        cv2.putText(img, "Next ▶", (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 3)
                    else:
                        # swipe LEFT -> Previous
                        current_song = (current_song - 1) % len(playlist)
                        pygame.mixer.music.load(playlist[current_song])
                        pygame.mixer.music.play()
                        cv2.putText(img, "◀ Previous", (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 3)

                    last_swipe_time = now
                    pos_buffer.clear()  # avoid multi-detection for same motion

            # --- Play / Pause detection using number of fingers up ---
            # use tip vs pip y to detect fingers up
            tip_ids = [4, 8, 12, 16, 20]
            fingers = []
            # thumb: check x relation (because thumb folds sideways)
            thumb_is_open = lm_px[4][0] > lm_px[3][0]  # works for mirrored view
            fingers.append(1 if thumb_is_open else 0)
            for tid in [8, 12, 16, 20]:
                fingers.append(1 if lm_px[tid][1] < lm_px[tid - 2][1] else 0)

            # only accept play/pause if cooldown passed
            if time.time() - last_swipe_time > SWIPE_COOLDOWN:
                if sum(fingers) == 0:  # fist -> pause
                    pygame.mixer.music.pause()
                    cv2.putText(img, "⏸ Pause", (50, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    last_swipe_time = time.time()
                elif sum(fingers) == 5:  # open palm -> play
                    pygame.mixer.music.unpause()
                    cv2.putText(img, "▶ Play", (50, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    last_swipe_time = time.time()

            # --- Volume control by vertical hand position ---
            # top of frame -> louder, bottom -> quieter
            vol = 1.0 - (palm_y / h)
            vol = float(np.clip(vol, 0.0, 1.0))
            pygame.mixer.music.set_volume(vol)
            cv2.putText(img, f"Vol: {int(vol*100)}%", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    else:
        # no hand detected -> clear buffer so stale positions don't affect next swipe
        pos_buffer.clear()

    cv2.imshow("AI DJ - Gesture Music Player (improved swipe)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
