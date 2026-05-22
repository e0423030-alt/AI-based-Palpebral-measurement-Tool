import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

# ====== CONSTANTS ======
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [159, 145]
RIGHT_EYE = [386, 374]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

IRIS_DIAMETER_MM = 11.7
TOLERANCE_MM = 0.3

blink_count = 0
prev_height = None

# ====== FUNCTIONS ======
def euclidean_dist(p1, p2, image_shape):
    h, w = image_shape[:2]
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return np.linalg.norm([x2 - x1, y2 - y1]), (x1, y1), (x2, y2)

def iris_diameter(landmarks, iris_indices, image_shape):
    iris_points = [landmarks[i] for i in iris_indices]
    max_dist = 0
    points_px = []

    for i in range(len(iris_points)):
        for j in range(i + 1, len(iris_points)):
            dist, pt1, pt2 = euclidean_dist(iris_points[i], iris_points[j], image_shape)
            if dist > max_dist:
                max_dist = dist
                points_px = [pt1, pt2]

    center = ((points_px[0][0] + points_px[1][0]) // 2,
              (points_px[0][1] + points_px[1][1]) // 2)

    return max_dist, points_px, center

def palpebral_height(landmarks, eye_indices, iris_indices, image_shape):
    top = landmarks[eye_indices[0]]
    bottom = landmarks[eye_indices[1]]

    height_px, top_px, bottom_px = euclidean_dist(top, bottom, image_shape)
    iris_px, iris_pts, iris_center = iris_diameter(landmarks, iris_indices, image_shape)

    if iris_px == 0:
        return None, None, None, None, None

    mm_per_px = IRIS_DIAMETER_MM / iris_px
    height_mm = height_px * mm_per_px

    return height_mm, top_px, bottom_px, iris_pts, iris_center

def draw_annotations(image, eye_name, height_mm, top_px, bottom_px, iris_pts, iris_center, y):
    center_x = int((top_px[0] + bottom_px[0]) / 2)

    cv2.line(image, (center_x, top_px[1]), (center_x, bottom_px[1]), (0, 255, 255), 2)

    if iris_pts:
        cv2.line(image, iris_pts[0], iris_pts[1], (255, 0, 0), 2)

    text = f"{eye_name}: {height_mm:.2f} mm"
    cv2.putText(image, text, (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if iris_center:
        label = 'L' if eye_name == "Left" else 'R'
        cv2.putText(image, label,
                    (iris_center[0]-10, iris_center[1]+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def analyze_frame(image):
    global blink_count, prev_height

    with mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True) as face_mesh:

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return image

        landmarks = results.multi_face_landmarks[0].landmark

        lh, lt, lb, lir, lic = palpebral_height(
            landmarks, LEFT_EYE, LEFT_IRIS, image.shape)

        rh, rt, rb, rir, ric = palpebral_height(
            landmarks, RIGHT_EYE, RIGHT_IRIS, image.shape)

        if lh is None or rh is None:
            return image

        # Swap labels (mirror fix)
        draw_annotations(image, "Right", lh, lt, lb, lir, lic, 40)
        draw_annotations(image, "Left", rh, rt, rb, rir, ric, 80)

        # ====== PTOSIS DETECTION ======
        left_status = "Normal"
        right_status = "Normal"

        if lh < 8:
            left_status = "Ptosis"
        if rh < 8:
            right_status = "Ptosis"

        # ====== ASYMMETRY ======
        if abs(lh - rh) > 1:
            symmetry = "Asymmetry Detected"
        else:
            symmetry = "Symmetric"

        # ====== STRABISMUS ======
        if lic and ric:
            if abs(lic[0] - ric[0]) > 20:
                alignment = "Misalignment"
            else:
                alignment = "Aligned"
        else:
            alignment = "Unknown"

        # ====== BLINK DETECTION ======
        avg_height = (lh + rh) / 2
        if prev_height is not None:
            if avg_height < prev_height * 0.6:
                blink_count += 1
        prev_height = avg_height

        # ====== DISPLAY ======
        cv2.putText(image, f"Left Eye: {left_status}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(image, f"Right Eye: {right_status}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(image, f"Symmetry: {symmetry}", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(image, f"Alignment: {alignment}", (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.putText(image, f"Blinks: {blink_count}", (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return image

# ====== IMAGE INPUT ======
def select_image():
    filepath = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.png *.jpeg")])

    if filepath:
        image = cv2.imread(filepath)

        if image is None:
            messagebox.showerror("Error", "Cannot load image")
            return

        image = cv2.flip(image, 1)

        result = analyze_frame(image)

        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ====== WEBCAM ======
def start_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not working")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        result = analyze_frame(frame)

        cv2.imshow("Advanced Eye Analyzer", result)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ====== GUI ======
root = tk.Tk()
root.title("Advanced Eye Analyzer")
root.geometry("400x250")

tk.Label(root, text="Select Input Mode", font=("Arial", 14)).pack(pady=20)

tk.Button(root, text="📁 Analyze Image", command=select_image).pack(pady=10)
tk.Button(root, text="🎥 Live Webcam", command=start_webcam).pack(pady=10)
tk.Button(root, text="❌ Exit", command=root.destroy).pack(pady=10)

root.mainloop()