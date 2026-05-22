import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox


from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ML
from sklearn.tree import DecisionTreeClassifier

# ML MODEL
X = [
    [10, 10, 0],
    [7, 10, 3],
    [9, 6, 3]
]
y = ["Normal", "Ptosis", "Asymmetry"]

model = DecisionTreeClassifier()
model.fit(X, y)

#  CONSTANTS 
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [159, 145]
RIGHT_EYE = [386, 374]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

IRIS_DIAMETER_MM = 11.7

blink_count = 0
prev_height = None
heights_buffer = []

def generate_pdf(lh, rh, result):
    doc = SimpleDocTemplate("eye_report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Eye Analysis Report", styles['Title']))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Left Eye: {lh:.2f} mm", styles['Normal']))
    content.append(Paragraph(f"Right Eye: {rh:.2f} mm", styles['Normal']))
    content.append(Paragraph(f"Prediction: {result}", styles['Normal']))

    doc.build(content)


def euclidean_dist(p1, p2, shape):
    h, w = shape[:2]
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return np.linalg.norm([x2 - x1, y2 - y1]), (x1, y1), (x2, y2)

def iris_diameter(landmarks, indices, shape):
    pts = [landmarks[i] for i in indices]
    max_dist = 0
    best = []

    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d, p1, p2 = euclidean_dist(pts[i], pts[j], shape)
            if d > max_dist:
                max_dist = d
                best = [p1, p2]

    center = ((best[0][0]+best[1][0])//2, (best[0][1]+best[1][1])//2)
    return max_dist, best, center

def palpebral_height(landmarks, eye, iris, shape):
    top = landmarks[eye[0]]
    bottom = landmarks[eye[1]]

    h_px, t_px, b_px = euclidean_dist(top, bottom, shape)
    iris_px, iris_pts, iris_center = iris_diameter(landmarks, iris, shape)

    if iris_px == 0:
        return None, None, None, None, None

    mm_per_px = IRIS_DIAMETER_MM / iris_px
    return h_px * mm_per_px, t_px, b_px, iris_pts, iris_center


def analyze_frame(img):
    global blink_count, prev_height, heights_buffer

    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return img

        lm = res.multi_face_landmarks[0].landmark

        lh, lt, lb, lir, lic = palpebral_height(lm, LEFT_EYE, LEFT_IRIS, img.shape)
        rh, rt, rb, rir, ric = palpebral_height(lm, RIGHT_EYE, RIGHT_IRIS, img.shape)

        if lh is None or rh is None:
            return img

        # ====== SMOOTHING ======
        heights_buffer.append((lh, rh))
        if len(heights_buffer) > 5:
            heights_buffer.pop(0)

        lh = np.mean([h[0] for h in heights_buffer])
        rh = np.mean([h[1] for h in heights_buffer])

        #  ML PREDICTION 
        result = model.predict([[lh, rh, abs(lh-rh)]])[0]

        # BLINK 
        avg = (lh + rh) / 2
        if prev_height and avg < prev_height * 0.6:
            blink_count += 1
        prev_height = avg

        # AUTO CAPTURE 
        if abs(lh - rh) < 1:
            cv2.putText(img, "Auto Capture!", (200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imwrite("captured.png", img)
            generate_pdf(lh, rh, result)

        #DISPLAY
        cv2.putText(img, f"L: {lh:.2f} mm", (30, 40), 0, 0.7, (0,255,255), 2)
        cv2.putText(img, f"R: {rh:.2f} mm", (30, 70), 0, 0.7, (0,255,255), 2)

        cv2.putText(img, f"Prediction: {result}", (30, 110),
                    0, 0.7, (0,255,0), 2)

        cv2.putText(img, f"Blinks: {blink_count}", (30, 140),
                    0, 0.7, (255,255,0), 2)

        return img

#  IMAGE 
def select_image():
    path = filedialog.askopenfilename()
    img = cv2.imread(path)

    img = cv2.flip(img, 1)
    out = analyze_frame(img)

    cv2.imshow("Result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#WEBCAM 
def webcam_mode():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        out = analyze_frame(frame)

        cv2.imshow("Live Analyzer", out)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# BLINK MODE
def blink_mode():
    global blink_count
    blink_count = 0

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        out = analyze_frame(frame)

        cv2.putText(out, f"Blink Mode Count: {blink_count}",
                    (30, 200), 0, 0.7, (0,255,255), 2)

        cv2.imshow("Blink Mode", out)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


root = tk.Tk()
root.title("AI Eye Analyzer")
root.geometry("400x300")

tk.Label(root, text="Advanced Eye Analyzer", font=("Arial", 14)).pack(pady=20)

tk.Button(root, text="📁 Image Analysis", command=select_image).pack(pady=10)
tk.Button(root, text="🎥 Live Detection", command=webcam_mode).pack(pady=10)
tk.Button(root, text="👁 Blink Mode", command=blink_mode).pack(pady=10)
tk.Button(root, text="❌ Exit", command=root.destroy).pack(pady=10)

root.mainloop()