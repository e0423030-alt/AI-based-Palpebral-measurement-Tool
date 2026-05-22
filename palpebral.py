import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Locate client_secrets.json properly
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

client_secrets_path = os.path.join(base_path, "client_secrets.json")

gauth = GoogleAuth()
gauth.LoadClientConfigFile(client_secrets_path)
# ====== GOOGLE DRIVE SETUP ======
gauth.LoadCredentialsFile("mycreds.txt")

if gauth.credentials is None:
    gauth.LocalWebserverAuth()
    gauth.SaveCredentialsFile("mycreds.txt")
elif gauth.access_token_expired:
    gauth.Refresh()
    gauth.SaveCredentialsFile("mycreds.txt")
else:
    gauth.Authorize()

drive = GoogleDrive(gauth)

folder_name = "Eye Comparison Results"
file_list = drive.ListFile({
    'q': f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
}).GetList()

if file_list:
    folder_id = file_list[0]['id']
else:
    folder = drive.CreateFile({'title': folder_name, 'mimeType': 'application/vnd.google-apps.folder'})
    folder.Upload()
    folder_id = folder['id']

# ====== ANALYSIS CONSTANTS ======
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [159, 145]
RIGHT_EYE = [386, 374]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
IRIS_DIAMETER_MM = 11.7
TOLERANCE_MM = 0.3

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
    center = ((points_px[0][0] + points_px[1][0]) // 2, (points_px[0][1] + points_px[1][1]) // 2)
    return max_dist, points_px, center

def palpebral_height(landmarks, eye_indices, iris_indices, image_shape, mm_per_px):
    top = landmarks[eye_indices[0]]
    bottom = landmarks[eye_indices[1]]
    height_px, top_px, bottom_px = euclidean_dist(top, bottom, image_shape)
    iris_px, iris_pts, iris_center = iris_diameter(landmarks, iris_indices, image_shape)
    if mm_per_px is None:
        if iris_px == 0:
            return None, None, None, None, None
        mm_per_px = IRIS_DIAMETER_MM / iris_px
    height_mm = height_px * mm_per_px
    return height_mm, top_px, bottom_px, iris_pts, iris_center

def draw_annotations(image, eye_name, height_mm, top_px, bottom_px, iris_pts, iris_center, text_y_offset):
    center_x = int((top_px[0] + bottom_px[0]) / 2)
    cv2.line(image, (center_x, top_px[1]), (center_x, bottom_px[1]), (0, 255, 255), 2)
    if iris_pts:
        cv2.line(image, iris_pts[0], iris_pts[1], (255, 0, 0), 2)
    text = f"{eye_name} Eye Height: {height_mm:.2f} mm (±{TOLERANCE_MM}mm)"
    cv2.putText(image, text, (30, text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if iris_center:
        label = 'L' if eye_name == 'Left' else 'R'
        cv2.putText(image, label, (iris_center[0] - 10, iris_center[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

def analyze_image(image, output_path, source_info=""):
    flip_applied = False
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            messagebox.showerror("Error", "Face landmarks not detected.")
            return
        landmarks = results.multi_face_landmarks[0].landmark

        mm_per_px = None
        lh, lt, lb, lir, lic = palpebral_height(landmarks, LEFT_EYE, LEFT_IRIS, image.shape, mm_per_px)
        rh, rt, rb, rir, ric = palpebral_height(landmarks, RIGHT_EYE, RIGHT_IRIS, image.shape, mm_per_px)

        if lh is None or rh is None:
            messagebox.showerror("Error", "Iris landmarks not detected properly.")
            return

        draw_annotations(image, "Left", lh, lt, lb, lir, lic, 40)
        draw_annotations(image, "Right", rh, rt, rb, rir, ric, 80)

        diff = abs(lh - rh)
        result = f"✅ Both eyes are equal." if diff <= TOLERANCE_MM else (
            "👁 Left eye larger." if lh > rh else "👁 Right eye larger.")
        cv2.putText(image, result, (30, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        flip_note = f"{source_info}"
        cv2.putText(image, flip_note, (30, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save image locally
        cv2.imwrite(output_path, image)

# Show result to user
        cv2.imshow("Analysis Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Now upload to Drive
        upload_to_drive(output_path)

# Inform the user
        messagebox.showinfo("Success", f"Result saved and uploaded to Drive: {output_path}")


def upload_to_drive(filepath):
    if os.path.exists(filepath):
        gfile = drive.CreateFile({'title': os.path.basename(filepath), 'parents': [{'id': folder_id}]})
        gfile.SetContentFile(filepath)
        gfile.Upload()
        print(f"✅ Uploaded {filepath} successfully!")
    else:
        print(f"❌ File not found: {filepath}")

def select_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if filepath:
        image = cv2.imread(filepath)
        if image is None:
            messagebox.showerror("Error", "Unable to load image.")
            return
        image = cv2.flip(image, 1)  # Always flip image files for mirror consistency
        output_path = os.path.join(os.path.dirname(filepath), "result_annotated.png")
        analyze_image(image, output_path, "[Image Input (Mirrored)]")

def capture_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open webcam.")
        return
    messagebox.showinfo("Info", "Press 'c' to capture, ESC to exit.")
    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam Feed - Press 'c' to capture, ESC to exit", frame)

        key = cv2.waitKey(1)
        if key == ord('c'):
            captured_frame = frame.copy()
            break
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        output_path = os.path.join(os.getcwd(), "webcam_capture_result.png")
        analyze_image(captured_frame, output_path, "[Webcam Input]")
root = tk.Tk()
root.title("Eye Palpebral Height Analyzer")
root.geometry("400x200")
root.resizable(False, False)

tk.Label(root, text="Select input mode:", font=("Arial", 14)).pack(pady=20)
tk.Button(root, text="📁 Analyze Image File", font=("Arial", 12), command=select_image).pack(pady=10)
tk.Button(root, text="🎥 Analyze Webcam Live", font=("Arial", 12), command=capture_webcam).pack(pady=10)

root.mainloop()  