import importlib

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

try:
    face_recognition = importlib.import_module("face_recognition")
    FACE_RECOGNITION_AVAILABLE = True
except ModuleNotFoundError:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False


# 1. The Vector DB (Exactly as you engineered it)
class SimpleVectorDB:
    def __init__(self):
        self.encodings = []
        self.names = []

    def insert_from_frame(self, frame, name):
        """Encodes a face directly from the live video matrix"""
        if not FACE_RECOGNITION_AVAILABLE:
            return False

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            self.encodings.append(encoding)
            self.names.append(name)
            return True
        return False

    def query(self, unknown_encoding, tolerance=0.5):
        if not FACE_RECOGNITION_AVAILABLE:
            return "Unknown"

        if not self.encodings:
            return "Unknown"
        distances = face_recognition.face_distance(self.encodings, unknown_encoding)
        best_match = np.argmin(distances)
        if distances[best_match] <= tolerance:
            return self.names[best_match]
        return "Unknown"


# 2. The Frontend Architecture
class VisionDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Compound AI Vision System")
        self.geometry("1000x600")

        # Initialize AI Engines
        self.db = SimpleVectorDB()
        self.yolo_model = YOLO("yolo11n.pt")  # Using base model for speed
        self.cap = cv2.VideoCapture(0)

        # Build UI Grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Controls) ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="Vision Dashboard",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.name_entry = ctk.CTkEntry(
            self.sidebar, placeholder_text="Enter Person's Name"
        )
        self.name_entry.grid(row=1, column=0, padx=20, pady=10)

        self.register_btn = ctk.CTkButton(
            self.sidebar, text="📸 Register Face", command=self.register_current_face
        )
        self.register_btn.grid(row=2, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(
            self.sidebar, text="Status: Ready", text_color="green"
        )
        self.status_label.grid(row=3, column=0, padx=20, pady=10)

        if not FACE_RECOGNITION_AVAILABLE:
            self.register_btn.configure(state="disabled")
            self.status_label.configure(
                text="Status: Face recognition unavailable", text_color="orange"
            )

        # --- Main Frame (Video Feed) ---
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.grid(row=0, column=1, padx=20, pady=20)

        # Start the video loop
        self.current_frame = None
        self.update_video_stream()

    def register_current_face(self):
        """Triggered when the button is clicked."""
        if not FACE_RECOGNITION_AVAILABLE:
            self.status_label.configure(
                text="Error: face_recognition is not installed", text_color="red"
            )
            return

        name = self.name_entry.get()
        if not name:
            self.status_label.configure(text="Error: Enter a name!", text_color="red")
            return

        if self.current_frame is not None:
            self.status_label.configure(text="Encoding Face...", text_color="yellow")
            self.update()  # Force UI refresh

            success = self.db.insert_from_frame(self.current_frame, name)

            if success:
                self.status_label.configure(
                    text=f"Registered: {name}", text_color="green"
                )
                self.name_entry.delete(0, "end")
            else:
                self.status_label.configure(
                    text="Error: No face detected!", text_color="red"
                )

    def update_video_stream(self):
        """The core loop: Pulls frame, runs AI, updates UI, schedules next run."""
        success, frame = self.cap.read()
        if success:
            # Save raw frame for the registration button to use
            self.current_frame = frame.copy()

            # --- 1. YOLO Object Detection ---
            # verbose=False keeps terminal clean
            results = self.yolo_model.predict(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()

            # --- 2. Identity Engine ---
            if FACE_RECOGNITION_AVAILABLE:
                small_frame = cv2.resize(annotated_frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(
                    rgb_small, face_locations
                )

                for (top, right, bottom, left), encoding in zip(
                    face_locations, face_encodings
                ):
                    name = self.db.query(encoding)
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw Red boxes for people, overlapping the YOLO boxes
                    cv2.rectangle(
                        annotated_frame, (left, top), (right, bottom), (0, 0, 255), 2
                    )
                    cv2.putText(
                        annotated_frame,
                        name,
                        (left, bottom - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

            # --- 3. Format for UI ---
            # Convert BGR (OpenCV) to RGB (Pillow)
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            ctk_image = ctk.CTkImage(
                light_image=pil_image, dark_image=pil_image, size=(640, 480)
            )

            # Update the UI Label
            self.video_label.configure(image=ctk_image)

        # Schedule this function to run again in 10 milliseconds
        self.after(10, self.update_video_stream)

    def on_closing(self):
        self.cap.release()
        self.destroy()


if __name__ == "__main__":
    # Standard Wayland compatibility override
    import os

    os.environ["QT_QPA_PLATFORM"] = "xcb"

    app = VisionDashboard()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
