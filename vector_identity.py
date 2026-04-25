import cv2
import face_recognition
import numpy as np
import os


class SimpleVectorDB:
    def __init__(self):
        # In-memory arrays acting as our Vector Database
        self.known_face_encodings = []
        self.known_face_names = []

    def insert(self, image_path, name):
        """Pass an image through the Siamese Encoder and store the Vector."""
        try:
            image = face_recognition.load_image_file(image_path)
            # The encoder outputs a 128-dimensional numpy array
            encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
            print(f"[DB] Inserted Vector for: {name}")
        except IndexError:
            print(f"[ERROR] No face found in {image_path}")

    def query(self, unknown_encoding, tolerance=0.6):
        """Calculate Euclidean distance against all stored vectors."""
        if not self.known_face_encodings:
            return "Unknown"

        # Calculate pure mathematical distance between the live vector and all DB vectors
        face_distances = face_recognition.face_distance(
            self.known_face_encodings, unknown_encoding
        )

        # Find the index of the smallest distance
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] <= tolerance:
            return self.known_face_names[best_match_index]
        return "Unknown"


def main():
    # 1. Initialize our Vector DB
    db = SimpleVectorDB()

    # 2. Register Identities (You can add photos to your directory to test this)
    # db.insert("dataset/lakshay_reference.jpg", "Lakshay")
    # db.insert("dataset/friend_reference.jpg", "Friend")

    # 3. Boot the Camera
    cap = cv2.VideoCapture(0)
    print("Booting Identity Engine... Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Resize frame to 1/4 size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert BGR (OpenCV) to RGB (face_recognition requirement)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            # Query the Vector DB using the math you proposed
            name = db.query(face_encoding)

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw the UI
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED
            )
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 0),
                1,
            )

        os.environ["QT_QPA_PLATFORM"] = "xcb"  # Wayland fix
        cv2.imshow("Vector Identity Engine", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
