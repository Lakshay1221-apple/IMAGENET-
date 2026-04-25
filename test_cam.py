import cv2
from ultralytics import YOLO


def main():
    # 1. Load your newly forged weights
    model_path = (
        YOLO("yolo11n.pt")
    )
    model = YOLO(model_path)

    # 2. Initialize the webcam feed
    # '0' is the default index for laptop webcams
    cap = cv2.VideoCapture(0)

    print("Booting Vision Engine... Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame. Is your webcam being used by another app?")
            break

        # 3. Run the frame through the YOLO network
        # conf=0.5 ensures it only draws boxes for predictions it is 50%+ sure about
        # verbose=False stops the terminal from printing a log for every single frame
        results = model.predict(frame, conf=0.5, verbose=False)

        # 4. Render the bounding boxes and labels onto the raw frame matrix
        annotated_frame = results[0].plot()

        # 5. Display the feed
        cv2.imshow("Custom YOLO Vision", annotated_frame)

        # 6. Listen for the 'q' key to gracefully shut down the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 7. Release hardware resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
