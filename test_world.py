import cv2
from ultralytics import YOLOWorld
import os


def main():
    # Force XWayland compatibility for your Hyprland setup
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    # 1. Load the YOLO-World architecture
    # The 's' version (small) is highly optimized and will fit in your 4GB VRAM
    model = YOLOWorld("yolov8s-world.pt")

    # 2. Define your custom vocabulary
    # You can type literally anything here. The model will try to find it.
    custom_classes = [
        "air conditioner",
        "pen",
        "book",
        "person",
        "plastic bottle",
        "computer mouse",
    ]
    model.set_classes(custom_classes)

    # 3. Initialize the webcam
    cap = cv2.VideoCapture(0)
    print(f"Booting YOLO-World Engine...")
    print(f"Searching for: {custom_classes}")
    print("Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 4. Run inference
        # We lower confidence to 0.1 because zero-shot predictions are naturally
        # less confident than heavily trained closed-world predictions.
        results = model.predict(frame, conf=0.1, verbose=False)

        # 5. Render and display
        annotated_frame = results[0].plot()
        cv2.imshow("Zero-Shot Open Vocabulary Vision", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
