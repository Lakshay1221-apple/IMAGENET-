from pathlib import Path

from ultralytics import YOLO


def main():
    checkpoint = Path(__file__).resolve().parent.parent / "runs" / "detect" / "live_vision" / "daily_objects-2" / "weights" / "last.pt"

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = YOLO(str(checkpoint))

    print("Resuming training from the last saved epoch...")
    model.train(resume=True)


if __name__ == "__main__":
    main()
