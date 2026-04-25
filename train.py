from ultralytics import YOLO
import torch


def main():
    # 1. Verify PyTorch can see your Nvidia GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Load the YOLO11 Nano architecture
    model = YOLO("yolo11n.pt")

    # 3. Execute Transfer Learning
    print("Starting training loop...")
    results = model.train(
        data="dataset/data.yaml",  # Points to your new Roboflow configuration
        epochs=50,  # Number of full passes through the dataset
        imgsz=416,  # Reduced resolution for VRAM safety
        batch=8,  # Small batch size prevents CUDA OOM errors
        device=0,  # Offloads matrix operations to the GPU
        workers=4,  # Uses CPU threads to load images into memory
        project="live_vision",  # Output directory for weights
        name="daily_objects",  # Sub-folder for this specific run
    )


if __name__ == "__main__":
    main()
