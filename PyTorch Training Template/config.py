CONFIG = {
    "data_dir": "./data",
    "batch_size": 128,
    "epochs": 30,
    "lr": 1e-3,
    "num_classes": 10,
    "save_path": "best_model.pth",
    "device": "cuda" if __import__('torch').cuda.is_available() else "cpu"
}
