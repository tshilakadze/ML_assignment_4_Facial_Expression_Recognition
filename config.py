import torch

class Config:
    DATA_PATH = "train.csv"
    IMG_SIZE = 48
    NUM_CLASSES = 7
    VAL_SPLIT = 0.1

    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-3
    SEED = 42

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WANDB_PROJECT = "fer2013-emotion-recognition"
    EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]