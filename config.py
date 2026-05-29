import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR    = "/mnt/j/biai/data_parveshi_224"
MODEL_DIR   = "model"
MODEL_PATH  = "model/model.pth"
RESULTS_CSV = "model/results.csv"
PLOT_PATH   = "model/training_curves.png"

BATCH_SIZE       = 256
EPOCHS_BASELINE  = 10
EPOCHS_FINETUNE  = 10
LR_BASELINE      = 0.001
LR_FINETUNE      = 0.0001

IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# fake=0, real=1
CLASSES = ["AI GENERATED", "REAL"]

SUPPORTED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp",
    ".gif", ".tiff", ".tif", ".webp",
    ".ico", ".ppm", ".pgm", ".pbm"
)
