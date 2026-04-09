# AI Image Detector

This project classifies images into two classes:
- REAL
- AI GENERATED

Training and inference are implemented in PyTorch, and the user interface is built with Streamlit.

<img width="1919" height="1085" alt="Zrzut ekranu 2026-04-09 160752" src="https://github.com/user-attachments/assets/a7efa793-8eb5-430a-abc4-b6542745486b" />

## What This Project Does

1. Trains two models:
- A custom Baseline CNN trained from scratch.
- A fine-tuned ResNet50 model (transfer learning).
2. Evaluates both models using Accuracy and F1-score.
3. Saves model checkpoints, metrics, and training plots.
4. Runs a web app where users upload an image and receive a prediction with confidence.

## Tech Stack

- Python
- torch, torchvision
- scikit-learn
- matplotlib, pandas
- streamlit

## Dataset Layout

Expected directory structure:

```text
data/
	train/
		fake/
		real/
	test/
		fake/
		real/
```

Supported image extensions for dataset loading:
.jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

Dataset I used for training :
https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images

## Train/Test Split Logic

The loader logic is implemented in model/dataset.py.

1. If both test class folders contain valid images:
- Training data is loaded from data/train.
- Test data is loaded from data/test.

2. If data/test is empty or incomplete:
- Training does not stop.
- A fallback split is created from data/train.
- Default fallback ratio is 80 percent train and 20 percent test.
- The split is reproducible with seed 42.

Current (quite small) dataset status I used in recent runs:
- data/train/fake: 540 images
- data/train/real: 435 images
- data/test/fake: 0 images
- data/test/real: 0 images
- Effective fallback split used by training: Train 780, Test 195

And results:

<img width="2100" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/d369834d-7f3a-4bf7-9f68-d335cadd3a57" />

| Model | Accuracy | F1-score |
|---|---|---|
| Baseline CNN | 61.03% | 0.5683 |
| ResNet50 (fine-tuned) | 81.03% | 0.8104 |


## Setup

1. Open the project directory.
2. Create and activate a virtual environment.
3. Install dependencies.

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/Mac:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train the Models

Run:

```bash
python train.py
```

Training workflow:

1. Create train and test loaders via get_loaders().
2. Train Baseline CNN.
3. Evaluate Baseline CNN on the test loader.
4. Train ResNet50.
5. Evaluate ResNet50 on the test loader.
6. Save checkpoints, metrics table, and plots.

## Baseline CNN vs ResNet50

| Aspect | Baseline CNN | ResNet50 |
|---|---|---|
| Model type | Custom small CNN | Deep residual network |
| Initialization | Random weights | ImageNet pretrained weights |
| Training strategy | Train from scratch | Transfer learning + fine tuning |
| Capacity | Lower | Higher |
| Speed and compute | Faster, lighter | Slower, heavier |
| Typical performance | Good baseline reference | Usually better final accuracy/F1 |
| Role in this project | Reference model | Main production model used in app |

In this repository, Baseline CNN is trained to provide a simple benchmark, while ResNet50 is the primary model expected to deliver stronger results on real vs AI-generated image classification.

## Training Outputs

- model/baseline.pth
- model/model.pth
- model/results.csv
- model/training_curves.png

## Model Weights Sharing

The .pth files contain real trained model weights and can be too large for regular GitHub tracking.

In this repository, .pth files are intentionally ignored by the gitignore file.

We can share .pth files using one of these options:
- GitHub Releases
- Google Drive or OneDrive

## Run the Web App

```bash
streamlit run app.py
```

The app:

1. Loads model/model.pth.
2. Accepts an uploaded image.
3. Returns predicted class, confidence, and class probabilities.

Important: the web app uses the ResNet50 checkpoint stored in model/model.pth (not model/baseline.pth).

## Project File Purposes

- app.py
Streamlit UI entry point. Loads the trained model, preprocesses uploaded images, runs inference, and displays prediction results and confidence.

- train.py
Main training script. Builds loaders, trains Baseline CNN and ResNet50, evaluates both models, and saves artifacts.

- config.py
Central configuration file. Stores device selection, paths, hyperparameters, class labels, and supported file extensions.

- requirements.txt
Python dependencies required for training and running the app.

- model/architectures.py
Defines model architectures: BaselineCNN and ResNet50 construction with optional backbone freezing.

- model/dataset.py
Creates transforms and data loaders. Handles test-folder availability and fallback train/test split logic.

- model/trainer.py
Contains training and evaluation loops, loss/accuracy tracking, and sklearn-based metrics computation.

- model/reporting.py
Saves summary metrics to CSV and training curves to an image file.

- model/model.pth
Final ResNet50 checkpoint used by the Streamlit app for inference.

- model/baseline.pth
Checkpoint of the baseline CNN model.

- model/results.csv
Table with final Accuracy and F1-score for trained models.

- model/training_curves.png
Training loss and accuracy curves for Baseline CNN and ResNet50.

- data/
Dataset root folder. Contains train and optionally test class folders.

## Important Configuration

Edit config.py to adjust key training and runtime parameters:

- DEVICE
Automatically selects `cuda` when available, otherwise `cpu`.

- DATA_DIR
Root path to the dataset directory (`train/` and `test/` folders).

- MODEL_DIR
Directory where model checkpoints are saved.

- MODEL_PATH
Path to the main checkpoint used by the app (`model/model.pth`, ResNet50).

- RESULTS_CSV
Path for the final metrics table saved after training.

- PLOT_PATH
Path for the saved training curves figure.

- BATCH_SIZE
Number of images per batch during training and evaluation.

- EPOCHS_BASELINE
Number of epochs used for the Baseline CNN.

- EPOCHS_FINETUNE
Number of epochs used for ResNet50 fine-tuning.

- LR_BASELINE
Learning rate for the Baseline CNN optimizer.

- LR_FINETUNE
Learning rate for the ResNet50 optimizer (usually lower than baseline).

- IMAGE_SIZE
Input size used by preprocessing transforms (default: 224x224).

- NORMALIZE_MEAN / NORMALIZE_STD
Image normalization values applied before passing images to the model.

- CLASSES
Display labels used in the app output (`AI GENERATED`, `REAL`).

- SUPPORTED_EXTENSIONS
Allowed image extensions for upload and preprocessing.

## Common Issues

1. ImageFolder FileNotFoundError:
- Verify class folders exist.
- Verify images use supported extensions.
- Verify you run commands from the project root directory.

2. Streamlit app says model is missing:
- Run python train.py first.
- Then run streamlit run app.py.

## Authors

- Błażej Jamrozik: [GitHub Profile](https://github.com/Blazejoida)
- Bartosz Kepa: [GitHub Profile](https://github.com/Dedeusz04)
