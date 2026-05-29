# Trening: 10 epok, domyslne parametry 224

## Parametry
- Zbior danych: data_parveshi_224
- Rozmiar obrazu: 224 x 224
- Epoki: Baseline CNN: 10, ResNet50: 10
- Learning Rate: Baseline CNN: 0.001, ResNet50: 0.0001
- Batch Size: 128

## Wyniki (Test Set)
| Model | Celnosc | F1-Score |
| :--- | :---: | :---: |
| Baseline CNN | 93.21% | 0.9331 |
| ResNet50 | 98.82% | 0.9883 |

## Zawartosc folderu
- `baseline.pth`: wagi wytrenowanego Baseline CNN
- `baseline_confusion_matrix.png`: macierz pomyłek dla Baseline CNN
- `baseline_epoch_logs.csv`: logi epok dla CNN
- `model.pth`: wagi wytrenowanego ResNet50
- `resnet_confusion_matrix.png`: macierz pomyłek dla ResNet50
- `resnet_epoch_logs.csv`: logi epok dla ResNet
- `results.csv`: plik CSV z metrykami celności i F1
- `roc_pr_curves.png`: wykresy krzywych ROC i Precision-Recall
- `training_curves.png`: wykresy zmian straty i celności podczas epok
