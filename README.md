# PyTorchCardClassifier

A PyTorch-based image classification project to identify playing cards from images using a fine-tuned EfficientNet-B0 model. This project demonstrates a complete deep learning pipeline, including dataset setup, model definition, training, and evaluation, for classifying 53 unique playing card classes (e.g., "ace of clubs," "ten of diamonds").

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Visualizing Predictions](#visualizing-predictions)
- [Training Details](#training-details)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview
PyTorchCardClassifier is a deep learning project built with PyTorch to classify images of playing cards into 53 categories, covering all standard cards (ace through king across four suits) plus the joker. The project uses **transfer learning** with a pretrained **EfficientNet-B0** model, fine-tuned on a custom dataset of card images. It follows the standard PyTorch workflow: dataset preparation, model definition, and training loop, with additional tools for evaluation and visualization.

This project is ideal for learning how to:
- Set up a custom PyTorch `Dataset` and `DataLoader`.
- Leverage pretrained CNNs for transfer learning.
- Implement a training loop with validation.
- Evaluate and visualize model predictions.

## Features
- **Custom Dataset**: A `PlayingCardDataset` class to load and preprocess card images using `torchvision`'s `ImageFolder`.
- **Pretrained Model**: Fine-tunes EfficientNet-B0 from the `timm` library for high performance with minimal resources.
- **Training Pipeline**: Includes a training loop with loss tracking, validation, and GPU support.
- **Evaluation**: Computes accuracy on validation and test sets (implemented as an extension).
- **Visualization**: Displays original images alongside predicted class probabilities using Matplotlib.
- **Progress Tracking**: Uses `tqdm` for progress bars during training and evaluation.

## Dataset
The project uses the [Playing Card Image Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) from Kaggle, which contains images of playing cards organized into 53 classes (52 standard cards + joker). The dataset is split into:
- **Train**: 7,624 images (`/train/`).
- **Validation**: ~265 images (`/valid/`).
- **Test**: ~265 images (`/test/`).

Each class (e.g., `ace_of_clubs`, `ten_of_diamonds`) has its own subfolder, and images are in JPEG format. The dataset is loaded using `ImageFolder`, with preprocessing applied via `torchvision.transforms`:
- Resize images to 128x128 pixels.
- Convert images to PyTorch tensors.

**Note**: Update dataset paths in the code if running locally.

## Model Architecture
The model, `SimpleCardClassifer`, is a convolutional neural network (CNN) built on top of **EfficientNet-B0**:
- **Backbone**: EfficientNet-B0, pretrained on ImageNet, loaded via `timm`. It extracts features from input images (shape: `[batch_size, 3, 128, 128]`), producing a 1280-dimensional feature vector.
- **Custom Head**: A fully connected layer (`nn.Linear(1280, 53)`) maps features to 53 class logits.
- **Output**: Produces logits for 53 classes (shape: `[batch_size, 53]`).
- **Parameters**: ~5.3 million (EfficientNet-B0) + ~67,813 (custom head).

The model uses **transfer learning**, leveraging pretrained weights to adapt to the card classification task.

## Installation
### Prerequisites
- Python 3.10+
- A GPU (optional, but recommended for faster training)
- A Kaggle account to download the dataset (if using the provided dataset)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/mtptisid/PyTorch-CNN-model/.git
   cd PyTorch-CNN-model
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch torchvision timm matplotlib pandas numpy tqdm
   ```

4. **Download the Dataset**:
   - Download the [Playing Card Image Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) from Kaggle.
   - Extract it to a local directory (e.g., `./data/cards-image-datasetclassification/`).
   - Update the `train_folder`, `valid_folder`, and `test_folder` paths in the code to match your local setup.

## Usage
The project is implemented in a Jupyter notebook (`PyTorchCardClassifier.ipynb`). Below are instructions for key tasks.

### Training the Model
1. Open the notebook:
   ```bash
   jupyter notebook PyTorchCardClassifier.ipynb
   ```

2. Run the cells to:
   - Load and preprocess the dataset.
   - Initialize the `SimpleCardClassifer` model.
   - Set up the training loop with 5 epochs, Adam optimizer (lr=0.001), and CrossEntropyLoss.

3. Example training command (in the notebook):
   ```python
   num_epochs = 5
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   model = SimpleCardClassifer(num_classes=53).to(device)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   # Run training loop cells
   ```

   **Output**: Training and validation loss per epoch, plotted at the end.

### Evaluating the Model
To compute accuracy on the validation and test sets, add the following function to the notebook:

```python
def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (correct / total) * 100
    return accuracy

val_accuracy = calculate_accuracy(model, val_loader, device)
test_accuracy = calculate_accuracy(model, test_loader, device)
print(f"Validation Accuracy: {val_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")
```

Run the cell to evaluate the model.

### Visualizing Predictions
To visualize predictions on test images:
1. Run the visualization cells in the notebook.
2. Example output: Displays the original image and a bar chart of predicted class probabilities for 10 random test images.

```python
test_images = glob('./data/cards-image-datasetclassification/test/*/*')
test_examples = np.random.choice(test_images, 10)
for example in test_examples:
    original_image, image_tensor = preprocess_image(example, transform)
    probabilities = predict(model, image_tensor, device)
    visualize_predictions(original_image, probabilities, dataset.classes)
```

## Training Details
- **Epochs**: 5
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Function**: CrossEntropyLoss
- **Device**: GPU (CUDA) if available, else CPU
- **Preprocessing**:
  - Resize images to 128x128
  - Convert to tensors
- **Training Time**: ~11 minutes per epoch for training (239 batches), ~4–6 seconds per epoch for validation (9 batches).

## Results
The model was trained for 5 epochs, with the following training and validation losses:

| Epoch | Train Loss | Validation Loss | Training Time | Validation Time |
|-------|------------|-----------------|---------------|-----------------|
| 1     | 1.5671     | 0.4169          | ~11:30        | ~6s (1.92it/s)  |
| 2     | 0.5664     | 0.2257          | ~11:28        | ~4s (1.95it/s)  |
| 3     | 0.3339     | 0.2095          | ~11:22        | ~4s (2.33it/s)  |
| 4     | 0.2379     | 0.1470          | ~11:22        | ~5s (1.72it/s)  |
| 5     | 0.1901     | 0.1216          | ~11:23        | ~4s (2.33it/s)  |

- **Loss Curves**: Training and validation losses decreased consistently, indicating effective learning. The validation loss dropped from 0.4169 to 0.1216, with a slight fluctuation between epochs 2 (0.2257) and 3 (0.2095), suggesting minor overfitting or dataset variability. Loss curves can be visualized using the provided plotting code.
- **Accuracy**: Validation and test accuracy were not computed in the provided output. Use the `calculate_accuracy` function (see [Evaluating the Model](#evaluating-the-model)) to compute these metrics. Given the low validation loss (0.1216), expect validation accuracy to be ~90% or higher.
- **Visualization**: The model correctly predicts card classes for test images, with high-confidence probabilities visualized as bar charts.

**Loss Plot Code**:
```python
import matplotlib.pyplot as plt

train_losses = [1.5671, 0.5664, 0.3339, 0.2379, 0.1901]
val_losses = [0.4169, 0.2257, 0.2095, 0.1470, 0.1216]

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
```

## Project Structure
```
PyTorchCardClassifier/
├── PyTorchCardClassifier.ipynb  # Main Jupyter notebook with code
├── data/                        # Dataset directory (not included, download from Kaggle)
│   └── cards-image-datasetclassification/
│       ├── train/
│       ├── valid/
│       └── test/
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies (optional, can be generated)
```

To generate `requirements.txt`:
```bash
pip freeze > requirements.txt
```

## Future Improvements
- **Data Augmentation**: Add transforms like `RandomHorizontalFlip` or `RandomRotation` to improve generalization.
- **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, or more epochs.
- **Regularization**: Introduce dropout or weight decay to reduce validation loss fluctuations.
- **Model Variants**: Try larger EfficientNet models (e.g., B1, B2) or other architectures (e.g., ResNet, Vision Transformers).
- **Accuracy Metrics**: Integrate accuracy tracking during training and validation.
- **Model Saving**: Save trained model weights for reuse (`torch.save(model.state_dict(), 'model.pth')`).

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a pull request with your changes.
3. Ensure code follows PEP 8 style guidelines and includes comments.

Please report issues or suggest features via GitHub Issues.


## Acknowledgments
- [Kaggle](https://kaggle.com) for the Playing Card Image Dataset.
- [PyTorch](https://pytorch.org/) and [Timm](https://github.com/rwightman/pytorch-image-models) for excellent frameworks and pretrained models.
- [EfficientNet Authors](https://arxiv.org/abs/1905.11946) for the EfficientNet architecture.
- The open-source community for tools like Matplotlib, NumPy, and TQDM.
