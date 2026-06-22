# Rice Leaf Disease Classification Using SVM

A classical machine-learning image classification project that uses **Histogram of Oriented Gradients (HOG)** features and a **Support Vector Machine (SVM)** with a radial basis function (RBF) kernel to classify rice-leaf images.

## Project Overview

The classifier recognizes the following six categories:

1. `BACTERIAL_BLIGHT`
2. `BROWN_SPOT`
3. `HEALTHY`
4. `LEAF_BLAST`
5. `LEAF_SCALD`
6. `NOT_A_RICE_LEAF`

The final dataset used in the notebook contains **2,100 images**, with **350 images per class**. The data is divided using a stratified 80:20 train-test split:

- Training set: 1,680 images
- Test set: 420 images

## Dataset Attribution

- **Dataset name:** Zambali Rice Dataset (Public V1)
- **Dataset creator/uploader:** Daniel Zachary M. Mercurio
- **Platform:** Kaggle
- **Original source:** [https://www.kaggle.com/datasets/username/dataset-name](https://www.kaggle.com/datasets/gettingintoml/zambali-rice-dataset-v3-1?select=ZAMBALI_RICE_DATASET_V3)
- **License:** Apache 2.0
- **Date accessed:** April 24, 2026

The dataset files are not included in this repository. To reproduce the
project, download the dataset from the original Kaggle source and arrange
the files according to the folder structure described below.

## Machine-Learning Pipeline

1. Load image filenames and class labels from the dataset folders.
2. Remove duplicate filename-label records and check for missing values.
3. Encode the class labels using `LabelEncoder`.
4. Split the data using stratified sampling with `random_state=42`.
5. Convert each image to grayscale.
6. Resize each image to 64 × 64 pixels.
7. Extract HOG features using:
   - Pixels per cell: `(4, 4)`
   - Cells per block: `(2, 2)`
8. Standardize the extracted features using `StandardScaler`.
9. Train an RBF-kernel support vector classifier using:
   - `C=10`
   - `gamma="scale"`
10. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.

## Results

### Overall Evaluation

| Metric | Result |
|---|---:|
| Test accuracy | 79.52% |
| Macro-average precision | 0.80 |
| Macro-average recall | 0.80 |
| Macro-average F1-score | 0.79 |
| External test accuracy | 75.00% (15/20 images) |

### Per-Class Test Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Bacterial Blight | 0.96 | 0.99 | 0.97 | 70 |
| Brown Spot | 0.80 | 0.53 | 0.64 | 70 |
| Healthy | 0.58 | 0.80 | 0.67 | 70 |
| Leaf Blast | 0.63 | 0.56 | 0.59 | 70 |
| Leaf Scald | 0.93 | 0.94 | 0.94 | 70 |
| Not a Rice Leaf | 0.93 | 0.96 | 0.94 | 70 |

### Confusion Matrix

```text
[[69, 0,  0,  0, 1, 0],
 [ 1, 37, 17, 9, 2, 4],
 [ 0, 1, 56, 13, 0, 0],
 [ 2, 5, 21, 39, 2, 1],
 [ 0, 3,  0,  1, 66, 0],
 [ 0, 0,  3,  0, 0, 67]]
```

The model performed strongest on `BACTERIAL_BLIGHT`, `LEAF_SCALD`, and `NOT_A_RICE_LEAF`. Most errors occurred among `BROWN_SPOT`, `HEALTHY`, and `LEAF_BLAST`, whose visual characteristics may overlap after grayscale conversion and resizing.

## Repository Contents

```text
rice-leaf-disease-svm-classifier/
├── rice_leaf_svm_classifier.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The dataset folders and generated model files are intentionally excluded from the repository.

## Dataset Setup

Obtain an authorized copy of the dataset used for the project and place it beside the notebook using this folder structure:

```text
RiceLeafsDisease/
├── BACTERIAL_BLIGHT/
├── BROWN_SPOT/
├── HEALTHY/
├── LEAF_BLAST/
├── LEAF_SCALD/
└── NOT_A_RICE_LEAF/

TEST_RICE_IMAGES/
├── BACTERIAL_BLIGHT_1.jpg
├── BROWN_SPOT_1.jpg
└── ...
```

The notebook expects the training dataset directory to be named `RiceLeafsDisease` and the external test directory to be named `TEST_RICE_IMAGES`.

External test filenames should follow the format:

```text
CLASS_NAME_number.extension
```

For example:

```text
BACTERIAL_BLIGHT_1.jpg
NOT_A_RICE_LEAF_4.jpg
```

The image files are not redistributed in this repository because of file-size and dataset-licensing considerations. Users should comply with the original dataset's license and terms of use.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/rice-leaf-disease-svm-classifier.git
cd rice-leaf-disease-svm-classifier
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Start JupyterLab

```bash
jupyter lab
```

Open `rice_leaf_svm_classifier.ipynb` and run the cells in order.

## Model Outputs

The notebook generates the following files after training:

```text
svm_rice_model.pkl
scaler.pkl
```

For complete standalone inference, also save the fitted label encoder:

```python
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(le, file)
```

Generated `.pkl` files are excluded by `.gitignore`. They can be recreated by rerunning the notebook.

> **Security note:** Only load pickle files from sources you trust.

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- pandas
- OpenCV
- scikit-image
- scikit-learn
- Matplotlib
- Seaborn

## Limitations

- The external evaluation contains only 20 images.
- HOG features may not fully capture subtle color and texture differences among visually similar diseases.
- Converting images to grayscale removes color information that may help distinguish some conditions.
- Performance on field images may differ because of lighting, background, camera quality, leaf orientation, and disease severity.
- The repository does not include the original dataset or trained model artifacts.

## Future Improvements

- Compare HOG-SVM performance with convolutional neural networks or transfer-learning models.
- Preserve and evaluate color-based features.
- Use cross-validation and systematic hyperparameter tuning.
- Expand the external test set using images from varied field conditions.
- Add a reusable inference script or a lightweight web interface.
- Save the model, scaler, and label encoder as a single reproducible pipeline.

## Academic and Ethical Use

This repository presents an academic machine-learning project. Predictions should not be treated as a substitute for diagnosis by an agricultural specialist. Dataset ownership, licensing, and contributor attribution should be respected when reproducing or extending the work.
