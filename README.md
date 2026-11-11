# 🏏 Indian Cricket Team Member Face Identification System

An end-to-end Computer Vision and Machine Learning system that detects and identifies Indian cricket team members from facial imagery. The system extracts multi-modal structural and frequency-domain facial descriptors and classifies input portraits into the corresponding cricketer's name with confidence probabilities and career profiles.

---

## 📌 Problem Statement

Develop a machine learning / computer vision system that can identify an individual from an image based on their facial features. The system is trained on images of selected Indian cricket team members and classifies any new input image into the corresponding player's identity with quantified confidence scores.

### 🇮🇳 Selected Indian Cricket Team Members
| Cricketer | Jersey | Role | Signature Facial Features |
| :--- | :---: | :--- | :--- |
| **Virat Kohli** | `#18` | Top-Order Batter / Former Captain | Sharp jawline, thick groomed beard, focused gaze, athletic fade |
| **Rohit Sharma** | `#45` | Opening Batter / Indian Captain | Rounder facial contour, calm eyes, trim beard and mustache |
| **MS Dhoni** | `#7` | Wicketkeeper-Batter / Legendary Captain | Broad masculine chin, steady calm eyes, salt-and-pepper stubble |
| **Sachin Tendulkar** | `#10` | Top-Order Batter / Master Blaster | Classic curly hair, compact facial structure, warm smile |
| **Hardik Pandya** | `#33` | Fast-Bowling All-Rounder | Sculpted cheekbones, styled goatee/designer beard, trendy fade |
| **Jasprit Bumrah** | `#93` | Premier Fast Bowler / Yorker King | Expressive smile, prominent smile lines, curly hair, light stubble |

---

## 🔬 System Architecture & Methodology

```
                   +------------------------+
                   |   Input Image / Feed   |
                   +-----------+------------+
                               |
                               v
             +------------------------------------+
             | OpenCV Haar Cascade Face Detection |
             | (haarcascade_frontalface_default)  |
             +-----------------+------------------+
                               |
                               v
             +------------------------------------+
             | Crop & Standardize Face ROI (64x64)|
             |  + Boundary Padding Margin (12%)   |
             +-----------------+------------------+
                               |
                               v
    +------------------------------------------------------+
    |          Multi-Modal Feature Extraction              |
    |  • HOG (Histogram of Oriented Gradients):  1,764-d   |
    |  • 2D Haar Wavelet Detail Transform:       1,024-d   |
    |  • Local Binary Patterns (LBP) Histogram:    256-d   |
    |  • Normalized Intensity Map:               1,024-d   |
    |  --------------------------------------------------  |
    |  Total Feature Dimension per Face:         4,068-d   |
    +--------------------------+---------------------------+
                               |
                               v
             +------------------------------------+
             |   Feature Scaling (StandardScaler) |
             +-----------------+------------------+
                               |
                               v
             +------------------------------------+
             |   Machine Learning Classification  |
             | • SVM (Linear Kernel) [Champion]   |
             | • SVM (RBF Kernel)                 |
             | • Random Forest (120 Trees)        |
             | • Multinomial Logistic Regression  |
             | • K-Nearest Neighbors              |
             +-----------------+------------------+
                               |
                               v
         +--------------------------------------------+
         | • Predicted Cricketer Name & Confidence %  |
         | • Probability Distribution Across Squad    |
         | • Bounding Box Overlay Annotation          |
         | • Career Highlights & Player Biography     |
         +--------------------------------------------+
```

### 1. Face Localization & Alignment
- Utilizes OpenCV's pre-trained Haar Cascades (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`).
- Adaptive multi-scale scanning (`scaleFactor=1.1`, fallback `1.05`) detects faces even in challenging angles.
- Adds an adaptive 12% margin around the bounding box to capture chin, forehead, and hair contours.
- Standardizes all detected faces to a uniform $64 \times 64$ dimension.

### 2. Multi-Modal Feature Representation (4,068 Dimensions)
- **Histogram of Oriented Gradients (HOG) (1,764-d):** Encodes edge directions and structural facial geometry (chin curve, eye sockets, nose bridge) invariant to lighting shifts.
- **2D Discrete Haar Wavelet Transform (1,024-d):** Isolates high-frequency facial textures while suppressing low-frequency ambient illumination bias.
- **Local Binary Patterns (LBP) (256-d):** Micro-texture descriptors capturing pore/skin patterns, stubble, and beard density.
- **Normalized Raw Intensity Map (1,024-d):** Spatial grayscale layout of facial landmarks.

### 3. Data Augmentation Pipeline
To ensure robust generalization against stadium lighting and camera noise, each face chip is augmented with:
- Horizontal reflection (facial symmetry)
- Subtle rotations ($\pm 8^\circ$)
- Illumination scaling ($0.85\times$ underexposed, $1.15\times$ overexposed)
- Motion blur emulation (Gaussian smoothing)

---

## 📊 Experimental Results & Model Benchmark

Trained and evaluated using **Stratified 5-Fold Cross Validation** and an independent **20% held-out test split**:

| Machine Learning Algorithm | 5-Fold CV Accuracy | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (Linear Kernel)** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | 🏆 **Champion** |
| **SVM (RBF Kernel)** | 99.60% | 100.00% | 1.0000 | 1.0000 | 1.0000 | Runner-up |
| **Logistic Regression** | 99.80% | 100.00% | 1.0000 | 1.0000 | 1.0000 | Top Performer |
| **Random Forest (120 Trees)** | 97.82% | 96.83% | 0.9695 | 0.9683 | 0.9682 | Strong Baseline |
| **K-Nearest Neighbors (KNN)** | 94.84% | 99.21% | 0.9924 | 0.9921 | 0.9921 | Baseline |

### Evaluation Artifacts
- **Confusion Matrix:** [`artifacts/confusion_matrix.png`](file:///c:/project%202/artifacts/confusion_matrix.png)
- **Model Comparison Chart:** [`artifacts/model_comparison.png`](file:///c:/project%202/artifacts/model_comparison.png)
- **Metrics Summary:** [`artifacts/metrics.json`](file:///c:/project%202/artifacts/metrics.json)

---

## 📁 Dataset Resources (Kaggle & Google Dataset Search)

This project natively integrates with public datasets from Kaggle and Google Dataset Search:

### 1. Recommended Kaggle Datasets
1. **[Indian Cricketers Image Dataset (Shilpee Prasad)](https://www.kaggle.com/datasets/shilpeeprasad/indian-cricketers-image-dataset)**
   - ~1,400 image files organized in subdirectories named after Indian cricketers (~50 images per player).
   - High-resolution coverage of Virat Kohli, Rohit Sharma, MS Dhoni, Sachin Tendulkar, etc.
2. **[Indian Cricketers Dataset (Aminul Hossain)](https://www.kaggle.com/datasets/aminulhossain01/indian-cricketers-dataset)**
   - Curated gallery of prominent Indian men and women cricketers.
3. **[Cricket Players Faces (Pranav Surte)](https://www.kaggle.com/datasets/pranavsurte/cricket-players-faces)**
   - Cropped facial dataset from ICC World Cup tournaments formatted for facial classification.

### 2. Google Dataset Search
- Query: `Indian cricketer face dataset` or `Indian cricketers image dataset` on [Google Dataset Search](https://datasetsearch.research.google.com/).

### 3. Automated Dataset Management Script
Use the built-in dataset manager `download_kaggle_dataset.py`:
```bash
# Option A: Automatic download via Kaggle API (requires ~/.kaggle/kaggle.json)
python download_kaggle_dataset.py --download shilpeeprasad/indian-cricketers-image-dataset

# Option B: Extract and auto-organize a downloaded zip archive
python download_kaggle_dataset.py --extract path/to/dataset.zip

# Option C: Normalize and clean folder names in data/raw/
python download_kaggle_dataset.py --clean-names

# Option D: View dataset guide
python download_kaggle_dataset.py --guide
```

---

## 📂 Project Directory Structure

```
c:\project 2\
├── data/
│   ├── raw/                       # Original images organized by player subfolder
│   │   ├── virat_kohli/
│   │   ├── rohit_sharma/
│   │   ├── ms_dhoni/
│   │   ├── sachin_tendulkar/
│   │   ├── hardik_pandya/
│   │   └── jasprit_bumrah/
│   ├── processed/                 # Cropped, normalized 64x64 face chips (with augmentations)
│   └── test_samples/              # Dedicated test images for inference & app evaluation
│       ├── virat_kohli_test1.jpg
│       ├── rohit_sharma_test1.jpg
│       └── ...
│
├── src/
│   ├── __init__.py
│   ├── detector.py                # Face detection & bounding box overlay with Haar Cascades
│   ├── features.py                # HOG, 2D Haar Wavelet, LBP, and Raw feature extractors
│   ├── preprocess.py              # Face crop pipeline with data augmentation
│   ├── models.py                  # Model definitions, CV benchmarking, and artifact serialization
│   ├── evaluate.py                # Confusion matrix, comparative bar charts, classification reports
│   └── player_info.py             # Player career stats, nicknames, jersey numbers, and highlights
│
├── artifacts/                     # Serialized models and evaluation plots
│   ├── best_model.pkl             # Trained Champion Model (SVM)
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── class_mapping.json         # Numerical class ID to Player Name map
│   ├── metrics.json               # JSON performance metrics across all models
│   ├── confusion_matrix.png       # Publication-quality heatmap
│   ├── model_comparison.png       # Benchmark comparison bar chart
│   └── predictions/               # Saved annotated inference outputs
│
├── build_starter_dataset.py       # Generates biometric starter dataset & test samples
├── download_kaggle_dataset.py     # Kaggle & Google Dataset Search downloader & importer
├── train.py                       # End-to-end model training & benchmark script
├── predict.py                     # Command-line inference tool for single/batch images
├── app.py                         # Streamlit interactive web application with webcam support
├── requirements.txt               # Python package dependencies
└── README.md                      # Comprehensive documentation
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
If using the built-in starter dataset:
```bash
python build_starter_dataset.py
```
*(If using Kaggle, download and extract images into `data/raw/<player_name>/` using `python download_kaggle_dataset.py`)*

### Step 3: Run Preprocessing & Augmentation
```bash
python src/preprocess.py
```

### Step 4: Train & Benchmark Models
```bash
python train.py
```
This trains all models, selects the champion, and exports artifacts to `artifacts/`.

### Step 5: Test via Command-Line Inference
```bash
# Evaluate on all dedicated test samples
python predict.py --test-all

# Run inference on a specific image
python predict.py --image data/test_samples/virat_kohli_test1.jpg
```

### Step 6: Launch Interactive Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to interact with the full web UI!

---

## 🖥️ Streamlit Web Application Features

1. **🔍 Single Face Identification:** Upload any JPG/PNG portrait or select from pre-loaded test images. See the bounding box overlay, identified cricketer name, and confidence percentage.
2. **📈 Confidence Distribution:** Interactive Plotly horizontal bar chart displaying prediction probabilities across all squad members.
3. **🏏 Player Bio Card:** Displays cricketer jersey number, playing role, nickname, facial signatures, and career achievements.
4. **📸 Live Webcam Recognition:** Capture a photo directly from your camera to test face recognition in real time.
5. **📊 Evaluation Analytics:** View the confusion matrix heatmap, cross-validation metrics, and multi-model benchmark chart.
6. **📁 Kaggle Integration Hub:** Built-in guidance for importing custom player datasets.

---

## 🛠️ Technologies Used
- **Language:** Python 3.11
- **Computer Vision:** OpenCV (Haar Cascades, HOG Descriptor, CLAHE)
- **Signal Processing:** 2D Discrete Haar Wavelet Transform (NumPy)
- **Machine Learning:** scikit-learn (SVM, Random Forest, Logistic Regression, KNN, StandardScaler, StratifiedKFold)
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Web Interface:** Streamlit

