# Satellite Land Cover Image Classification using Deep Learning

An end-to-end deep learning system that classifies satellite and aerial imagery into four terrain categories: **Cloudy**, **Desert**, **Green Area**, and **Water**.

## Project Modules (As per Specification)
1. **Module 1: Data Acquisition & Preparation**
   - Quality control and integrity checks on 5,631 satellite images.
   - Conversion of all color profiles to standardized 3-channel RGB.
   - Stratified splitting into 80% Train, 10% Validation, and 10% Unseen Test sets.
   - Data augmentation: Random horizontal/vertical flips, rotations, and zoom for orientation-invariant aerial feature learning.

2. **Module 2: Model Architecture & Design**
   - **Custom CNN Architecture**: 3 Convolutional blocks with Batch Normalization, Max Pooling, and progressive Dropout (0.15 to 0.45).
   - Global Average Pooling layer to prevent dense parameter explosion.
   - Alternative transfer learning design (MobileNetV2) for benchmarking.

3. **Module 3: Model Training & Optimization**
   - Optimizer: Adam (`learning_rate=0.001`).
   - Loss Function: Sparse Categorical Crossentropy.
   - Dynamic Callbacks: Early Stopping (`patience=5`), Learning Rate Plateau Reduction (`factor=0.5`), and Best Weight Checkpointing.

4. **Module 4: Model Evaluation & Error Analysis**
   - Independent testing on holdout test set.
   - Metrics: Accuracy, Precision, Recall, and F1-Score per class.
   - Confusion matrix visualization saved to `reports/confusion_matrix.png`.
   - Error analysis plotting misclassified samples to diagnose borderline cases.

5. **Module 5: Deployment & User Interface**
   - Interactive web application powered by **Streamlit**.
   - Supports drag-and-drop file upload or sample selection from the dataset.
   - Real-time classification confidence and class probability distribution charts.

---

## Directory Structure
```
d:\ProjectsFDA\
│
├── data/
│   ├── cloudy/
│   ├── desert/
│   ├── green_area/
│   └── water/
│
├── models/
│   └── best_satellite_model.keras    # Saved trained neural network
│
├── reports/
│   ├── classification_report.txt      # Precision, Recall, F1
│   ├── confusion_matrix.png          # Confusion Matrix Heatmap
│   ├── training_curves.png           # Accuracy and Loss vs Epochs
│   └── error_analysis_samples.png    # Top misclassified images
│
├── src/
│   ├── data_loader.py                # Module 1
│   ├── model.py                      # Module 2
│   ├── train.py                      # Module 3
│   ├── evaluate.py                   # Module 4
│   └── utils.py                      # Visualization utilities
│
├── app.py                            # Module 5 Web App
├── run_pipeline.py                   # Full automated pipeline
└── README.md
```

---

## How to Run

### 1. Execute Training & Evaluation
To train the model and generate all evaluation reports in one step:
```bash
python run_pipeline.py
```

### 2. Launch the Web Application
To test predictions in the interactive web interface:
```bash
streamlit run app.py
```
