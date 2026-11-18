 """
Master Pipeline Orchestrator
Executes Modules 1 through 4 sequentially:
- Module 1: Data verification and split preparation
- Module 2 & 3: Model construction, training, and optimization
- Module 4: Holdout test evaluation, confusion matrix, and error analysis
"""

import sys
import time
from src.data_loader import clean_and_verify_dataset
from src.train import train_classification_model
from src.evaluate import evaluate_model_performance

def main():
    start_time = time.time()
    print("=" * 65)
    print("  DEEP LEARNING SATELLITE IMAGE CLASSIFICATION PIPELINE")
    print("=" * 65)

    # 1. Module 1: Data Verification
    print("\n>>> [STEP 1 / 4] Dataset Verification & Preprocessing...")
    valid_count = clean_and_verify_dataset("data")
    print(f"[Step 1 Complete] Verified {valid_count} clean image samples.")

    # 2. Module 2 & 3: Model Training
    print("\n>>> [STEP 2 / 4] Building & Training Convolutional Neural Network...")
    model, history, test_ds, split_info = train_classification_model(
        data_dir="data",
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        model_save_path="models/best_satellite_model.h5",
        report_dir="reports"
    )

    # 3. Module 4: Evaluation & Error Analysis
    print("\n>>> [STEP 3 / 4] Model Evaluation on Unseen Test Set...")
    eval_results = evaluate_model_performance(
        model_path="models/best_satellite_model.h5",
        test_ds=test_ds,
        class_names=split_info["class_names"],
        report_dir="reports"
    )

    elapsed_minutes = (time.time() - start_time) / 60
    print("\n" + "=" * 65)
    print("  PIPELINE EXECUTION COMPLETE")
    print(f"  Total Elapsed Time: {elapsed_minutes:.2f} minutes")
    print(f"  Final Test Accuracy: {eval_results['accuracy'] * 100:.2f}%")
    print(f"  Best Model Saved: models/best_satellite_model.h5")
    print(f"  Artifacts Generated in: reports/")
    print("=" * 65)

if __name__ == "__main__":
    main()
