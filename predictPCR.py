import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

def calculate_metrics(df, label_col='pcr_label', pred_col='pcr_pred'):
    """Helper to calculate performance metrics for a given dataframe slice."""
    tp = ((df[pred_col] == 1) & (df[label_col] == 1)).sum()
    tn = ((df[pred_col] == 0) & (df[label_col] == 0)).sum()
    fp = ((df[pred_col] == 1) & (df[label_col] == 0)).sum()
    fn = ((df[pred_col] == 0) & (df[label_col] == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    bal_acc = (sensitivity + specificity) / 2
    
    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Balanced Accuracy': bal_acc,
        'Count': len(df)
    }

def scorePCR(predictionsDF_path: str,
             threshold: float = 0.5,
             clinicalInfoPath: str = rf"{os.environ.get('MAMAMIA_DATA')}/clinical_and_imaging_info.xlsx"):
    
    # 1. Load Data
    df_clinical = pd.read_excel(clinicalInfoPath, sheet_name="dataset_info")
    predDF = pd.read_csv(predictionsDF_path)

    # 2. Normalize IDs for Merging
    df_clinical['join_id'] = df_clinical['patient_id'].astype(str).str.lower()
    predDF['join_id'] = predDF['Patient ID'].astype(str).str.lower()

    # 3. Clean and Group Clinical Variables before merging
    # --- Menopause Status ---
    df_clinical['menopause'] = df_clinical['menopause'].fillna('unknown').astype(str).str.lower()
    df_clinical['menopause'] = df_clinical['menopause'].apply(
        lambda x: 'post' if 'post' in x else ('pre' if ('pre' in x or 'peri' in x) else x)
    )

    # --- Tumor Subtype (Group Luminal) ---
    df_clinical['tumor_subtype'] = df_clinical['tumor_subtype'].fillna('unknown').astype(str).str.lower()
    df_clinical['tumor_subtype'] = df_clinical['tumor_subtype'].apply(
        lambda x: 'luminal' if 'luminal' in x else x
    )

    # --- Age Binning ---
    age_bins = [0, 40, 50, 60, 120]
    age_labels = ['0-40', '41-50', '51-60', '61+']
    df_clinical['age_group'] = pd.cut(df_clinical['age'], bins=age_bins, labels=age_labels)

    # 4. Merge Clinical Info into Predictions
    predDF = predDF.merge(df_clinical, on='join_id', how='inner')

    # 5. Generate Predictions based on threshold
    predDF['pcr_pred'] = (predDF['Pred PCR'] > threshold).astype(int)
    predDF['pcr_label'] = predDF['PCR']
    
    print(f"Overall Accuracy: { (predDF['pcr_pred'] == predDF['pcr_label']).mean() * 100:.2f}%")
    print("-" * 30)

    # 6. Iterate through each grouping variable
    group_vars = ['age_group', 'tumor_subtype', 'menopause']

    for var in group_vars:
        if var not in predDF.columns:
            print(f"Skipping {var}: Column not found.")
            continue
            
        print(f"\n--- Metrics Grouped by: {var.upper()} ---")
        
        # Sort groups for consistent display
        unique_groups = sorted(predDF[var].dropna().unique())
        
        for val in unique_groups:
            subset = predDF[predDF[var] == val]
            if subset.empty:
                continue
                
            m = calculate_metrics(subset)
            print(f"Group: {val} (n={m['Count']})")
            print(f"  Sens: {m['Sensitivity']*100:.2f}% | Spec: {m['Specificity']*100:.2f}% | BalAcc: {m['Balanced Accuracy']*100:.2f}%")

if __name__ == "__main__":
    # Ensure you use the raw string path
    path = r"transformerResults/SpatioTemporalPCRNoSkips/outputsJan16-RFPFixMaskHopefullyBestPCR/scoresOG.csv"
    scorePCR(path, threshold=-0.85127)