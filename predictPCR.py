import SimpleITK as sitk
import os
import numpy as np
import pandas as pd

def scorePCR(outputDir: str,
             threshold: float = 0.5,
             clinicalInfoPath: str = rf"{os.environ["MAMAMIA_DATA"]}/clinical_and_imaging_info.xlsx"):
    df = pd.read_excel(clinicalInfoPath, sheet_name="dataset_info")

    out_df = pd.DataFrame(columns=['patient_id', 'pcr_pred', 'pcr_label', 'correct'])
    
    for imgName in os.listdir(outputDir):
        if not imgName.endswith('.nii.gz'):
            continue
        img = sitk.ReadImage(os.path.join(outputDir, imgName))
        arr = sitk.GetArrayFromImage(img)
        pcr_pred = int(np.mean(arr) > threshold)
        patient_id = imgName.split('.')[0].upper()
        # print("Processing patient:", patient_id)
        # print("Arr", np.unique(arr))
        # print("fg/total:", np.mean(arr))
        # print("final prediction:", pcr_pred)

        pcr_label = df.loc[df['patient_id'] == patient_id, 'pcr'].values[0]
        # print("Actual label:", pcr_label)

        correct = pcr_pred == pcr_label
        new_row = pd.DataFrame({
            'patient_id': [patient_id],
            'pcr_pred': [pcr_pred],
            'pcr_label': [pcr_label],
            'correct': [correct]
        })
        out_df = pd.concat([out_df, new_row], ignore_index=True)

    out_df.to_csv(os.path.join(os.path.dirname(outputDir), 'pcr_scores.csv'), index=False)

    out_df['pcr_pred'] = 0
    out_df['correct'] = out_df['pcr_label'] == out_df['pcr_pred']

    print("Percentage of correct predictions:",
          out_df['correct'].mean() * 100, "%")
    groups = ['DUKE', 'ISPY1', 'ISPY2', 'NACT']
    avg_accuracy = {group : out_df[out_df['patient_id'].str.contains(group, na=False)]['correct'].astype(int).mean() \
                    for group in groups}
    print("Average accuracy per group:")
    for group, accuracy in avg_accuracy.items():
        print(f"{group}: {accuracy * 100:.2f}%")

    # Calculate Specificity, Sensitivity, Recall, and Precision for each dataset group
    # Include both Precision and Balanced Accuracy calculations
    metrics = ['Specificity', 'Sensitivity', 'Precision', 'Balanced Accuracy']
    results = {group: {metric: 0 for metric in metrics} for group in groups}

    for group in groups:
        group_df = out_df[out_df['patient_id'].str.contains(group, na=False)]
        tp = ((group_df['pcr_pred'] == 1) & (group_df['pcr_label'] == 1)).sum()
        tn = ((group_df['pcr_pred'] == 0) & (group_df['pcr_label'] == 0)).sum()
        fp = ((group_df['pcr_pred'] == 1) & (group_df['pcr_label'] == 0)).sum()
        fn = ((group_df['pcr_pred'] == 0) & (group_df['pcr_label'] == 1)).sum()

        results[group]['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        results[group]['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        results[group]['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        results[group]['Balanced Accuracy'] = (results[group]['Sensitivity'] + results[group]['Specificity']) / 2

    print("Metrics per group:")
    for group, metrics in results.items():
        print(f"{group}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value * 100:.2f}%")

    # Calculate overall metrics for the full dataset
    overall_metrics = ['Specificity', 'Sensitivity', 'Precision', 'Balanced Accuracy']
    overall_results = {metric: 0 for metric in overall_metrics}

    tp = ((out_df['pcr_pred'] == 1) & (out_df['pcr_label'] == 1)).sum()
    tn = ((out_df['pcr_pred'] == 0) & (out_df['pcr_label'] == 0)).sum()
    fp = ((out_df['pcr_pred'] == 1) & (out_df['pcr_label'] == 0)).sum()
    fn = ((out_df['pcr_pred'] == 0) & (out_df['pcr_label'] == 1)).sum()

    overall_results['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    overall_results['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    overall_results['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    overall_results['Balanced Accuracy'] = (overall_results['Sensitivity'] + overall_results['Specificity']) / 2

    print("Overall metrics:")
    for metric, value in overall_results.items():
        print(f"  {metric}: {value * 100:.2f}%")

if __name__ == "__main__":
    predDir = r"nnUNet_results\Dataset200_global_local_4ch_breast\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_4_pcr_transformer\outputs\pred_PCR_cropped" 
    scorePCR(predDir)