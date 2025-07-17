import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def scorePCR(predictionsDF: str,
             threshold: float = 0.5,
             clinicalInfoPath: str = rf"{os.environ["MAMAMIA_DATA"]}/clinical_and_imaging_info.xlsx"):
    df = pd.read_excel(clinicalInfoPath, sheet_name="dataset_info")

    predDF = pd.read_csv(predictionsDF)

    # Update scoring script to use 'pcr_prob' for predictions
    predDF['pcr_pred'] = (predDF['pcr_prob'] > threshold).astype(int)
    predDF['pcr_label'] = predDF['patient_id'].apply(lambda pid: df.loc[df['patient_id'] == pid, 'pcr'].values[0])
    predDF['correct'] = (predDF['pcr_pred'] == predDF['pcr_label']).astype(int)

    predDF.to_csv(predictionsDF, index=False)

    # predDF['pcr_pred'] = 0
    # predDF['correct'] = predDF['pcr_label'] == predDF['pcr_pred']

    print("Percentage of correct predictions:",
          predDF['correct'].mean() * 100, "%")
    groups = ['DUKE', 'ISPY1', 'ISPY2', 'NACT']
    avg_accuracy = {group : predDF[predDF['patient_id'].str.contains(group, na=False)]['correct'].astype(int).mean() \
                    for group in groups}
    print("Average accuracy per group:")
    for group, accuracy in avg_accuracy.items():
        print(f"{group}: {accuracy * 100:.2f}%")

    # Calculate Specificity, Sensitivity, Recall, and Precision for each dataset group
    # Include both Precision and Balanced Accuracy calculations
    metrics = ['Specificity', 'Sensitivity', 'Precision', 'Balanced Accuracy']
    results = {group: {metric: 0 for metric in metrics} for group in groups}

    for group in groups:
        group_df = predDF[predDF['patient_id'].str.contains(group, na=False)]
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

    tp = ((predDF['pcr_pred'] == 1) & (predDF['pcr_label'] == 1)).sum()
    tn = ((predDF['pcr_pred'] == 0) & (predDF['pcr_label'] == 0)).sum()
    fp = ((predDF['pcr_pred'] == 1) & (predDF['pcr_label'] == 0)).sum()
    fn = ((predDF['pcr_pred'] == 0) & (predDF['pcr_label'] == 1)).sum()

    overall_results['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    overall_results['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    overall_results['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    overall_results['Balanced Accuracy'] = (overall_results['Sensitivity'] + overall_results['Specificity']) / 2

    print("Overall metrics:")
    for metric, value in overall_results.items():
        print(f"  {metric}: {value * 100:.2f}%")

    # Generate ROC curve by evaluating predictions across thresholds
    thresholds = np.linspace(0, 1, 1000)
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        predDF['pcr_pred'] = (predDF['pcr_prob'] > threshold).astype(int)
        tp = ((predDF['pcr_pred'] == 1) & (predDF['pcr_label'] == 1)).sum()
        tn = ((predDF['pcr_pred'] == 0) & (predDF['pcr_label'] == 0)).sum()
        fp = ((predDF['pcr_pred'] == 1) & (predDF['pcr_label'] == 0)).sum()
        fn = ((predDF['pcr_pred'] == 0) & (predDF['pcr_label'] == 1)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    roc_auc = auc(fpr_list, tpr_list)

    # Print optimal threshold
    optimal_idx = np.argmax(np.array(tpr_list) - np.array(fpr_list))
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold}")
    print(f"AUC:", roc_auc)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(os.path.join(os.path.dirname(predictionsDF), 'ROC.png'))

if __name__ == "__main__":
    predDFpath = r"nnUNet_results\Dataset200_global_local_4ch_breast\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_4_pcr_transformer_more_classifier\outputs\pcr_predictions.csv" 
    scorePCR(predDFpath, threshold=0.4444444)
    from MAMAMIA.src.challenge.scoring_task2 import doScoring
    doScoring(os.path.dirname(predDFpath))