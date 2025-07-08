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

    print("Percentage of correct predictions:",
          out_df['correct'].mean() * 100, "%")
    groups = ['DUKE', 'ISPY1', 'ISPY2', 'NACT']
    avg_accuracy = {group : out_df[out_df['patient_id'].str.contains(group, na=False)]['correct'].astype(int).mean() \
                    for group in groups}
    print("Average accuracy per group:")
    for group, accuracy in avg_accuracy.items():
        print(f"{group}: {accuracy * 100:.2f}%")

if __name__ == "__main__": 
    ...