import SimpleITK as sitk
import os
import pandas as pd

def scorePCR(outputDir: str,
             clinicalInfoPath: str = rf"{os.environ["MAMAMIA_DATA"]}/clinical_and_imaging_info.xlsx"):
    df = pd.read_excel(clinicalInfoPath, sheet_name="dataset_info")

    out_df = pd.DataFrame(columns=['patient_id', 'pcr_pred', 'pcr_label', 'correct'])
    
    for imgName in os.listdir(outputDir):
        if not imgName.endswith('.nii.gz'):
            continue
        img = sitk.ReadImage(os.path.join(outputDir, imgName))
        # print(img.GetSize())
        pcr_pred = sitk.GetArrayFromImage(img)[0, 0, 0]
        patient_id = imgName.split('.')[0].upper()
        pcr_label = df.loc[df['patient_id'] == patient_id, 'pcr'].values[0]

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