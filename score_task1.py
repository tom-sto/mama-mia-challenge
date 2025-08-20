import SimpleITK as sitk
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from MAMAMIA.src.challenge.metrics import compute_segmentation_metrics

og_coords = {}
data_dir = os.environ.get('MAMAMIA_DATA')

# ----------------------------
# Utility Functions
# ----------------------------
def plot_combined_segmentation_heatmaps(fairness_varibles_df, variable_1, variable_2, variable_3, name_1, name_2, name_3, 
                                        output_plot,  metric='DSC', cmap='coolwarm'):
    """
    Generate two side-by-side heatmaps visualizing the average metric across different demographic groups.

    Parameters:
        fairness_varibles_df (DataFrame): Dataframe with patient-wise metrics and demographic variables
        variable_1, variable_2, variable_3 (str): Column names for group variables
        name_1, name_2, name_3 (str): Pretty names for heatmap axes
        output_plot (str): Path to save the resulting figure
        metric (str): The metric to visualize (e.g., 'DSC', 'NormHD')
        cmap (str): Colormap
    """
    # Heatmap 1: Average DSC by Age and Breast Density
    pivot_metric_v12 = fairness_varibles_df.pivot_table(index=variable_1, columns=variable_2, values=metric, aggfunc='mean')
    # Heatmap 2: Average DSC by Age and Menopausal Status
    pivot_metric_v13 = fairness_varibles_df.pivot_table(index=variable_1, columns=variable_3, values=metric, aggfunc='mean')
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # Plot Heatmap 1
    sns.heatmap(pivot_metric_v12, annot=True, cmap=cmap, fmt=".3f", cbar_kws={'label': f'Average {metric}'}, ax=axes[0])
    axes[0].set_title(f'Average {metric} by {name_1} and {name_2}')
    axes[0].set_ylabel(f'{name_1} Group')
    axes[0].set_xlabel(f'{name_2}')
    # Plot Heatmap 2
    sns.heatmap(pivot_metric_v13, annot=True, cmap=cmap, fmt=".3f", cbar_kws={'label': f'Average {metric}'}, ax=axes[1])
    axes[1].set_title(f'Average {metric} by {name_1} and {name_3}')
    axes[1].set_ylabel(f'{name_1} Group')
    axes[1].set_xlabel(f'{name_3}')
    # Invert the order of the y axis values
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[1].invert_xaxis()
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save the combined figure
    plt.savefig(output_plot)
    plt.close()

def getBoundingBox(imagePath: str, patient_id: str, patientInfo: dict):
    assert imagePath.endswith(".nii.gz"), f"Bad image path ending: {imagePath}"
    image = sitk.ReadImage(imagePath)
    
    boundingBox = patientInfo["primary_lesion"]["breast_coordinates"]
    
    zmin = boundingBox["x_min"]
    ymin = boundingBox["y_min"]
    xmin = boundingBox["z_min"]
    zmax = boundingBox["x_max"]
    ymax = boundingBox["y_max"]
    xmax = boundingBox["z_max"]
    
    # Store original coordinates and size for uncropping
    og_coords[patient_id] = {
        "original_size": image.GetSize(),
        "bounding_box": {
            "x_min": xmin, "x_max": xmax,
            "y_min": ymin, "y_max": ymax,
            "z_min": zmin, "z_max": zmax
        }
    }
    # print(og_coords)

def uncropToOriginalCoords(cropped_image_path: str, patient_id: str):
    cropped_image = sitk.ReadImage(cropped_image_path)
    # print(cropped_image.GetSize())
    original_size = og_coords[patient_id]["original_size"]
    bounding_box = og_coords[patient_id]["bounding_box"]

    # Create a blank image with the original size
    original_image = sitk.Image(original_size, cropped_image.GetPixelID())
    original_image.SetSpacing(cropped_image.GetSpacing())
    original_image.SetOrigin(cropped_image.GetOrigin())
    original_image.SetDirection(cropped_image.GetDirection())

    # Place the cropped image back into the original coordinate space
    x_min, x_max = bounding_box["x_min"], bounding_box["x_max"]
    y_min, y_max = bounding_box["y_min"], bounding_box["y_max"]
    z_min, z_max = bounding_box["z_min"], bounding_box["z_max"]

    original_array = sitk.GetArrayFromImage(original_image)
    cropped_array = sitk.GetArrayFromImage(cropped_image)

    original_array[z_min:z_max, y_min:y_max, x_min:x_max] = cropped_array

    uncropped_image = sitk.GetImageFromArray(original_array)
    uncropped_image.CopyInformation(original_image)

    return uncropped_image

def doUncropping(inpDir: str):
    croppedDir = os.path.join(inpDir, 'pred_segmentations_cropped')
    uncroppedDir = os.path.join(inpDir, 'pred_segmentations')
    os.makedirs(uncroppedDir, exist_ok=True)
    for imgName in os.listdir(croppedDir):
        if not imgName.endswith('.nii.gz'):
            continue
        patientID = imgName.split('.')[0]
        patientInfoPath = os.path.join(f'{data_dir}/patient_info_files', f'{patientID}.json')
        with open(patientInfoPath, 'r') as f:
            patientInfo = json.load(f)
        imgPath = os.path.join(croppedDir, imgName)
        ogImgPath = os.path.join(f'{data_dir}/images', patientID.upper(), f'{patientID}_0000.nii.gz')
        getBoundingBox(ogImgPath, patientID, patientInfo)
        uncroppedImg = uncropToOriginalCoords(imgPath, patientID)
        sitk.WriteImage(uncroppedImg, os.path.join(uncroppedDir, f'{patientID}.nii.gz'))

def generate_scores(data_dir: str, testingDir: str, forCorrupted: bool = False):
    # Settings
    HD_MAX = 150
    alpha = 0.5  # Weight for balancing performance and fairness
    selected_fairness_variables = ['age', 'menopause', 'breast_density']
    # The challenge will also evaluate the breast density variable, but it is not included in all the training data

    # Define paths (modify as needed)
    clinical_data_xlsx = f'{data_dir}/clinical_and_imaging_info.xlsx' # Path to the clinical data
    gt_segmentations = f'{data_dir}/segmentations/expert' # Path to the ground truth expert segmentations
    json_info_files = f'{data_dir}/patient_info_files' # Path to the patient JSON info files
    pred_segmentations = f'{testingDir}/pred_segmentations' # Path to your predicted segmentations
    output_csv = f'{testingDir}/results_task1.csv'
    output_plots_dir = f'{testingDir}/plots'

    # Read clinical data and get the fairness groups
    clinical_df = pd.read_excel(clinical_data_xlsx, sheet_name='dataset_info')
    # For fairness_varibles_df, we will drop all the clinical_df columns except the selected_fairness_variables and patient_id
    fairness_varibles_df = clinical_df[['patient_id'] + selected_fairness_variables]
    # Modify age column values mapping them by age groups
    fairness_varibles_df['age'] = pd.cut(fairness_varibles_df['age'], bins=[0, 40, 50, 60, 70, 100], labels=['0-40', '41-50', '51-60', '61-70', '71+'])
    # Map the menopausal status values to 'pre', 'post', and 'unknown'
    fairness_varibles_df['menopause'] = fairness_varibles_df['menopause'].fillna('unknown')
    fairness_varibles_df['menopause'] = fairness_varibles_df['menopause'].apply(lambda x: 'pre' if 'peri' in x else x)
    fairness_varibles_df['menopause'] = fairness_varibles_df['menopause'].apply(lambda x: 'post' if 'post' in x else x)
    fairness_varibles_df['menopause'] = fairness_varibles_df['menopause'].apply(lambda x: 'pre' if 'pre' in x else x)

    # Create output directories if they do not exist
    os.makedirs(pred_segmentations, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    # Read clinical data
    dice_scores = []
    hausdorff_distances = []
    pred_patient_list = list(os.listdir(pred_segmentations))
    pred_patient_list = [x.split(".")[0] for x in pred_patient_list if ".nii.gz" in x]

    for idx, patient_id in enumerate(pred_patient_list):
        print(f'Processing patient {idx + 1}/{len(pred_patient_list)}: {patient_id}')
        # Read the segmentation files
        if forCorrupted:
            gt_file = os.path.join(gt_segmentations, f'{patient_id[:-2]}.nii.gz')
            augmentation = patient_id[-2:]
        else:
            gt_file = os.path.join(gt_segmentations, f'{patient_id}.nii.gz')
        # Read it with SimpleITk and convert to numpy
        itk_image = sitk.ReadImage(gt_file)
        gt_mask = sitk.GetArrayFromImage(itk_image)

        # Read the predicted segmentation
        pred_file = os.path.join(pred_segmentations, f'{patient_id}.nii.gz')
        # Read it with SimpleITk and convert to numpy
        itk_image = sitk.ReadImage(pred_file)
        pred_mask = sitk.GetArrayFromImage(itk_image)
        
        metrics = compute_segmentation_metrics(gt_mask, pred_mask, hd_max=HD_MAX)
        patient_id = str(patient_id).upper()
        if forCorrupted:
            fairness_varibles_df.loc[fairness_varibles_df['patient_id']==patient_id[:-2], f'DSC{augmentation}'] = metrics['DSC']
            fairness_varibles_df.loc[fairness_varibles_df['patient_id']==patient_id[:-2], f'NormHD{augmentation}'] = metrics['NormHD']
        else:
            fairness_varibles_df.loc[fairness_varibles_df['patient_id']==patient_id, 'DSC'] = metrics['DSC']
            fairness_varibles_df.loc[fairness_varibles_df['patient_id']==patient_id, 'NormHD'] = metrics['NormHD']
        dice_scores.append(metrics['DSC'])
        hausdorff_distances.append(metrics['NormHD'])

    print(f'Average Dice: {np.mean(dice_scores):.4f}')
    print(f'Average Hausdorff distance: {np.mean(hausdorff_distances):.4f}')
    # Export results
    fairness_varibles_df.to_csv(output_csv, index=False)

    # Compute performance score (combining Dice and NormHD)
    performance_score = 0.5 * (np.mean(dice_scores) + (1 - np.mean(hausdorff_distances)))
    print(f'Performance score: {performance_score:.4f}')

    if forCorrupted:
        print("Not doing plots for corrupted dataset. Finishing up...")
        return
    
    upper_pred_patient_ids = [x.upper() for x in pred_patient_list]
    fairness_varibles_df = fairness_varibles_df[fairness_varibles_df['patient_id'].isin(upper_pred_patient_ids)]

    # Compute fairness score across selected variables
    fairness_score_dict = {}
    for variable in selected_fairness_variables:
        # Split the fairness_varibles_df into groups based on the values of the column 'variable'
        groups = fairness_varibles_df.groupby(variable)
        # Initialize lists to store group-level metrics
        dice_scores_groups = []
        norm_hd_scores_groups = []
        # Compute group-level averages for Dice and Hausdorff Distance
        for group in groups:
            group_name = group[0]
            avg_dice = group[1]['DSC'].mean()
            avg_norm_hd = group[1]['NormHD'].mean()
            dice_scores_groups.append(avg_dice)
            norm_hd_scores_groups.append(avg_norm_hd)
        # Compute disparities: max - min across groups
        dice_disparity = max(dice_scores_groups) - min(dice_scores_groups)
        normalized_hd_disparity = max(norm_hd_scores_groups) - min(norm_hd_scores_groups)
        fairness_score = 1 - 0.5 * (dice_disparity + normalized_hd_disparity)
        # Append fairness variable and disparity to the dictionary
        fairness_score_dict[variable] = fairness_score
    
    avg_fairness_score = sum([fairness_score_dict[variable] for variable in selected_fairness_variables])
    avg_fairness_score = avg_fairness_score/len(selected_fairness_variables)
    print(f'Average fairness score: {avg_fairness_score:.4f}')

    # Final ranking score: combination of performance and fairness
    ranking_score = (1 - alpha)*performance_score + alpha*avg_fairness_score
    print(f'Ranking score: {ranking_score:.4f}')

    # Save fairness heatmaps
    output_plot = os.path.join(output_plots_dir, 'heatmap_dsc_combined.png')
    plot_combined_segmentation_heatmaps(fairness_varibles_df, 'age', 'breast_density', 'menopause', 'Age', 'Breast Density',
                                        'Menopausal Status', output_plot,  metric='DSC', cmap='coolwarm_r') 
    output_plot = os.path.join(output_plots_dir, 'heatmap_normhd_combined.png')
    plot_combined_segmentation_heatmaps(fairness_varibles_df, 'age', 'breast_density', 'menopause', 'Age', 'Breast Density',
                                        'Menopausal Status', output_plot,  metric='NormHD', cmap='coolwarm')

def evaluateAcrossDatasets(inpDir: str):
    import pandas as pd

    # Load your CSV file into a DataFrame
    df = pd.read_csv(f'{inpDir}/results_task1.csv')

    # Filter and calculate average DSC for each group
    groups = ["DUKE", "ISPY1", "ISPY2", "NACT"]
    average_dsc = {}
    average_hd = {}

    for group in groups:
        # Filter rows where patient_id contains the group and ignore NaN values in DSC
        group_df = df[df['patient_id'].str.contains(group, na=False)]
        average_dsc[group] = group_df['DSC'].mean()
        average_hd[group] = group_df['NormHD'].mean()

    # Print the results
    for group, avg in average_dsc.items():
        print(f"Average DSC for {group}: {avg}")

    print("Overall Average DSC:", df['DSC'].dropna().mean())

    for group, avg in average_hd.items():
        print(f"Average HD for {group}: {avg}")

    print("Overall Average HD:", df['NormHD'].dropna().mean())

    performance_score = 0.5 * (df['DSC'].dropna().mean() + (1 - df['NormHD'].dropna().mean()))
    print(f'Performance score: {performance_score:.4f}')

    # Compute fairness score across selected variables
    selected_fairness_variables = ['age', 'menopause', 'breast_density']
    fairness_score_dict = {}
    for variable in selected_fairness_variables:
        # Split the fairness_varibles_df into groups based on the values of the column 'variable'
        groups = df.groupby(variable)
        # Initialize lists to store group-level metrics
        dice_scores_groups = []
        norm_hd_scores_groups = []
        # Compute group-level averages for Dice and Hausdorff Distance
        for group in groups:
            group_name = group[0]
            avg_dice = group[1]['DSC'].mean()
            avg_norm_hd = group[1]['NormHD'].mean()
            dice_scores_groups.append(avg_dice)
            norm_hd_scores_groups.append(avg_norm_hd)
        # Compute disparities: max - min across groups
        dice_disparity = max(dice_scores_groups) - min(dice_scores_groups)
        normalized_hd_disparity = max(norm_hd_scores_groups) - min(norm_hd_scores_groups)
        fairness_score = 1 - 0.5 * (dice_disparity + normalized_hd_disparity)
        # Append fairness variable and disparity to the dictionary
        fairness_score_dict[variable] = fairness_score
    
    avg_fairness_score = sum([fairness_score_dict[variable] for variable in selected_fairness_variables])
    avg_fairness_score = avg_fairness_score/len(selected_fairness_variables)
    print(f'Average fairness score: {avg_fairness_score:.4f}')
    alpha = 0.5
    # Final ranking score: combination of performance and fairness
    ranking_score = (1 - alpha)*performance_score + alpha*avg_fairness_score
    print(f'Ranking score: {ranking_score:.4f}')

def doScoring(inpDir: str, corrupted: bool = False):
    doUncropping(inpDir)
    generate_scores(data_dir, inpDir, forCorrupted=corrupted)
    evaluateAcrossDatasets(inpDir)

if __name__ == "__main__":
    inpDir = r"nnUNet_results\Dataset104_cropped_3ch_breast\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_4"
    # doUncropping(inpDir)
    # generate_scores(data_dir, inpDir)
    evaluateAcrossDatasets(inpDir)

