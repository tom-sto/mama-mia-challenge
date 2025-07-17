import SimpleITK as sitk
import os
import json
from MAMAMIA.src.challenge.scoring_task1 import generate_scores

og_coords = {}

data_dir = os.environ.get('MAMAMIA_DATA')

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
