import pandas as pd
import zarr
import os
import matplotlib.pyplot as plt
import numpy as np

x = ["Test Patches", "Test Whole Image", "Val Patches", "Val Whole Image"]
x = ["Val Whole Image"]
x = ["TransformerSTWithSkips", "ConvWithSkips"]
for t in x:
    # Path settings
    csv_path = rf"transformerResults\{t}\outputsExtraNegativeSampling-FixedDMapAndShuffleBestSeg\scores.csv"  # Update this to your actual CSV file
    zarr_dir = rf"my_preprocessed_data\Dataset106_cropped_Xch_breast_no_norm\testing"

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Keep only relevant columns
    df = df[['Patient ID', 'Dice (Full Image)']].dropna()

    # Prepare lists to store results
    image_sizes = []
    dice_scores = []

    # Iterate through the DataFrame
    for idx, row in df.iterrows():
        patient_id = str(row['Patient ID'])
        dice_score = row['Dice (Full Image)']
        
        # Construct Zarr file path
        zarr_path = os.path.join(zarr_dir, f"{patient_id}_0000.zarr")

        # Check if Zarr file exists
        if not os.path.exists(zarr_path):
            print(f"Warning: Zarr file not found for Patient ID {patient_id}")
            continue

        try:
            # Open Zarr and get image shape
            z = zarr.open(zarr_path, mode='r')
            # image_shape = z.shape  # e.g., (C, H, W, D) or (H, W, D) — depends on how saved
            image_size = int(z.size)  # total number of voxels
            # Optionally, use a specific dimension, e.g., z.shape[1] * z.shape[2] * z.shape[3]
        except Exception as e:
            print(f"Error reading {zarr_path}: {e}")
            continue

        image_sizes.append(image_size)
        dice_scores.append(dice_score)

    x = np.array(image_sizes)
    y = np.array(dice_scores)

    # Fit a line (degree=1)
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(image_sizes, dice_scores, alpha=0.7)
    plt.plot(x, regression_line, color='red', label='Linear Regression')
    plt.xlabel("Image Size (Total Voxels)")
    plt.ylabel("Dice Score (Full Image)")
    plt.title("Dice Score vs Image Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(csv_path), "Dice vs Size.png"))
    plt.show()

# Optional: Save data to new CSV
# output_df = pd.DataFrame({
#     'Patient ID': df['Patient ID'],
#     'Dice Score': dice_scores,
#     'Image Size': image_sizes
# })
# output_df.to_csv("dice_vs_image_size.csv", index=False)
