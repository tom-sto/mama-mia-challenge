import matplotlib.pyplot as plt
import pandas as pd
import pdb
import warnings
from PIL import Image
import os

EXCEL_PATH = r"E:\MAMA-MIA\clinical_and_imaging_info.xlsx"

def main():
    warnings.filterwarnings("ignore")
    df = pd.read_excel(EXCEL_PATH)

    groups = ["DUKE", "ISPY1", "ISPY2", "NACT"]


    for group in groups:
        # Filter rows where patient_id contains the group and ignore NaN values in DSC
        group_df = df[df['patient_id'].str.contains(group, na=False)]

        # Patient age distribution as a histogram
        plt.figure()
        group_df['age'] = group_df['age'].fillna(-1)
        nbins = 20
        if -1 in group_df['age'].values:
            nbins = 40
        group_df['age'].hist(bins=nbins)
        plt.title(f'{group} - Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.savefig(f'{group}_age_distribution.png')

        # Menopausal status as a pie chart
        plt.figure()
        group_df['menopause'] = group_df['menopause'].apply(lambda x: 'pre' if 'peri' in str(x) else x)
        group_df['menopause'] = group_df['menopause'].apply(lambda x: 'post' if 'post' in str(x) else x)
        group_df['menopause'] = group_df['menopause'].apply(lambda x: 'pre' if 'pre' in str(x) else ('post' if 'post' in str(x) else 'unknown'))
        menopause_counts = group_df['menopause'].value_counts()

        # Sort categories by count in descending order
        menopause_counts = menopause_counts.sort_values(ascending=False)

        # Check if all values are 'unknown' and set pie chart color to light grey
        if menopause_counts.index.tolist() == ['unknown']:
            menopause_counts.plot.pie(labels=menopause_counts.index, colors=['lightgrey'], autopct=lambda p: f'{int(p * menopause_counts.sum() / 100)}', ylabel='')
        else:
            menopause_counts.plot.pie(labels=menopause_counts.index, autopct=lambda p: f'{round(p * menopause_counts.sum() / 100)}', ylabel='')
        plt.title(f'{group} - Menopausal Status')
        plt.savefig(f'{group}_menopausal_status.png')

        # Breast density label as a pie chart
        plt.figure()
        group_df['breast_density'] = group_df['breast_density'].fillna('unknown')
        breast_density_counts = group_df['breast_density'].value_counts()

        # Sort categories by count in descending order
        breast_density_counts = breast_density_counts.sort_values(ascending=False)

        # Check if all values are 'unknown' and set pie chart color to light grey
        if breast_density_counts.index.tolist() == ['unknown']:
            breast_density_counts.plot.pie(labels=breast_density_counts.index, colors=['lightgrey'], autopct=lambda p: f'{int(p * breast_density_counts.sum() / 100)}', ylabel='')
        else:
            breast_density_counts.plot.pie(labels=breast_density_counts.index, autopct=lambda p: f'{round(p * breast_density_counts.sum() / 100)}', ylabel='')
        plt.title(f'{group} - Breast Density')
        plt.savefig(f'{group}_breast_density.png')

        # Number of phases as a pie chart
        plt.figure()
        group_df['num_phases'] = group_df['num_phases'].fillna('unknown').astype(int)
        num_phases_counts = group_df['num_phases'].value_counts()
        num_phases_counts.plot.pie(labels=num_phases_counts.index, autopct=lambda p: f'{round(p * num_phases_counts.sum() / 100)}', ylabel='')
        plt.title(f'{group} - Number of Phases')
        plt.savefig(f'{group}_num_phases.png')

        # Tumor focality with subcategories
        plt.figure()
        group_df['unifocal'] = ~group_df['multifocal_cancer'].fillna(False).astype(bool)
        group_df['unilateral'] = ~group_df['bilateral_breast_cancer'].fillna(False).astype(bool)

        # Define categories
        categories = {
            'Unilateral-Unifocal': group_df[(group_df['unilateral']) & (group_df['unifocal'])].shape[0],
            'Unilateral-Multifocal': group_df[(group_df['unilateral']) & (~group_df['unifocal'])].shape[0],
            'Bilateral-Unifocal': group_df[(~group_df['unilateral']) & (group_df['unifocal'])].shape[0],
            'Bilateral-Multifocal': group_df[(~group_df['unilateral']) & (~group_df['unifocal'])].shape[0],
            'Unilateral-Unknown Focality': group_df[(group_df['unilateral']) & (group_df['multifocal_cancer'].isna())].shape[0],
            'Bilateral-Unknown Focality': group_df[(~group_df['unilateral']) & (group_df['multifocal_cancer'].isna())].shape[0]
        }

        # Filter categories with count > 0
        filtered_categories = {k: v for k, v in categories.items() if v > 0}

        # Sort tumor focality categories by count in descending order
        filtered_categories = dict(sorted(filtered_categories.items(), key=lambda item: item[1], reverse=True))

        # Create a pie chart
        plt.pie(filtered_categories.values(), labels=filtered_categories.keys(), autopct=lambda p: f'{round(p * sum(filtered_categories.values()) / 100)}')
        plt.title(f'{group} - Tumor Focality and Laterality')
        plt.savefig(f'{group}_tumor_focality_laterality.png')

        # pCR labels as a pie chart
        plt.figure()
        group_df['pcr'] = group_df['pcr'].fillna('unknown')
        pcr_counts = group_df['pcr'].value_counts()

        # Sort categories by count in descending order
        pcr_counts = pcr_counts.sort_values(ascending=False)

        # Check if all values are 'unknown' and set pie chart color to light grey
        if pcr_counts.index.tolist() == ['unknown']:
            pcr_counts.plot.pie(labels=pcr_counts.index, colors=['lightgrey'], autopct='%1.1f%%', ylabel='')
        else:
            pcr_counts.plot.pie(labels=pcr_counts.index, autopct='%1.1f%%', ylabel='')
        plt.title(f'{group} - pCR Labels')
        plt.savefig(f'{group}_pcr_labels.png')

        # Tumor subtype as a pie chart
        plt.figure()
        group_df['tumor_subtype'] = group_df['tumor_subtype'].apply(lambda x: 'luminal' if 'luminal' in str(x).lower() else x).fillna('unknown')
        tumor_subtype_counts = group_df['tumor_subtype'].value_counts()

        # Sort categories by count in descending order
        tumor_subtype_counts = tumor_subtype_counts.sort_values(ascending=False)

        # Check if all values are 'unknown' and set pie chart color to light grey
        if tumor_subtype_counts.index.tolist() == ['unknown']:
            tumor_subtype_counts.plot.pie(labels=tumor_subtype_counts.index, colors=['lightgrey'], autopct=lambda p: f'{int(p * tumor_subtype_counts.sum() / 100)}', ylabel='')
        else:
            tumor_subtype_counts.plot.pie(labels=tumor_subtype_counts.index, autopct=lambda p: f'{round(p * tumor_subtype_counts.sum() / 100)}', ylabel='')
        plt.title(f'{group} - Tumor Subtype')
        plt.savefig(f'{group}_tumor_subtype.png')

        # pdb.set_trace()
        stitch_plots(group)
        print(f"Finished {group}")

def trim_white_background(image, padding):
    # Convert image to grayscale
    gray_image = image.convert('L')

    # Get bounding box of non-white areas
    bbox = gray_image.point(lambda x: 0 if x == 255 else 1, '1').getbbox()

    # Crop image using bounding box
    cropped_image = image.crop(bbox)

    # Add constant padding around the cropped image
    padded_image = Image.new('RGB', (cropped_image.width + 2 * padding, cropped_image.height + 2 * padding), (255, 255, 255))
    padded_image.paste(cropped_image, (padding, padding))

    return padded_image

def stitch_plots(group):
    # Define plot file paths
    plot_files = [
        f'{group}_age_distribution.png',
        f'{group}_menopausal_status.png',
        f'{group}_breast_density.png',
        f'{group}_num_phases.png',
        f'{group}_tumor_focality_laterality.png',
        f'{group}_pcr_labels.png',
        f'{group}_tumor_subtype.png'
    ]

    # Load images
    images = [Image.open(file) for file in plot_files]

    # Define constant padding for pie charts
    padding = 30

    # Trim white background and add constant padding to pie charts
    for i in range(1, len(images)):
        images[i] = trim_white_background(images[i], padding)

    # Create a blank canvas for stitching
    row1_width = sum(image.width for image in images[:3])
    row2_width = sum(image.width for image in images[3:])
    canvas_width = max(row1_width, row2_width)
    canvas_height = images[0].height + images[3].height
    stitched_image = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

    # Paste images into canvas
    x_offset = (canvas_width - row1_width) // 2
    for image in images[:3]:
        stitched_image.paste(image, (x_offset, 0))
        x_offset += image.width

    x_offset = (canvas_width - row2_width) // 2  # Center second row horizontally
    for image in images[3:]:
        stitched_image.paste(image, (x_offset, images[0].height))
        x_offset += image.width

    # Save stitched image
    stitched_image.save(f'stitched_plots_{group}.png')

if __name__ == '__main__':
    main()