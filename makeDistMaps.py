import blosc2
import numpy as np
from scipy.ndimage import distance_transform_edt
import os
import SimpleITK as sitk

if __name__ == "__main__":

    preprocessed_dir = r"E:\MAMA-MIA\nnUNet_preprocessed_data\Dataset104_cropped_3ch_breast\nnUNetPlans_3d_fullres"

    cparams = {
            'codec': 8,
            # 'filters': [blosc2.Filter.SHUFFLE],
            # 'splitmode': blosc2.SplitMode.ALWAYS_SPLIT,
            'clevel': blosc2.Codec.ZSTD,
        }
    
    for filename in os.listdir(preprocessed_dir):
        if filename.endswith("seg.b2nd"):
            # print(f"Looking at {filename}")
            mask_path = os.path.join(preprocessed_dir, filename)
            mask: blosc2.ndarray.NDArray = blosc2.open(mask_path, mode='r')
            mask = blosc2.asarray(mask).squeeze()
            # print(type(mask))
            # print(np.unique(mask))
            # print(mask.shape)
            # sitk.WriteImage(sitk.GetImageFromArray(mask), 'mask.mha')
            mask = np.where(mask == -1, 0, mask)
            # print(np.unique(mask))
            # print(mask.shape)
            # sitk.WriteImage(sitk.GetImageFromArray(mask), 'mask-unfilled.mha')
            edt = distance_transform_edt(mask)
            # print(np.unique(edt))
            inv = distance_transform_edt(1 - mask.astype(np.int_))
            # print(np.unique(inv))
            sdm: np.ndarray = inv - edt
            # print(np.unique(sdm))
            # print(sdm.shape)
            # sitk.WriteImage(sitk.GetImageFromArray(sdm), 'sdm.mha')

            # sitk.WriteImage(sitk.GetImageFromArray(sdm * mask), 'smd-masked.mha')
            output_path = os.path.join(preprocessed_dir, filename.replace("seg", "dist"))
            blosc2.asarray(np.ascontiguousarray(sdm), urlpath=output_path, mmap_mode='w+')
            print(f"Processed {filename} -> {output_path}")