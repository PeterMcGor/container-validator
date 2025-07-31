import json
import os
import shutil
import subprocess
import tempfile
from typing import Tuple

import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join, save_json


import json
import os
import pandas as pd

def read_json_as_dict(jsonFile: str) -> dict:
    with open(jsonFile) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    return jsonObject

class nnUNetSumReader:
    def __init__(self, jsonFile: str):
        self.json_file = jsonFile
             
    @property
    def json_file(self):
        return self._json_file
 
    @json_file.setter
    def json_file(self, jsonFile):
        assert os.path.exists(jsonFile), f"File {jsonFile} does not exist"
        self._json_file = jsonFile
        self.json_dict = read_json_as_dict(self.json_file)
        
        # Check for new JSON structure (has metric_per_case)
        if 'metric_per_case' in self.json_dict.keys():
            self._parse_new_format()
        # Check for old JSON structure (has results)
        elif 'results' in self.json_dict.keys():
            self._parse_old_format()
        else:
            raise ValueError("Unknown JSON format: expected either 'metric_per_case' or 'results' key")
    
    def _parse_new_format(self):
        """Parse the new nnUNet evaluation format"""
        self.results = []
        metric_per_case = self.json_dict['metric_per_case']
        
        for case_data in metric_per_case:
            # Extract file paths
            ref_file = case_data.get('reference_file', '')
            pred_file = case_data.get('prediction_file', '')
            
            # Extract metrics for each label
            metrics_dict = case_data.get('metrics', {})
            for label, metrics in metrics_dict.items():
                case_result = {
                    'ref': ref_file,
                    'tst': pred_file,
                    'label': label
                }
                # Add all metric values
                case_result.update(metrics)
                self.results.append(case_result)
    
    def _parse_old_format(self):
        """Parse the old nnUNet evaluation format"""
        results_dct = self.json_dict['results']['all']
        self.results = []
        
        for evaluation in results_dct:
            ref_tst = {'ref': evaluation['reference'], 'tst': evaluation['test']}
            evaluation_labels = evaluation.copy()
            [evaluation_labels.pop(k) for k in ['reference', 'test']]
            
            for label, metrics in evaluation_labels.items():
                ref_tst['label'] = label
                self.results.append({**ref_tst, **{m: val for m, val in metrics.items()}})
                     
    def get_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)
                     
    def get_csv(self, csv_path):
        self.get_data_frame().to_csv(csv_path, index=False)
             
    def __str__(self):
        return f"{self.json_file} (Results: {len(self.results)})"

    def get_summary_stats(self):
        """Get summary statistics from the JSON (if available)"""
        summary = {}
        if 'foreground_mean' in self.json_dict:
            summary['foreground_mean'] = self.json_dict['foreground_mean']
        if 'mean' in self.json_dict:
            summary['mean'] = self.json_dict['mean']
        return summary
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Convert summary statistics to DataFrame format"""
        summary_results = []
        
        # Add foreground_mean metrics if available
        if 'foreground_mean' in self.json_dict:
            foreground_data = self.json_dict['foreground_mean'].copy()
            foreground_data['metric_type'] = 'foreground_mean'
            foreground_data['label'] = 'all'
            summary_results.append(foreground_data)
        
        # Add mean metrics by label if available
        if 'mean' in self.json_dict:
            mean_data = self.json_dict['mean']
            for label, metrics in mean_data.items():
                label_data = metrics.copy()
                label_data['metric_type'] = 'mean'
                label_data['label'] = label
                summary_results.append(label_data)
        
        return pd.DataFrame(summary_results)
    
    def get_summary_csv(self, csv_path):
        """Export summary statistics to CSV"""
        summary_df = self.get_summary_dataframe()
        summary_df.to_csv(csv_path, index=False)
        
    def export_both_csvs(self, per_case_csv_path, summary_csv_path):
        """Export both per-case results and summary statistics to separate CSVs"""
        # Export per-case results
        self.get_csv(per_case_csv_path)
        
        # Export summary statistics
        self.get_summary_csv(summary_csv_path)

def cast_label_image_to_int(label_image: sitk.Image) -> sitk.Image:
    """
    Cast a SimpleITK label image to an appropriate integer type.

    This function determines the most suitable integer type for the label image
    based on its minimum and maximum values, and casts the image to that type.

    :param label_image: The input label image to be cast.
    :type label_image: sitk.Image
    :return: The label image cast to an appropriate integer type.
    :rtype: sitk.Image
    """
    min_value = sitk.GetArrayViewFromImage(label_image).min()
    max_value = sitk.GetArrayViewFromImage(label_image).max()

    if min_value >= 0:
        if max_value <= 255:
            cast_type = sitk.sitkUInt8
        elif max_value <= 65535:
            cast_type = sitk.sitkUInt16
        else:
            cast_type = sitk.sitkUInt32
    else:
        if min_value >= -128 and max_value <= 127:
            cast_type = sitk.sitkInt8
        elif min_value >= -32768 and max_value <= 32767:
            cast_type = sitk.sitkInt16
        else:
            cast_type = sitk.sitkInt32

    return sitk.Cast(label_image, cast_type)


def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None, reference: str = None, release: str = None, license: str = None,
                          description: str = None,
                          overwrite_image_reader_writer: str = None, **kwargs):
    """
    Generates a dataset.json file in the output folder

    Parameters
    ----------
    output_folder : str
        folder where the dataset.json should be saved
    channel_names : dict
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.
    labels : dict
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }
        
        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!
    num_training_cases : int
        is used to double check all cases are there!
    file_ending : str
        needed for finding the files correctly. IMPORTANT! File endings must match between images and
        segmentations!
    regions_class_order : Tuple[int, ...]
        If you have defined regions (see above), you need to specify the order in which they should be
        processed. This is important because it determines the color in the 2d/3d visualizations.
    dataset_name : str, optional
        dataset name, by default None
    reference : str, optional
        reference, by default None
    release : str, optional
        release, by default None
    license : str, optional
        license, by default None
    description : str, optional
        description, by default None
    overwrite_image_reader_writer : str, optional
        If you need a special IO class for your dataset you can derive it from
        BaseReaderWriter, place it into nnunet.imageio and reference it here by name
    **kwargs
        whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
                                                f"You need that."
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        'channel_names': channel_names,  # previously this was called 'modality'. I didn't like this so this is
        # channel_names now. Live with it.
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if license is not None:
        dataset_json['licence'] = license
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)



class SCRIPTS:
    dockerName = 'petermcgor/ants:2.3.1'
    registrationSyN = 'antsRegistrationSyN.sh'
    applyTransform = 'antsApplyTransforms'
    dockerSynthStrip = 'freesurfer/synthstrip'

imgs_folder_dck = '/data'
out_folder_dck = '/out' #imgs_folder_dck if out_folder is None else '/out'

def synth_strip(imgs_folder:str, 
                input_img:str, 
                out_folder:str, 
                out_img:str=None, 
                b:int=1, 
                save_brain_mask = False):
    
    """
    Use FreeSurfer's SynthStrip to skull strip an image. Requires docker and a GPU.

    Parameters
    ----------
    imgs_folder : str
        Folder where the input image is located.
    input_img : str
        Input image to skull strip.
    out_folder : str
        Folder where the output image will be saved.
    out_img : str, optional
        Output image name. If None, the same as the input image is used.
    b : int, optional
        Brain extraction threshold. Default is 1.

    Returns
    -------
    None
    """
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    out_img = input_img if out_img is None else out_img
    brain_mask = out_img.split('.')[0]+'_brain-mask.nii.gz'

    # Sometimes dockers mounts (config) can (should) struggle with symbolic links or already mounted filesystems
    # Working on the temp should help but there is a overload moving data
    tmp_dir = tempfile.TemporaryDirectory() 
    shutil.copy(os.path.join(imgs_folder, input_img), tmp_dir.name)
    
    subprocess.run(['docker', 'run','-v', tmp_dir.name+':'+imgs_folder_dck, '-v', tmp_dir.name+':'+out_folder_dck, '--gpus', 'device=0',  
                    SCRIPTS.dockerSynthStrip, 
                    '-i', os.path.join(imgs_folder_dck, input_img), 
                    '-o', os.path.join(out_folder_dck, out_img), 
                    '-m',os.path.join(out_folder_dck, brain_mask),
                    '-b', str(b)])
    shutil.copy(os.path.join(tmp_dir.name, out_img), out_folder)
    if save_brain_mask:
        shutil.copy(os.path.join(tmp_dir.name, brain_mask), out_folder)
    os.remove(os.path.join(tmp_dir.name, input_img))
    
    #remove out


def batch_synth_strip_shared_mask(imgs_folder: str,
                                  input_imgs: list,
                                  out_folder: str,
                                  out_imgs: list = None,
                                  b: int = 1, 
                                  save_brain_mask = False):
    """
    Apply skull stripping to a list of images using a shared mask from the first image.
    
    Parameters
    ----------
    imgs_folder : str
        Folder where the input images are located.
    input_imgs : list
        List of input image filenames to skull strip.
    out_folder : str
        Folder where the output images will be saved.
    out_imgs : list, optional
        List of output image names. If None, same names as input images are used.
    b : int, optional
        Brain extraction threshold for the first image. Default is 1.
        
    Returns
    -------
    None
    """
    if not input_imgs:
        raise ValueError("input_imgs list cannot be empty")
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Set output names if not provided
    if out_imgs is None:
        out_imgs = input_imgs.copy()
    elif len(out_imgs) != len(input_imgs):
        raise ValueError("out_imgs must have the same length as input_imgs")
    
    # Create temporary directory for mask storage
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Processing first image: {input_imgs[0]}")
        
        # Apply synth_strip to the first image and save the brain mask
        synth_strip(imgs_folder=imgs_folder,
                   input_img=input_imgs[0],
                   out_folder=tmp_dir,
                   out_img=out_imgs[0],
                   b=b,
                   save_brain_mask=save_brain_mask)
        
        # Copy the first processed image to output folder
        shutil.copy(os.path.join(tmp_dir, out_imgs[0]), out_folder)
        
        # Load the brain mask from the first image
        mask_filename = out_imgs[0].split('.')[0] + '_brain-mask.nii.gz'
        mask_path = os.path.join(tmp_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Brain mask not found: {mask_path}")
            
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        
        print(f"Loaded brain mask: {mask_filename}")
        
        # Apply the same mask to the remaining images
        for i, (input_img, out_img) in enumerate(zip(input_imgs[1:], out_imgs[1:]), 1):
            print(f"Processing image {i+1}/{len(input_imgs)}: {input_img}")
            
            # Load the current image
            img_path = os.path.join(imgs_folder, input_img)
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}. Skipping...")
                continue
                
            img = nib.load(img_path)
            img_data = img.get_fdata()
            
            # Check if image and mask have compatible dimensions
            if img_data.shape != mask_data.shape:
                print(f"Warning: Image {input_img} has shape {img_data.shape} "
                      f"but mask has shape {mask_data.shape}. Skipping...")
                continue
            
            # Apply the mask (element-wise multiplication)
            masked_data = img_data * mask_data
            
            # Create new NIfTI image with the masked data
            masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)
            
            # Save the masked image
            output_path = os.path.join(out_folder, out_img)
            nib.save(masked_img, output_path)
            print(f"Saved masked image: {output_path}")
        
        # Optionally save the brain mask to the output folder
        mask_output_path = os.path.join(out_folder, mask_filename)
        shutil.copy(mask_path, mask_output_path)
        print(f"Saved brain mask: {mask_output_path}")



    
     

