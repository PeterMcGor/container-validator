import os
import shutil
import pandas as pd

from task2_segmentation.nnunet_utils.utils import generate_dataset_json, synth_strip

class SEQUENCES:
    MP2RAGE = 'MP2RAGE'
    MPRAGE = 'MPRAGE'
    FLAIR = 'FLAIR'
    T1W = 'T1W'
    DWIB1000 = "dwi_b1000"
    SWI = "swI"
    T2START = 'T2s'

class FOLDERS:
    IMAGES_TRAIN = 'imagesTr'
    IMAGES_TEST = 'imagesTs'
    LABELS_TRAIN = 'labelsTr'
    LABELS_TEST = 'labelsTs'
    DATASET = 'Dataset'

def create_nnUNet_folder_struct(main_dataset_folder:str):
    dataset_name = os.path.split(main_dataset_folder)[1]
    assert dataset_name.startswith(FOLDERS.DATASET)
    os.makedirs(main_dataset_folder, exist_ok=True)
    for folder in [FOLDERS.IMAGES_TRAIN, FOLDERS.IMAGES_TEST, FOLDERS.LABELS_TRAIN, FOLDERS.LABELS_TEST]:
        os.makedirs(os.path.join(main_dataset_folder, folder), exist_ok=True)

if __name__ == '__main__':
    import glob
    #main_img_bids_path = '/media/chuv2/CL-Mock-BIDS/'
    main_img_split_path = '/media/secondary/fomo-fine-tuning-datasets/fomo-task2/'
    #main_lbl_path = '/media/chuv2/MSSeg/data/split4/'
    splits_csv = '/media/secondary/fomo-fine-tuning-datasets/fomo-task2/meningioma_train_test_splits_bins-3_seed-42.csv'
    skull_stripping = False
    dataset_name = 'Dataset499_FOMO-Men_No-SK_FL-DWI'  
    # some images are not present in the split. The labels are notin the BIDS because is really horrible to follow the stadard for it
    check_in_bids_for = {'train_tp2':'merged_insider','val_tp2':'merged_insider', 'test_tp2':'merged_insider','test_out':'advanced', 'test_out_nih_3t':'nih3t', 'test_out_nih_7t':'nih7t'}
    splits_df = pd.read_csv(splits_csv)

    dataset_path = os.path.join(main_img_split_path, dataset_name)
    create_nnUNet_folder_struct(dataset_path)
    nnunet_img_train = os.path.join(dataset_path, FOLDERS.IMAGES_TRAIN)
    nnunet_img_test = os.path.join(dataset_path, FOLDERS.IMAGES_TEST)
    nnunet_lbl_train = os.path.join(dataset_path, FOLDERS.LABELS_TRAIN)
    nnunet_lbl_test = os.path.join(dataset_path, FOLDERS.LABELS_TEST)
    
    n_train = 0
    #for row in splits_df.sample(n=3).itertuples():
    for row in splits_df.itertuples(): 
        lbl_pth = row.filename
        img_dir = os.path.dirname(lbl_pth).replace('labels','preprocessed')
        img_pth = []
        for sequence in ['flair.nii.gz', 'dwi_b1000.nii.gz']:#, 't2s.nii.gz', 'swi.nii.gz']:
            img_pth.extend(glob.glob(os.path.join(img_dir, sequence)))

        if not os.path.exists(lbl_pth):
            print('Label',lbl_pth, 'not found')
         
         
        for img in img_pth:
            print('Processing',img)
            if SEQUENCES.FLAIR.lower() in img.lower():
                sequence = SEQUENCES.FLAIR
                sufix = '0000'
            elif SEQUENCES.DWIB1000.lower() in img.lower():
                sequence = SEQUENCES.DWIB1000
                sufix = '0001'
            elif SEQUENCES.T2START.lower() in img.lower():
                sequence = SEQUENCES.T2START
                sufix = '0002'
            else:
                sequence = SEQUENCES.SWI
                sufix = '0002' # use the one available t2* or swi

            nnunet_filename = f"{row.subject_id}"
            nnunet_filename_img = f"{nnunet_filename}_{sufix}.nii.gz"
            save_img_pth_base = nnunet_img_train if row.train else nnunet_img_test
            save_img_pth = os.path.join(save_img_pth_base, nnunet_filename_img)
            
            if skull_stripping: # not sure how the skull stripping was done before so for homogenization I re-do it
                synth_strip(os.path.dirname(img), os.path.basename(img),save_img_pth_base, nnunet_filename_img)
            else:
                shutil.copyfile(img, save_img_pth)
        save_lbl_pth = os.path.join(nnunet_lbl_train, f"{nnunet_filename}.nii.gz") if row.train else os.path.join(nnunet_lbl_test, f"{nnunet_filename}.nii.gz")
        shutil.copyfile(lbl_pth, save_lbl_pth)    
        n_train = n_train +1 if row.train else n_train 
        

    
    ## generate json 
     
    description = f"This dataset use Flair, DWI b1000 from FOMO challenge for segmentation task. The dataset is split into train and test sets based on the provided splits file {splits_csv}."
    generate_dataset_json(dataset_path, 
                          channel_names={0:SEQUENCES.FLAIR, 1:SEQUENCES.DWIB1000, 2:f"{SEQUENCES.T2START}/{SEQUENCES.SWI}"},  
                          labels={'background':0, "Meningioma": 1,}, 
                          num_training_cases=n_train, 
                          file_ending='.nii.gz', 
                          dataset_name=dataset_name, 
                          description=description)   
    