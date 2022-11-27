import argparse
import zipfile
import os
import glob
import shutil

import wget


URL_BASE_STR = 'https://zenodo.org/record/7327516/files/{}?download=1'
DWN_BASE_STR = '[INFO] downloading {}'
UNZIP_BASE_STR = '[INFO] unzipping {}'
ARNG_BASE_STR = '[INFO] arranging {}'


def download(names, out_folder):
    for n in names:
        print(DWN_BASE_STR.format(n))
        wget.download(URL_BASE_STR.format(n), out=out_folder)


def unzip(in_folder, names, out_folder, remove_zip):
    for n in names:
        path = os.path.join(in_folder, n)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(out_folder)
        if remove_zip:
            os.remove(path)


def main(out_dataset_folder, remove_zips):
    if os.path.isdir(out_dataset_folder):
        raise Exception('The directory {} already exists'
                        .format(out_dataset_folder))
    print('[INFO] creating directory {}'.format(out_dataset_folder))
    os.mkdir(out_dataset_folder)

    ## REAL FRAMES
    r_fr_dirs = os.path.join(
        out_dataset_folder, 'real', 'frames', 'iHannesDataset'
    )
    print('[INFO] creating directories {}'.format(r_fr_dirs))
    os.makedirs(r_fr_dirs)

    REAL_FRAMES_NAMES = [
        'REAL_FRAMES_big_ball.zip',
        'REAL_FRAMES_big_tool.zip', 'REAL_FRAMES_box.zip' 
        'REAL_FRAMES_brush.zip', 'REAL_FRAMES_can.zip',
        'REAL_FRAMES_dispenser.zip', 'REAL_FRAMES_glass.zip', 
        'REAL_FRAMES_long_fruit.zip', 'REAL_FRAMES_mug.zip', 
        'REAL_FRAMES_plate.zip', 'REAL_FRAMES_ringbinder.zip',
        'REAL_FRAMES_small_ball.zip', 'REAL_FRAMES_small_cube.zip',
        'REAL_FRAMES_small_tool.zip', 'REAL_FRAMES_sponge.zip', 
        'REAL_FRAMES_tube.zip', 'REAL_FRAMES_wallet.zip',
    ]
    download(REAL_FRAMES_NAMES, r_fr_dirs)
    unzip(r_fr_dirs, REAL_FRAMES_NAMES, r_fr_dirs, remove_zips)

    ## REAL FEATURES
    r_ft_dirs = os.path.join(
        out_dataset_folder, 'real', 'features', 'mobilenet_v2', 'iHannesDataset'
    )
    print('[INFO] creating directories {}'.format(r_ft_dirs))
    os.makedirs(r_ft_dirs)

    REAL_FEATURES_NAMES = ['REAL_FEATURES_ALL.zip']
    download(REAL_FEATURES_NAMES, r_ft_dirs)
    unzip(r_ft_dirs, REAL_FEATURES_NAMES, r_ft_dirs, remove_zips)
    wrong_folder = os.path.join(
        r_ft_dirs, 'features', 'mobilenet_v2', 'iHannesDataset'
    ) 
    obj_folders = glob.glob(os.path.join(wrong_folder, '*'))
    for o_f in obj_folders:
        shutil.move(o_f, r_ft_dirs)
    os.rmdir(wrong_folder)


    ## SYNTHETIC FRAMES
    s_fr_dirs = os.path.join(
        out_dataset_folder, 'synthetic', 'frames', 'ycb_synthetic_dataset'
    )
    print('[INFO] creating directories {}'.format(s_fr_dirs))
    os.makedirs(s_fr_dirs)
    
    SYNTHETIC_FRAMES_NAMES = [
        'SYNTHETIC_FRAMES_big_ball.zip',
        'SYNTHETIC_FRAMES_big_tool.zip', 'SYNTHETIC_FRAMES_can.zip',
        'SYNTHETIC_FRAMES_dispenser.zip', 'SYNTHETIC_FRAMES_long_fruit.zip',
        'SYNTHETIC_FRAMES_mug.zip', 'SYNTHETIC_FRAMES_plate.zip',
        'SYNTHETIC_FRAMES_small_ball.zip', 'SYNTHETIC_FRAMES_small_cube.zip',
        'SYNTHETIC_FRAMES_small_tool.zip', 'SYNTHETIC_FRAMES_tube.zip',
    ]
    download(SYNTHETIC_FRAMES_NAMES, s_fr_dirs)
    unzip(s_fr_dirs, SYNTHETIC_FRAMES_NAMES, s_fr_dirs, remove_zips)

    ## SYNTHETIC FEATURES
    s_ft_dirs = os.path.join(
        out_dataset_folder, 'synthetic', 'features', 'mobilenet_v2', 'ycb_synthetic_dataset'
    )
    print('[INFO] creating directories {}'.format(s_ft_dirs))
    os.makedirs(s_ft_dirs)

    SYNTHETIC_FEATURES_NAMES = ['SYNTHETIC_FEATURES_ALL.zip']
    download(SYNTHETIC_FEATURES_NAMES, s_ft_dirs)
    unzip(s_ft_dirs, SYNTHETIC_FEATURES_NAMES, s_ft_dirs, remove_zips)
    wrong_folder = os.path.join(
        s_ft_dirs, 'features', 'mobilenet_v2', 'ycb_synthetic_dataset'
    ) 
    obj_folders = glob.glob(os.path.join(wrong_folder, '*'))
    for o_f in obj_folders:
        shutil.move(o_f, s_ft_dirs)
    os.rmdir(wrong_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dataset_folder', type='str', required=True)
    parser.add_argument('--remove_zips', action='store_true')
    args = parser.parse_args()

    main(args.out_dataset_folder, args.remove_zips)
