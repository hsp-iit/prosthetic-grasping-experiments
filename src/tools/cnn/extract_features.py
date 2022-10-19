'''
launch command example:
python3 src/tools/cnn/extract_features.py \
--batch_size 1 --source Wrist_d435 \
--input rgb --model cnn --dataset_type SingleSourceImage \
--feature_extractor mobilenet_v2 --pretrain imagenet \
--dataset_name iHannesDataset 
python3 src/tools/cnn/extract_features.py \
--batch_size 1 --source Wrist_d435 \
--input rgb --model cnn --dataset_type SingleSourceImage \
--feature_extractor mobilenet_v2 --pretrain imagenet \
--dataset_name ycb_synthetic_dataset --synthetic --train_test_split_size 1.0 \
--val_test_split_size 1.0
'''
from email.mime import base
import sys
import os
import time
import math
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

sys.path.append(os.getcwd())
from src.utils.pipeline import load_dataset, load_model, MyCompose
from src.configs.arguments import parse_args
from src.configs.conf import BASE_DIR

def main(args):
    if args.from_features:
        raise ValueError('Wrong argument: --from_features has no sense '
                         'for this pipeline')

    p = pathlib.Path(__file__)
    try:
        p.parts.index(args.model)
    except:
        raise ValueError(
            'Wrong pipeline launched: this pipeline is intended for --model=={}'
            .format(os.path.basename(__file__))
        )

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.dataset_type = 'SingleSourceVideo'
    dataloader = load_dataset(args)
    phases = list(dataloader.keys())

    # The training transform has randomizations, we do not want them since we 
    # are extracting fixed features, therefore replace it with the test transform
    base_dataset = dataloader[phases[0]].dataset
    if args.test_type in [None, 'test_same_person']:
        base_dataset = base_dataset.dataset
    if args.pretrain == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        resize_side_size = 256
        crop_size = 224
    else:
        raise ValueError('Not yet implemented for --pretrain {}'
                         .format(args.pretrain))
    transform = None
    if args.dataset_type == 'SingleSourceVideo':
        transform = MyCompose([
            transforms.Resize(resize_side_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        raise ValueError
    base_dataset._transform['train'] = transform

    num_videos = 0
    for p in phases:
        num_videos += len(dataloader[p].dataset)
    print('Extracting features for {} videos'.format(num_videos))

    model = load_model(args)
    model = model._feature_extractor

    if args.parallel:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        else:
            print('WARNING: Only one GPU found, running on single GPU')

    model = model.to(device)
    model.eval()
    with torch.set_grad_enabled(False):
        for p in phases:
            base_dataset = dataloader[p].dataset
            if args.test_type in [None, 'test_same_person']:
                base_dataset = base_dataset.dataset
            base_dataset.train(p=='train')

            for batch_idx, (frames, _, _, _, _, videos_path) in \
                    enumerate(dataloader[p]):

                batch_size, num_frames_in_video, C, H, W = frames.shape

                start = time.time()

                # fuse batch size and num_frames_in_video
                frames = frames.view(
                    -1, frames.shape[2], frames.shape[3], frames.shape[4]
                )
                frames = frames.to(device)

                features = model(frames)
                # features.shape (batch_size * num_frames_in_video, feature_vector_dim)

                features = features.view(
                    batch_size, num_frames_in_video, features.shape[1]
                )
                features = features.cpu().numpy()

                for sample_idx, v_p in enumerate(videos_path):
                    # Replace intermediate /frames/ folder 
                    # with /features/feat_extr_name/ folders
                    new_folders = os.path.join('features', args.feature_extractor)
                    old_path = pathlib.Path(v_p)
                    index_to_replace = old_path.parts.index('frames')
                    new_path = os.path.join(*old_path.parts[:index_to_replace], 
                                            new_folders, 
                                            *old_path.parts[index_to_replace+1:])
                    if not os.path.isdir(new_path):
                        os.makedirs(new_path)

                    features_path = os.path.join(new_path, 'features.npy')

                    np.save(features_path, features[sample_idx])

                    if args.verbose:
                        print('Saved features at {}'.format(features_path))

                end = time.time()
                print('Processed batch {} of {} |Elapsed time: {:.2f}'
                      .format((batch_idx//args.batch_size)+1,
                              math.ceil(num_videos/args.batch_size),
                              end-start))


if __name__ == '__main__':
    cur_base_dir = os.getcwd()
    cur_base_dir = os.path.basename(cur_base_dir)
    if cur_base_dir != BASE_DIR:
        raise Exception(
            'Wrong base dir, this file must be run from {} directory.'
            .format(BASE_DIR)
        )

    main(parse_args())