'''
launch command example:
python3 src/tools/cnn/extract_features.py \
--batch_size 32 --source Wrist_d435 --dataset_type SingleSourceImage \
--input rgb --model cnn \
--feature_extractor mobilenet_v2 --pretrain imagenet \
--dataset_name iHannesDataset
'''
import sys
import os
import time
import glob
import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

sys.path.append(os.getcwd())
from src.utils.pipeline import load_dataset, load_model
from src.configs.arguments import parse_args


def main(args):
    if args.from_features:
        raise ValueError('Wrong argument: --from_features has no sense '
                         'for this pipeline')

    modelname_dir = __file__.split('/')[-2]
    if args.model != modelname_dir:
        raise ValueError(
            'Wrong pipeline launched: this pipeline is intended for --model=={}'
            .format(modelname_dir)
        )

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.dataset_type = 'SingleSourceVideo'
    dataloader = load_dataset(args)
    # if args.test_type is None:
    #     phases = ['train', 'val']
    # else:
    #     phases = ['test']
    phases = list(dataloader.keys())

    num_videos = 0
    for p in phases:
        num_videos += len(dataloader[p].dataset)
    print('Extracting features for {} videos'.format(num_videos))

    if args.pretrain == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        resize_side_size = 256
        crop_size = 224
        transform = transforms.Compose([
            transforms.Resize(resize_side_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        raise NotImplementedError('Not yet implemented for --pretrain {}'
                                  .format(args.pretrain))

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
                    new_video_dir = v_p.replace(
                        '/frames/', '/features/{}/'.format(args.feature_extractor)
                    )
                    features_path = os.path.join(new_video_dir, 'features.npy')

                    np.save(features[sample_idx], features_path)

                    if args.verbose:
                        print('Saved features at {}'.format(features_path))

                end = time.time()
                print('Processed batch {} of {} |Elapsed time: {:.2f}'
                      .format((batch_idx//args.batch_size)+1,
                              math.ceil(num_videos/args.batch_size),
                              end-start))


if __name__ == '__main__':
    BASE_DIR = 'iHannes_experiments'

    cur_base_dir = os.getcwd()
    cur_base_dir = cur_base_dir.split('/')[-1]
    if cur_base_dir != BASE_DIR:
        raise Exception(
            'Wrong base dir, this file must be run from {} directory.'
            .format(BASE_DIR)
        )

    main(parse_args())
