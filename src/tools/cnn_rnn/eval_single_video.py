'''
launch command example:
python3 src/tools/cnn_rnn/eval_single_video.py --epochs 10 \
--batch_size 32 --source Wrist_d435 --dataset_type SingleSourceVideo \
--split random --input RGB --output preshape --model cnn_rnn --rnn_type lstm \
--rnn_hidden_size 256 --feature_extractor mobilenet_v2 --pretrain imagenet \
--freeze_all_conv_layers --from_features --dataset_name iHannesDataset \
--video_specs 019_pitcher_base,lateral_no3,rgb_00012 --show_perframe_preds
'''
import sys
import os
import glob
import pathlib

import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2

sys.path.append(os.getcwd())
from src.utils.pipeline import check_arguments_matching, set_seed, \
    load_model, check_arguments_matching
from src.configs.arguments import parse_args
from src.configs.conf import BASE_DIR


def main(args):
    if args.video_specs is None:
        raise ValueError('When evaluating a model on a single video, you must '
                         'specify the video with --video_specs argument. '
                         'A video is uniquely identified by the instance, '
                         'preshape and sequence number. '
                         'E.g., --video_specs *instance_name*,*preshape_class*,'
                         '*rgb_xxxxx*')

    if args.checkpoint is None:
        raise ValueError('You have to specify the checkpoint file of the '
                         'model to evalute via --checkpoint argument.'
                         'The file name must be best_model.pth and it has to '
                         'be located under the tensorboard log_dir of the '
                         'corresponding training. The path is intended '
                         'relative to the base repo folder '
                         '(i.e., prosthetic-grasping-experiments).')

    if args.synthetic:
        print('Evaluating the synthetic-trained model on the real {} '
              'test set of the {} dataset'
              .formt(args.test_type, 'iHannesDataset'))
        # in order to change the dataset here in code, both 
        # args.dataset_base_folder and args.dataset_name need to be adjusted
        # (to understand why, see src/configs/arguments.py)

        # replace the dataset_name folder of the
        #  args_dataset_base_name folder path
        old_path = pathlib.Path(args.dataset_base_folder)
        idx_to_replace = old_path.parts.index(args.dataset_name)
        new_path = os.path.join(*old_path.parts[:idx_to_replace],
                                'iHannesDataset',
                                *old_path.parts[idx_to_replace+1:])
        args.dataset_base_folder = new_path
        # we need to replace also synthetic with real folder
        old_path = pathlib.Path(args.dataset_base_folder)
        idx_to_replace = old_path.parts.index('synthetic')
        new_path = os.path.join(*old_path.parts[:idx_to_replace],
                                'real',
                                *old_path.parts[idx_to_replace+1:])
        args.dataset_base_folder = new_path
        # change the dataset name
        args.dataset_name = 'iHannesDataset'

    if args.dataset_name != 'iHannesDataset':
        raise ValueError('{} is not a valid value for --dataset_name argument.'
                         ' Currently, you can evaluate your model only on '
                         'iHannesDataset test set.'.format(args.dataset_name))

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
    set_seed(args.seed)

    check_arguments_matching(os.path.dirname(args.checkpoint), sys.argv)

    model = load_model(args)

    if args.parallel:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        else:
            print('WARNING: Only one GPU found, running on single GPU')

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    softmax = nn.Softmax(dim=2)

    if not args.from_features:
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

    model.eval()
    model = model.to(device)
    with torch.set_grad_enabled(False):
        args.video_specs = args.video_specs.split(',')
        instance = args.video_specs[0]
        preshape = args.video_specs[1]
        seq_num = args.video_specs[2]

        # Load video frames/features
        video_path = glob.glob(
            os.path.join(args.dataset_base_folder,
                         '*',
                         instance,
                         preshape,
                         args.source,
                         seq_num)
        )
        if len(video_path) != 1:
            raise ValueError('Something went wrong, exaclty one video should '
                             'be found, but {} were found'
                             .format(len(video_path)))
        video_path = video_path[0]
        frames_path = glob.glob(os.path.join(video_path, '*.jpg'))
        frames_path.sort()
        if args.from_features:
            new_folders = os.path.join('features', args.feature_extractor)
            old_path = pathlib.Path(video_path)
            index_to_replace = old_path.parts.index('frames')
            video_path_features = os.path.join(*old_path.parts[:index_to_replace],
                                               new_folders,
                                               *old_path.parts[index_to_replace+1:])

            video_path_features = os.path.join(video_path_features,
                                               'features.npy')
            frames = np.load(video_path_features)
            frames = torch.tensor(frames).to(dtype=torch.float32)
        else:
            frames_list = []
            for f_p in frames_path:
                frame = Image.open(f_p)
                frame = transform(frame).to(dtype=torch.float32)
                if args.source == 'Wrist_t265':
                    raise RuntimeError('Check whether Wrist_t265 frames are all'
                                       'upside down or not')
                    # frame = ImageOps.flip(frame)
                frames_list.append(frame)
            frames = torch.stack(frames_list)
        # Add batch dimension
        frames = frames.unsqueeze(0)

        # Load label and metadata
        metadata_video_path = video_path.replace(args.source, 'metadata')\
            .replace(args.input, 'seq')
        with open(os.path.join(metadata_video_path, 'data.log')) as metadata_file:
            lines = metadata_file.readlines()
            grasp_type = lines[0].split(' ')[5]
            wrist_ps = lines[-1].split('pronation-supination')[1].strip()
        preshape_wrist_ps = preshape + '_wps' + wrist_ps
        idx_grasp_type = args.data_info['grasp_types'].index(grasp_type)
        idx_preshape = args.data_info['preshapes'].index(preshape)
        idx_instance = args.data_info['instances'].index(instance)
        idx_wrist_ps = args.data_info['wrist_pss'].index(wrist_ps)
        idx_preshape_wrist_ps = args.data_info['preshape_wrist_pss'].index(preshape_wrist_ps)
        idx_no_grasp = 0
        num_frames = len(frames[0])
        perframe_grasp_type = torch.tensor(idx_grasp_type).repeat(num_frames)
        perframe_preshape = torch.tensor(idx_preshape).repeat(num_frames)
        perframe_instance = torch.tensor(idx_instance).repeat(num_frames)
        perframe_wrist_ps = torch.tensor(idx_wrist_ps).repeat(num_frames)
        perframe_preshape_wrist_ps = torch.tensor(idx_preshape_wrist_ps).repeat(num_frames)
        if num_frames != 90:
            raise RuntimeError('The labeling scheme below works only with '
                               'videos 90 frames long')
        perframe_grasp_type[75:] = idx_no_grasp
        perframe_preshape[75:] = idx_no_grasp
        perframe_instance[75:] = idx_no_grasp
        perframe_wrist_ps[75:] = idx_no_grasp
        perframe_preshape_wrist_ps[75:] = idx_no_grasp
        # Add batch dimension
        perframe_grasp_type = perframe_grasp_type.unsqueeze(0)
        perframe_preshape = perframe_preshape.unsqueeze(0)
        perframe_instance = perframe_instance.unsqueeze(0)
        perframe_wrist_ps = perframe_wrist_ps.unsqueeze(0)
        perframe_preshape_wrist_ps = perframe_preshape_wrist_ps.unsqueeze(0)
        if args.output == 'grasp_type':
            perframe_target = perframe_grasp_type
        elif args.output == 'preshape':
            perframe_target = perframe_preshape
        elif args.output == 'instance':
            perframe_target = perframe_instance
        elif args.output == 'wrist_ps':
            perframe_target = perframe_wrist_ps
        elif args.output == 'preshape_wrist_ps':
            perframe_target = perframe_preshape_wrist_ps
        else:
            raise ValueError(
                'Not yet implemented for --output {}'
                .format(args.output)
            )

        # frames.shape (1 ,num_frames_in_video==90, 3, 224, 224)
        # perframe_target.shape (1, num_frames_in_video==90)

        frames = frames.to(device)

        perframe_scores = model(frames)
        # scores.shape (1, num_frames_in_video==90, num_classes)
        perframe_scores = softmax(perframe_scores).cpu()
        perframe_pred = perframe_scores.argmax(dim=2)

        # The code below is to evaluate also at video granularity,
        # not only frame granularity

        if args.output == 'grasp_type':
            video_target = grasp_type
        elif args.output == 'preshape':
            video_target = preshape
        elif args.output == 'instance':
            video_target = instance
        elif args.output == 'wrist_ps':
            video_target = wrist_ps
        elif args.output == 'preshape_wrist_ps':
            video_target = preshape_wrist_ps
        else:
            raise ValueError(
                'Not yet implemented for --output {}'
                .format(args.output)
            )

        # Get the video prediction: per-frame majority voting without
        # considering the background class predictions
        perframe_pred = perframe_pred.squeeze(0)
        perframe_pred_wo_backgr = perframe_pred[perframe_pred != idx_no_grasp]
        if len(perframe_pred_wo_backgr) == 0:
            # i.e. the model predicted always background
            # along the whole video
            video_pred_wo_backgr = torch.tensor(0)
        else:
            # Video prediction performed via majority voting
            # over all the non-background-predicted frames
            # of the video
            video_pred_wo_backgr, _ = torch.mode(perframe_pred_wo_backgr)
        video_pred_wo_backgr = video_pred_wo_backgr.item()

        video_target_idx = args.data_info[args.output + 's'].index(video_target)
        if video_target_idx == video_pred_wo_backgr:
            print('Video CORRECTLY predicted')
        else:
            print('Video WRONGLY predicted')

        perframe_target = perframe_target.squeeze(0)
        print('{} out of {} frames are correctly predicted'
              .format((perframe_pred == perframe_target).sum().item(),
                      num_frames))

        if args.show_perframe_preds:
            print('Visualizing frames and predictions.\nPress a key to move '
                  'to the next frame')

            first_pred_cls_list = []
            perframe_scores = perframe_scores.squeeze(0)
            for idx, f_p in enumerate(frames_path):
                frame = cv2.imread(f_p)
                H, W, _ = frame.shape

                idxs_pred = perframe_scores[idx].sort(descending=True)[1].tolist()
                first_pred_cls = args.data_info[args.output+'s'][idxs_pred[0]]
                cv2.putText(
                    frame,
                    '{:<12s}: {:<.2f}'.format(first_pred_cls, perframe_scores[idx, idxs_pred[0]]),
                    (0, int(H * 0.15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if idxs_pred[0] == perframe_target[idx] else (0, 0, 255),
                    2
                )
                second_pred_cls = args.data_info[args.output+'s'][idxs_pred[1]]
                cv2.putText(
                    frame,
                    '{:<12s}: {:<.2f}'.format(second_pred_cls, perframe_scores[idx, idxs_pred[1]]),
                    (0, int(H * 0.30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2
                )
                third_pred_cls = args.data_info[args.output+'s'][idxs_pred[2]]
                cv2.putText(
                    frame,
                    '{:<12s}: {:<.2f}'.format(third_pred_cls, perframe_scores[idx, idxs_pred[2]]),
                    (0, int(H * 0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2
                )
                # Ground-truth class
                cv2.putText(
                    frame,
                    '{:<15s}'.format(args.data_info[args.output+'s'][perframe_target[idx]]),
                    (int(W - W * 0.30), int(H * 0.15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )
                # Majority voting (w/o considering background class) until the 
                # current frame
                first_pred_cls_list.append(idxs_pred[0])
                pred_wo_backgr = np.array(first_pred_cls_list) 
                pred_wo_backgr = pred_wo_backgr[pred_wo_backgr != idx_no_grasp]
                if len(pred_wo_backgr) == 0:
                    out_cls = 'no prediction'
                else:
                    pred_wo_backgr = torch.mode(torch.tensor(pred_wo_backgr))[0].item()
                    out_cls = args.data_info[args.output+'s'][pred_wo_backgr]
                    count_cls = np.array(first_pred_cls_list)
                    count_cls = count_cls[count_cls == pred_wo_backgr]
                    count_cls = len(count_cls)
                cv2.putText(
                    frame,
                    '{:<16s} {:2}/{:2}'.format(out_cls, count_cls, len(first_pred_cls_list)),
                    (0, int(H * 0.75)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if out_cls == video_target else (0, 0, 255),
                    2
                )

                cv2.imshow('frame', frame)
                cv2.waitKey(0)

            cv2.destroyAllWindows()
                

if __name__ == '__main__':
    cur_base_dir = os.getcwd()
    cur_base_dir = os.path.basename(cur_base_dir)
    if cur_base_dir != BASE_DIR:
        raise Exception(
            'Wrong base dir, this file must be run from {} directory.'
            .format(BASE_DIR)
        )

    main(parse_args())
