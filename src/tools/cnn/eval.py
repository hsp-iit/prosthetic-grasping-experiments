'''
launch command example:
python3 src/tools/cnn/eval.py --epochs 10 \
--batch_size 32 --source Wrist_d435 --dataset_type SingleSourceImage \
--split random --input rgb --output preshape --model cnn \
--feature_extractor mobilenet_v2 --pretrain imagenet --freeze_all_conv_layers \
--from_features --dataset_name iHannesDataset \
--checkpoint runs/logs_folder_name/best_model.pth \
--test_type test_different_velocity
python3 src/tools/cnn/eval.py --epochs 5 \
--batch_size 256 --source Wrist_d435 --dataset_type SingleSourceImage \
--split random --input rgb --output preshape --model cnn \
--feature_extractor mobilenet_v2 --pretrain imagenet --freeze_all_conv_layers \
--from_features --dataset_name ycb_50samples --synthetic \
--checkpoint runs/logs_folder_name/best_model.pth \
--test_type test_different_velocity
'''
import shutil
import sys
import os
import time

import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

sys.path.append(os.getcwd())
from src.utils.pipeline import set_seed, load_dataset, load_model, \
    check_arguments_matching
from src.utils.metrics import perframe_mAP, per_class_accuracy, \
    per_instance_preshape_accuracy, per_instance_grasp_type_accuracy, \
    plot_confusion_matrix
from src.configs.arguments import parse_args
from src.configs import conf


def main(args):
    if args.test_type is None:
        raise ValueError('You must specify the test set on which to evaluate '
                         'the model, it must be one among {}'
                         .format(conf.TEST_TYPE))

    if args.checkpoint is None:
        raise ValueError('You have to specify the checkpoint file of the '
                         'model to evalute via --checkpoint argument.'
                         'The file name must be best_model.pth and it has to '
                         'be located under the tensorboard log_dir of the '
                         'corresponding training. The path is intended '
                         'relative to the base repo folder '
                         '(i.e., iHannes_experiments)')

    if args.dataset_name != 'iHannesDataset':
        raise ValueError('{} is not a valid value for --dataset_name argument.'
                         'Currently, you can evaluate your model only on '
                         'iHannesDataset test set.'.format(args.dataset_name))
    # REMEMBER that if you change args.dataset_name, you have to change
    # also args.dataset_base_folder accordingly, see src/configs/arguments.py

    modelname_dir = __file__.split('/')[-2]
    if args.model != modelname_dir:
        raise ValueError(
            'Wrong pipeline launched: this pipeline is intended for --model=={}'
            .format(modelname_dir)
        )

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    writer_log_dir = os.path.dirname(args.checkpoint)

    check_arguments_matching(writer_log_dir, sys.argv)

    eval_dir = os.path.join(writer_log_dir, 'eval_' + args.test_type)
    # Create eval folder (delete then create if already exists)
    if os.path.isdir(eval_dir):
        shutil.rmtree(eval_dir)
    writer = SummaryWriter(log_dir=eval_dir)

    # Since we want to do the evaluation at video granularity (instead of
    # frame granularity only), we have to use the SingleSourceVideo dataset.
    # Moreover, for sake of convenience of all the code below, we set the
    # batch size to 1. PAY ATTENTION that if batch size is not 1, all the
    # code below requires small changes.
    args.dataset_type = 'SingleSourceVideo'
    args.batch_size = 1

    dataloader = load_dataset(args)

    with open(writer.log_dir + '/log.txt', 'w') as f:
        f.write('--seed: {}\n'.format(args.seed))
        f.write('--train_test_split_seed: {}\n'.format(args.train_test_split_seed))
        f.write('--val_test_split_seed: {}\n'.format(args.val_test_split_seed))
        f.write('--train_test_split_size: {}\n'.format(args.train_test_split_size))
        f.write('--val_test_split_size: {}\n\n'.format(args.val_test_split_size))

        log = '=== Evaluation on test set. (# of videos : {:<3}) ===' \
            .format(len(dataloader['test'].dataset))
        print(log)
        f.write(log + '\n\n')

        command = 'python3 ' + ' '.join(sys.argv)
        f.write(command + '\n')

    model = load_model(args)

    if args.parallel:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        else:
            print('WARNING: Only one GPU found, running on single GPU')

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    softmax = nn.Softmax(dim=1)

    model.eval()
    model = model.to(device)
    with torch.set_grad_enabled(False):
        accuracy = {granularity: 0 for granularity in ['perframe', 'video']}
        accuracy['video'] = {type: 0 for type in ['w_backgr', 'w/o_backgr']}
        scores_all = {'perframe': []}
        targets_all = {granularity: [] for granularity in ['perframe', 'video']}
        preds_all = {granularity: [] for granularity in ['perframe', 'video']}
        instances_all = {granularity: [] for granularity in ['perframe', 'video']}

        log_wrong_video_predictions = '=== VIDEO WRONG PREDICTIONS ===\n'
        total_start = time.time()

        for batch_idx, (frames, perframe_grasp_type, perframe_preshape, perframe_instance, _, video_path) \
                in enumerate(dataloader['test'], start=1):
            batch_start = time.time()

            if frames.shape[0] != 1:
                raise RuntimeError('Wrong batch size: in the test '
                                   'phase, you need to use '
                                   'SingleSourceVideo dataset with '
                                   'batch_size 1 to make the '
                                   'code below working')
            # frames.shape (batch_size==1, num_frames_in_video==90, 3, 224, 224)
            # grasp_types (batch_size==1, num_frames_in_video==90)

            frames = frames.squeeze(0)
            perframe_grasp_type = perframe_grasp_type.squeeze(0)
            perframe_preshape = perframe_preshape.squeeze(0)
            perframe_instance = perframe_instance.squeeze(0)

            if args.output == 'grasp_type':
                perframe_target = perframe_grasp_type
            elif args.output == 'preshape':
                perframe_target = perframe_preshape
            elif args.output == 'instance':
                perframe_target = perframe_instance
            else:
                raise NotImplementedError(
                    'Not yet implemented for --output {}'
                    .format(args.output)
                )

            frames = frames.to(device)

            perframe_scores = model(frames)
            # scores.shape (num_frames_in_video==90, num_classes)
            perframe_scores = softmax(perframe_scores).cpu()

            scores_all['perframe'].append(perframe_scores)
            targets_all['perframe'].append(perframe_target)

            perframe_pred = perframe_scores.argmax(dim=1)
            preds_all['perframe'].append(perframe_pred)

            instances_all['perframe'].append(perframe_instance)

            accuracy['perframe'] += (perframe_pred == perframe_target).sum().item()

            # The code below is needed since we want to evaluate also
            # at video granularity, not only frame granularity.

            idx_no_grasp = 0
            # Convert instance from per frame labels to single video label
            appo = torch.unique(perframe_instance)
            video_instance = appo[appo != idx_no_grasp]
            if video_instance.shape != (1,):
                raise RuntimeError('Something went wrong with the '
                                   'labels vector received: it '
                                   'should contain only two '
                                   'indexes, the background and '
                                   'class indexes.')
            video_instance = video_instance.item()
            instances_all['video'].append(video_instance)
            # Convert target from per frame labels to single video label
            appo = torch.unique(perframe_target)
            video_target = appo[appo != idx_no_grasp]
            if video_target.shape != (1,):
                raise RuntimeError('Something went wrong with the '
                                   'labels vector received: it '
                                   'should contain only two '
                                   'indexes, the background and '
                                   'class indexes.')
            video_target = video_target.item()
            targets_all['video'].append(video_target)

            # Compute video accuracy, w/ and w/o background

            # Video accuracy w/ background
            video_pred, _ = torch.mode(perframe_pred)
            video_pred = video_pred.item()
            accuracy['video']['w_backgr'] += video_pred == video_target

            # Video accuracy w/o background
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
            accuracy['video']['w/o_backgr'] += video_pred_wo_backgr == video_target

            preds_all['video'].append(video_pred_wo_backgr)

            batch_end = time.time()
            if args.verbose:
                if args.verbose:
                    print('[test] Video: {:>3}/{:<3}  |Running time: {:.1f}s'
                          .format(batch_idx, len(dataloader['test'].dataset),
                                  batch_end - batch_start))

            # Log the video wrongly predicted.
            # The prediction considered for the video is the majority
            # voting without background class frames prediction
            if video_pred_wo_backgr != video_target:
                idx_no_grasp = 0
                video_grasp_type = torch.unique(perframe_grasp_type)
                video_grasp_type = video_grasp_type[video_grasp_type != idx_no_grasp]
                video_grasp_type = video_grasp_type.item()
                video_preshape = torch.unique(perframe_preshape)
                video_preshape = video_preshape[video_preshape != idx_no_grasp]
                video_preshape = video_preshape.item()
                video_instance = torch.unique(perframe_instance)
                video_instance = video_instance[video_instance != idx_no_grasp]
                video_instance = video_instance.item()

                log = 'Video with GRASP_TYPE:{:<20} PRESHAPE:{:<20} ' \
                      'INSTANCE:{:<20}   predicted as   {}:{:<20}\n' \
                    .format(video_grasp_type, video_preshape, video_instance,
                            args.output.upper(),
                            args.data_info[args.output+'s'][video_pred_wo_backgr])

                log_wrong_video_predictions += log
                log_wrong_video_predictions += \
                    '(the video path is {})\n-\n'.format(video_path[0])

        total_end = time.time()
        log = 'Running time: {:3.1f}s'.format(total_end - total_start)
        with open(writer.log_dir + '/log.txt', 'a+') as f:
            print(log)
            f.write(log + '\n\n')

        for granularity in ['perframe', 'video']:
            log = '=== {} METRICS ===\n\n'.format(granularity.upper())

            if granularity == 'perframe':
                targets_all['perframe'] = torch.cat(targets_all['perframe']).numpy()
                preds_all['perframe'] = torch.cat(preds_all['perframe']).numpy()
                instances_all['perframe'] = torch.cat(instances_all['perframe']).numpy()
            elif granularity == 'video':
                targets_all['video'] = np.array(targets_all['video'])
                preds_all['video'] = np.array(preds_all['video'])
                instances_all['video'] = np.array(instances_all['video'])
            else:
                raise NotImplementedError

            classes = args.data_info[args.output+'s']

            if granularity == 'perframe':
                dataset_len = len(dataloader['test'].dataset) * \
                    dataloader['test'].dataset._NUM_FRAMES_IN_VIDEO
            elif granularity == 'video':
                dataset_len = len(dataloader['test'].dataset)
            else:
                raise NotImplementedError

            if granularity == 'perframe':
                log += 'ACCURACY: {:.1f}%\n\n'.format(
                    accuracy['perframe'] / dataset_len
                )
            elif granularity == 'video':
                log += 'ACCURACY: {:.1f}%\n\n'.format(
                    accuracy['video']['w/o_backgr'] / dataset_len
                )
            else:
                raise NotImplementedError

            log += 'PER-CLASS ACCURACY:\n'
            per_class_accuracies = per_class_accuracy(
                targets_all[granularity], preds_all[granularity], np.arange(len(classes))
            )
            for idx, c in enumerate(classes):
                log += '{:<12}: {:<.1f}%\n'.format(c, per_class_accuracies[idx] * 100)
            log += '\n\n'

            log += 'CLASSIFICATION REPORT:\n'
            log += classification_report(targets_all[granularity],
                                         preds_all[granularity],
                                         np.arange(len(classes)).tolist(),
                                         classes)
            log += '\n\n'

            if args.output == 'preshape':
                title = 'PER-INSTANCE-PRESHAPE {}-ACCURACY'\
                        .format(granularity.upper())
                per_instance_preshape_accuracy(
                    targets_all[granularity], preds_all[granularity],
                    instances_all[granularity], args.data_info, writer=writer,
                    title=title
                )
            elif args.output == 'grasp_type':
                title = 'PER-INSTANCE-GRASP_TYPE {}-ACCURACY'\
                        .format(granularity.upper())
                per_instance_grasp_type_accuracy(
                    targets_all[granularity], preds_all[granularity],
                    instances_all[granularity], args.data_info, writer=writer,
                    title=title
                )
            elif args.output == 'instance':
                raise NotImplementedError
            else:
                raise NotImplementedError

            figsize = (10, 10)
            plot_confusion_matrix(targets_all[granularity],
                                  preds_all[granularity],
                                  classes, normalize=True, writer=writer,
                                  figsize=figsize)
            plot_confusion_matrix(targets_all[granularity],
                                  preds_all[granularity],
                                  classes, normalize=False, writer=writer,
                                  figsize=figsize)

            if granularity == 'perframe':
                scores_all['perframe'] = torch.cat(scores_all['perframe']).numpy()

                mAP_results = perframe_mAP(
                    scores_all['perframe'], targets_all['perframe'], classes
                )
                log += 'PER-FRAME MEAN AVERAGE PRECISION\n'
                log += 'mAP along all classes: {:<3.4f}%\n'.format(mAP_results['mAP_all_cls'] * 100)
                log += 'mAP along valid classes: {:<3.4f}%\n'.format(mAP_results['mAP_valid_cls'] * 100)
                for cls in mAP_results['AP'].keys():
                    log += '{} AP: {:<3.4f}%   |'.format(cls, mAP_results['AP'][cls] * 100)
                log += '\n\n'
            else:
                log += log_wrong_video_predictions

            log += '\n\n\n\n'


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
