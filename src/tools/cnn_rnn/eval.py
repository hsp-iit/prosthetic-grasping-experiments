'''
launch command example:
python3 src/tools/cnn_rnn/eval.py --epochs 10 \
--batch_size 32 --source Wrist_d435 --dataset_type SingleSourceVideo \
--split random --input rgb --output preshape --model cnn_rnn --rnn_type lstm \
--rnn_hidden_size 256 --feature_extractor mobilenet_v2 --pretrain imagenet \
--freeze_all_conv_layers --from_features --dataset_name iHannesDataset \
--checkpoint runs/logs_folder_name/best_model.pth \
--test_type test_different_velocity
'''
import shutil
import sys
import os
import time
import pathlib

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
                         '(i.e., prosthetic-grasping-experiments).')

    if args.synthetic:
        print('Evaluating the synthetic-trained model on the real {} '
              'test set of the {} dataset'
              .format(args.test_type, 'iHannesDataset'))
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

    writer_log_dir = os.path.dirname(args.checkpoint)

    check_arguments_matching(writer_log_dir, sys.argv)

    eval_dir = os.path.join(writer_log_dir, 'eval_' + args.test_type)
    # Create eval folder (delete then create if already exists)
    if os.path.isdir(eval_dir):
        shutil.rmtree(eval_dir)
    writer = SummaryWriter(log_dir=eval_dir)

    dataloader = load_dataset(args)

    with open(os.path.join(writer.log_dir, 'log.txt'), 'w') as f:
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

    softmax = nn.Softmax(dim=2)

    base_dataset = dataloader['test'].dataset
    if args.test_type in [None, 'test_same_person']:
        base_dataset = base_dataset.dataset
    base_dataset.eval()
    model.eval()
    model = model.to(device)
    with torch.set_grad_enabled(False):
        accuracy = {granularity: 0 for granularity in ['perframe', 'video']}
        accuracy['video'] = {type: 0 for type in ['w_backgr', 'w/o_backgr']}
        scores_all = {'perframe': []}
        targets_all = {granularity: [] for granularity in ['perframe', 'video']}
        preds_all = {granularity: [] for granularity in ['perframe', 'video']}
        instances_all = {granularity: [] for granularity in ['perframe', 'video']}

        if args.output in ['wrist_ps', 'preshape_wrist_ps']:
            FRAMES_THRESHOLD = 30
            print('=== Evaluating on the first ' + str(FRAMES_THRESHOLD) + 
                  ' frames of each video. === \n\n')

        log_wrong_video_predictions = '=== VIDEO WRONG PREDICTIONS ===\n'
        total_start = time.time()

        for batch_idx, (frames, perframe_grasp_type, perframe_preshape, 
                        perframe_instance, _, video_path, perframe_wrist_ps,
                        perframe_preshape_wrist_ps) \
                in enumerate(dataloader['test'], start=1):
            batch_start = time.time()

            if args.output in ['wrist_ps', 'preshape_wrist_ps']:
                frames = frames[:FRAMES_THRESHOLD]
                preframe_grasp_type = perframe_grasp_type[:FRAMES_THRESHOLD]
                perframe_preshape = perframe_preshape_wrist_ps[:FRAMES_THRESHOLD]
                perframe_instance = perframe_instance[:FRAMES_THRESHOLD]
                perframe_wrist_ps = perframe_wrist_ps[:FRAMES_THRESHOLD]
                perframe_preshape_wrist_ps = perframe_preshape_wrist_ps[:FRAMES_THRESHOLD]

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
                raise NotImplementedError(
                    'Not yet implemented for --output {}'
                    .format(args.output)
                )

            frames = frames.to(device)

            perframe_scores = model(frames)
            # scores.shape (batch_size, num_frames_in_video, num_classes)
            perframe_scores = softmax(perframe_scores).cpu()

            scores_all['perframe'].append(
                perframe_scores.view(-1, perframe_scores.shape[2])
            )
            targets_all['perframe'].append(perframe_target.view(-1))

            perframe_pred = perframe_scores.argmax(dim=2)
            preds_all['perframe'].append(perframe_pred.view(-1))

            instances_all['perframe'].append(perframe_instance.view(-1))

            accuracy['perframe'] += \
                (perframe_pred.view(-1) == perframe_target.view(-1)).sum().item()

            # Evaluate also at video granularity level

            batch_size = frames.shape[0]
            idx_no_grasp = 0
            for b in range(batch_size):
                # Convert instance from per frame labels to single video label
                appo = torch.unique(perframe_instance[b])
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
                appo = torch.unique(perframe_target[b])
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
                video_pred, _ = torch.mode(perframe_pred[b])
                video_pred = video_pred.item()
                accuracy['video']['w_backgr'] += video_pred == video_target

                # Video accuracy w/o background
                perframe_pred_b = perframe_pred[b]
                perframe_pred_wo_backgr = perframe_pred_b[perframe_pred_b != idx_no_grasp]
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

                # Log the video wrongly predicted.
                # The prediction considered for the video is the majority
                # voting without background class frames prediction
                if video_pred_wo_backgr != video_target:
                    video_grasp_type = torch.unique(perframe_grasp_type[b])
                    video_grasp_type = video_grasp_type[video_grasp_type != idx_no_grasp]
                    video_grasp_type = video_grasp_type.item()
                    video_grasp_type = args.data_info['grasp_types'][video_grasp_type]
                    video_preshape = torch.unique(perframe_preshape[b])
                    video_preshape = video_preshape[video_preshape != idx_no_grasp]
                    video_preshape = video_preshape.item()
                    video_preshape = args.data_info['preshapes'][video_preshape]
                    video_instance = torch.unique(perframe_instance[b])
                    video_instance = video_instance[video_instance != idx_no_grasp]
                    video_instance = video_instance.item()
                    video_instance = args.data_info['instances'][video_instance]

                    log = 'Video with GRASP_TYPE:{:<20} PRESHAPE:{:<20} ' \
                          'INSTANCE:{:<20}   predicted as   {}:{:<20}\n' \
                          .format(video_grasp_type, video_preshape,
                                  video_instance,
                                  args.output.upper(),
                                  args.data_info[args.output+'s'][video_pred_wo_backgr])

                    log_wrong_video_predictions += log
                    log_wrong_video_predictions += \
                        '(the video path is {})\n-\n'.format(video_path[b])

            batch_end = time.time()
            if args.verbose:
                if args.verbose:
                    print('[test] Video: {:>3}/{:<3}  |Running time: {:.1f}s'
                          .format(batch_idx, len(dataloader['test'].dataset),
                                  batch_end - batch_start))

        total_end = time.time()
        log = 'Running time: {:3.1f}s'.format(total_end - total_start)
        with open(os.path.join(writer.log_dir, 'log.txt'), 'a+') as f:
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
                # dataset_len = len(dataloader['test'].dataset) * \
                #     base_dataset._NUM_FRAMES_IN_VIDEO
                dataset_len = len(dataloader['test'].dataset) * FRAMES_THRESHOLD
            elif granularity == 'video':
                dataset_len = len(dataloader['test'].dataset)
            else:
                raise NotImplementedError

            if granularity == 'perframe':
                log += 'ACCURACY: {:.2f}%\n\n'.format(
                    accuracy['perframe'] / dataset_len * 100
                )
            elif granularity == 'video':
                log += 'ACCURACY W/  BACKGR: {:.2f}%\n\n'.format(
                    accuracy['video']['w_backgr'] / dataset_len * 100
                )
                log += 'ACCURACY W/O BACKGR: {:.2f}%\n\n'.format(
                    accuracy['video']['w/o_backgr'] / dataset_len * 100
                )
            else:
                raise NotImplementedError

            log += 'PER-CLASS ACCURACY:\n'
            per_class_accuracies = per_class_accuracy(
                targets_all[granularity], preds_all[granularity], classes
            )
            for idx, c in enumerate(classes):
                log += '{:<12}: {:<.1f}%\n'.format(c, per_class_accuracies[idx] * 100)
            log += '\n\n'

            log += 'CLASSIFICATION REPORT:\n'
            log += classification_report(targets_all[granularity],
                                         preds_all[granularity],
                                         labels=np.arange(len(classes)).tolist(),
                                         target_names=classes)
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
                print('PLOT NOT IMPLEMENTED')
                # raise NotImplementedError
            elif args.output == 'wrist_ps':
                print('PLOT NOT IMPLEMENTED')
                # raise NotImplementedError()
            elif args.output == 'preshape_wrist_ps':
                print('PLOT NOT IMPLEMENTED')
                # raise NotImplementedError()
            else:
                raise NotImplementedError

            figsize = (10, 10)
            plot_confusion_matrix(targets_all[granularity],
                                  preds_all[granularity],
                                  classes, normalize=True, writer=writer,
                                  figsize=figsize,
                                  title=granularity + ' confusion matrix')
            plot_confusion_matrix(targets_all[granularity],
                                  preds_all[granularity],
                                  classes, normalize=False, writer=writer,
                                  figsize=figsize,
                                  title=granularity + ' confusion matrix')

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

            with open(os.path.join(writer.log_dir, 'log.txt'), 'a+') as f:
                print(log)
                f.write(log + '\n\n')


if __name__ == '__main__':
    cur_base_dir = os.getcwd()
    cur_base_dir = os.path.basename(cur_base_dir)
    if cur_base_dir != conf.BASE_DIR:
        raise Exception(
            'Wrong base dir, this file must be run from {} directory.'
            .format(conf.BASE_DIR)
        )

    main(parse_args())
