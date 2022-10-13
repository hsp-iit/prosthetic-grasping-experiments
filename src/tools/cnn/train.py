'''
launch command example:
python3 src/tools/cnn/train.py --epochs 10 \
--batch_size 32 --source Wrist_d435 --dataset_type SingleSourceImage \
--split random --input rgb --output preshape --model cnn \
--feature_extractor mobilenet_v2 --pretrain imagenet --freeze_all_conv_layers \
--from_features --dataset_name iHannesDataset
python3 src/tools/cnn/train.py --epochs 5 \
--batch_size 256 --source Wrist_d435 --dataset_type SingleSourceImage \
--split random --input rgb --output preshape --model cnn \
--feature_extractor mobilenet_v2 --pretrain imagenet --freeze_all_conv_layers \
--from_features --dataset_name ycb_50samples --synthetic
'''
import shutil
import sys
import os
import time
import copy
import pathlib

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from src.utils.pipeline import set_seed, load_dataset, load_model, \
    EarlyStopping
from src.utils.metrics import perframe_mAP
from src.configs.arguments import parse_args
from src.configs.conf import BASE_DIR


def main(args):
    # TODO: find a better validation synthetic dataset, e.g., a background
    #       condition never seen at training time
    # TODO: add image augmentation

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

    if args.log_dir is None:
        writer = SummaryWriter()
    else:
        log_dir = os.path.join(os.getcwd(), 'runs', args.log_dir)
        if os.path.isdir(log_dir):
            out = None
            while out not in ['y', 'n']:
                out = input('The folder '+log_dir+' already exists, do you '
                            'want to delete it and create a new one? [y/n]')
            if out == 'y':
                shutil.rmtree(log_dir)
            elif out == 'n':
                raise Exception('STOP EXECUTION... USE A DIFFERENT VALUE '
                                'FOR --log_dir')

        writer = SummaryWriter(log_dir=log_dir)
    print('LOGGING METRICS AT: {}'.format(writer.log_dir))

    command = 'python3 ' + ' '.join(sys.argv)
    with open(os.path.join(writer.log_dir, 'log.txt'), 'w') as f:
        f.write('--seed: {}\n'.format(args.seed))
        f.write('--train_test_split_seed: {}\n'.format(args.train_test_split_seed))
        f.write('--val_test_split_seed: {}\n'.format(args.val_test_split_seed))
        f.write('--train_test_split_size: '.format(args.train_test_split_size))
        f.write('--val_test_split_size: {}\n\n'.format(args.val_test_split_size))
        f.write(command + '\n')

    dataloader = load_dataset(args)

    # DIRTY TRICK: since we want to evaluate the accuracy at video granularity
    #              instead of frame granularity, we use the SingleSourceVideo
    #              dataset instead of the SingleSourceImage dataset.
    #              Therefore we need to replace 'val' dataset accordingly.
    appo_args = copy.deepcopy(args)
    appo_args.dataset_type = 'SingleSourceVideo'
    appo_args.batch_size = 1        # NEEDED TO HAVE ALL THE BELOW CODE WORKING
    dataloader_val_video = load_dataset(appo_args)['val']
    dataloader['val'] = dataloader_val_video

    if args.synthetic:
        # In case of training on synthetic data, we do the validation both
        # on synthetic and real ihannes data.
        phases.append('val_real')

        # WARNING: Currently hard-coded: we force here the same split and seed
        #          that we usually use when training on real data,
        #          such that the generated train-val-test subsets are the same
        appo_args = copy.deepcopy(args)
        appo_args.train_test_split_size = 0.7
        appo_args.val_test_split_size = 0.5
        appo_args.train_test_split_seed = 1
        appo_args.val_test_split_seed = 1
        appo_args.dataset_base_folder = os.path.join(
            os.getcwd(), 'data', 'real', 'frames', 'iHannesDataset'
        )
        appo_args.synthetic = False
        # Since we want to evaluate the video accuracy, we necessarily need the
        # SingleSourceVideo instead of the SingleSourceImage dataset
        appo_args.dataset_type = 'SingleSourceVideo'
        appo_args.batch_size = 1        # FOR THE SAME REASON AS ABOVE

        dataloader_val_real = load_dataset(appo_args)['val']

        dataloader['val_real'] = dataloader_val_real

    classes = args.data_info[args.output + 's']

    model = load_model(args)

    if args.parallel:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        else:
            print('WARNING: Only one GPU found, running on single GPU')

    start_epoch = 1
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_epoch += 1
        args.epochs += 1

    softmax = nn.Softmax(dim=1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.patience_lrscheduler, verbose=True
    )
    early_stopping = EarlyStopping(args.patience_earlystopping, delta=0)
    criterion = nn.CrossEntropyLoss().to(device)

    # We take the model that performs better on the video accuracy without
    # background metric.
    best_val_video_accuracywobackgr = -1
    best_epoch = -1

    phases = ['train', 'val']

    model = model.to(device)
    for epoch in range(start_epoch, start_epoch + args.epochs, 1):
        start = time.time()
        loss_epoch = {phase: 0 for phase in phases}
        perframe_accuracy_epoch = {phase: 0 for phase in phases}
        scores_epoch = {phase: [] for phase in phases}
        targets_epoch = {phase: [] for phase in phases}
        video_accuracy = {}
        video_accuracy['w_backgr'] = {phase: 0 for phase in phases}
        video_accuracy['w/o_backgr'] = {phase: 0 for phase in phases}

        for phase in phases:
            is_training = True if phase == 'train' else False
            model.train(is_training)
            dataloader[phase].dataset.dataset.train(is_training)

            with torch.set_grad_enabled(is_training):
                for batch_idx, (frames, grasp_types, preshapes, instances, _, _) \
                        in enumerate(dataloader[phase], start=1):

                    if not is_training:
                        if frames.shape[0] != 1:
                            raise RuntimeError('Wrong batch size: in the {} '
                                               'phase, you need to use '
                                               'SingleSourceVideo dataset with'
                                               ' batch_size 1 to make the '
                                               'code below working'
                                               .format(phase))
                        # frames.shape (batch_size==1, num_frames_in_video==90, 3, 224, 224)
                        # grasp_types (batch_size==1, num_frames_in_video==90)
                        frames = frames.squeeze(0)
                        grasp_types = grasp_types.squeeze(0)
                        preshapes = preshapes.squeeze(0)
                        instances = instances.squeeze(0)

                    if args.output == 'grasp_type':
                        targets = grasp_types
                    elif args.output == 'preshape':
                        targets = preshapes
                    elif args.output == 'instance':
                        targets = instances
                    else:
                        raise ValueError(
                            'Not yet implemented for --output {}'
                            .format(args.output)
                        )

                    # targets.shape (batch_size)
                    # frames.shape (batch_size, 3, 224, 224)
                    #              or (batch_size, feat_vect_dim)
                    batch_size = frames.shape[0]
                    frames = frames.to(device)

                    if is_training:
                        optimizer.zero_grad()

                    scores = model(frames)
                    # scores.shape (batch_size, num_classes)

                    scores_epoch[phase].append(softmax(scores.cpu()))
                    targets_epoch[phase].append(targets)

                    targets = targets.to(device)
                    loss = criterion(scores, targets)

                    if is_training:
                        loss.backward()
                        optimizer.step()

                    loss_epoch[phase] += loss.item() * batch_size

                    preds = scores.argmax(dim=1)

                    perframe_accuracy_epoch[phase] += (preds == targets).sum().item()

                    if not is_training:
                        # Evaluate at video granularity instead of frame
                        # granularity

                        # Convert from per frame label to single video label
                        idx_no_grasp = 0
                        appo = torch.unique(targets)
                        video_target = appo[appo != idx_no_grasp]
                        if video_target.shape != (1,):
                            raise RuntimeError('Something went wrong with the '
                                               'labels vector received: it '
                                               'should contain only two '
                                               'indexes, the background and '
                                               'class indexes.')
                        video_target = video_target.item()

                        # Compute video accuracy, w/ and w/o background

                        # Video accuracy w/ background
                        video_pred, _ = torch.mode(preds)
                        video_pred = video_pred.item()
                        video_accuracy['w_backgr'][phase] += video_pred == video_target

                        # Video accuracy w/o background
                        preds_wo_backgr = preds[preds != idx_no_grasp]
                        if len(preds_wo_backgr) == 0:
                            # i.e. the model predicted always background
                            # along the whole video
                            video_pred = torch.tensor(0)
                        else:
                            # Video prediction performed via majority voting
                            # over all the non-background-predicted frames
                            # of the video
                            video_pred, _ = torch.mode(preds_wo_backgr)
                        video_pred = video_pred.item()
                        video_accuracy['w/o_backgr'][phase] += video_pred == video_target

                    if args.verbose:
                        print('[{:5}] Epoch: {:>3}/{:<3}  Iteration: {:<3}  '
                              'Loss: {:<3.4f}'
                              .format(phase, epoch, args.epochs, batch_idx,
                                      loss.item()))

        perframe_dataset_len = len(dataloader['val'].dataset) * \
            dataloader['val'].dataset.dataset._NUM_FRAMES_IN_VIDEO
        lr_sched.step(loss_epoch['val'] / perframe_dataset_len)
        retval_earlystopping = early_stopping(
            loss_epoch['val'] / perframe_dataset_len
        )

        scores_epoch = {phase: torch.cat(scores_epoch[phase]).detach().numpy() for phase in phases}
        targets_epoch = {phase: torch.cat(targets_epoch[phase]).detach().numpy() for phase in phases}

        mAP_results = {phase: perframe_mAP(scores_epoch[phase], targets_epoch[phase], classes) for phase in phases}

        end = time.time()

        log = 'Epoch: {:<3} |'.format(epoch)
        for phase in phases:
            if phase == 'train':
                str_results = '[{:<8}]  Loss: {:<3.4f}  ' \
                              'Per-frame_Accuracy: {:<3.1f}%  ' \
                              'Per-frame_mAP: {:<3.4f}% |\n{}'
                log += str_results.format(
                    phase,
                    loss_epoch[phase] / len(dataloader[phase].dataset),
                    perframe_accuracy_epoch[phase] / len(dataloader[phase].dataset) * 100,
                    mAP_results[phase]['mAP_valid_cls'] * 100,
                    ''.ljust(12)
                )
            else:
                str_results = '[{:<8}]  Loss: {:<3.4f}  ' \
                              'Per-frame_Accuracy: {:<3.1f}%  ' \
                              'Per-frame_mAP: {:<3.4f}%  ' \
                              'Video Accuracy: {:<3.4f}%  ' \
                              'Video Accuracy w/o backgr: {:<3.4f}% |\n{}'
                perframe_dataset_len = len(dataloader[phase].dataset) * \
                    dataloader[phase].dataset.dataset._NUM_FRAMES_IN_VIDEO
                log += str_results.format(
                    phase,
                    loss_epoch[phase] / perframe_dataset_len,
                    perframe_accuracy_epoch[phase] / perframe_dataset_len * 100,
                    mAP_results[phase]['mAP_valid_cls'] * 100,
                    video_accuracy['w_backgr'][phase] / len(dataloader[phase].dataset) * 100,
                    video_accuracy['w/o_backgr'][phase] / len(dataloader[phase].dataset) * 100,
                    ''.ljust(12)
                )
        log += 'Running time: {:3.1f}s'.format(end - start)

        if not args.suppress_epoch_print:
            print(log)
        with open(os.path.join(writer.log_dir, 'log.txt'), 'a+') as f:
            f.write(log + '\n')

        log = 'Epoch :{:<3} |'.format(epoch)
        for phase in phases:
            log += '[{:<5}]  '.format(phase)
            log += 'mAP_all_cls: {:<3.4f}%  -- '\
                .format(mAP_results[phase]['mAP_all_cls'] * 100)
            for cls in classes:
                if cls not in mAP_results[phase]['AP']:
                    continue
                log += '{} AP: {:<3.4f}%   '\
                    .format(cls, mAP_results[phase]['AP'][cls] * 100)
            log += ' |\n{}'.format(''.ljust(12))
        with open(os.path.join(writer.log_dir, 'log_mAP.txt'), 'a+') as f:
            f.write(log + '\n')

        for phase in phases:
            if phase == 'train':
                writer.add_scalars(
                    'Perframe/Loss_epoch',
                    {phase: loss_epoch[phase] / len(dataloader[phase].dataset)},
                    epoch
                )
                writer.add_scalars(
                    'Perframe/Accuracy_epoch',
                    {phase: perframe_accuracy_epoch[phase] / len(dataloader[phase].dataset) * 100},
                    epoch
                )
                writer.add_scalars(
                    'Perframe/mAP_epoch',
                    {phase: mAP_results[phase]['mAP_valid_cls'] * 100},
                    epoch
                )
            else:
                perframe_dataset_len = len(dataloader[phase].dataset) * \
                    dataloader[phase].dataset.dataset._NUM_FRAMES_IN_VIDEO
                writer.add_scalars(
                    'Perframe/Loss_epoch',
                    {phase: loss_epoch[phase] / perframe_dataset_len},
                    epoch
                )
                writer.add_scalars(
                    'Perframe/Accuracy_epoch',
                    {phase: perframe_accuracy_epoch[phase] / perframe_dataset_len * 100},
                    epoch
                )
                writer.add_scalars(
                    'Perframe/mAP_epoch',
                    {phase: mAP_results[phase]['mAP_valid_cls'] * 100},
                    epoch
                )
                writer.add_scalar(
                    'Pervideo/Accuracy_W_Backgr_epoch',
                    video_accuracy['w_backgr'][phase] / len(dataloader[phase].dataset) * 100,
                    epoch
                )
                writer.add_scalar(
                    'Pervideo/Accuracy_Wo_Backgr_epoch',
                    video_accuracy['w/o_backgr'][phase] / len(dataloader[phase].dataset) * 100,
                    epoch
                )
        writer.close()

        appo_phase = 'val_real' if args.synthetic else 'val'
        if best_val_video_accuracywobackgr < (video_accuracy['w/o_backgr'][appo_phase] / len(dataloader[appo_phase].dataset) * 100):
            best_val_video_accuracywobackgr = video_accuracy['w/o_backgr'][appo_phase] / len(dataloader[appo_phase].dataset) * 100
            best_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'video_accuracywobackgr': best_val_video_accuracywobackgr,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                },
                os.path.join(writer.log_dir, 'best_model.pth')
            )

        if retval_earlystopping:
            log = 'Training will be stopped due to validation '\
                  'loss not decreasing after {} epochs'.\
                  format(args.patience_earlystopping)
            print(log)
            with open(os.path.join(writer.log_dir, 'log.txt'), 'a+') as f:
                f.write(log + '\n')

            break

    log = '--- Best validation{} Video Accuracy w/o backgr is {:.1f}% ' \
          'obtained at epoch {:<3} ---'\
          .format('(on real dataset)' if args.synthetic else '',
                  best_val_video_accuracywobackgr,
                  best_epoch)
    print(log)
    with open(os.path.join(writer.log_dir, 'log.txt'), 'a+') as f:
        f.write(log)


if __name__ == '__main__':
    cur_base_dir = os.getcwd()
    cur_base_dir = os.path.basename(cur_base_dir)
    if cur_base_dir != BASE_DIR:
        raise Exception(
            'Wrong base dir, this file must be run from {} directory.'
            .format(BASE_DIR)
        )

    main(parse_args())
