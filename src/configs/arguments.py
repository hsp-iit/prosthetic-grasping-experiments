import os
import argparse
import random
import yaml

from src.configs.conf import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_info_file_path', default='data/data_info.yaml',
                        type=str)

    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to the model checkpoint file. The path must'
                             ' be specified relative to the repo base folder.')

    parser.add_argument('--verbose', default=False, action='store_true')

    parser.add_argument('--parallel', default=False, action='store_true')
    parser.add_argument('--gpu', default='0', type=str,
                        help='Specify on which gpus to train, '
                             'separated by comma')

    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--suppress_epoch_print', default=False,
                        action='store_true')

    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--lr', default=5e-04, type=float)
    parser.add_argument('--patience_lrscheduler', default=6, type=int)
    parser.add_argument('--patience_earlystopping', default=10, type=int)
    parser.add_argument('--weight_decay', default=5e-04, type=float)

    parser.add_argument('--seed', default=None, type=int,
                        help='Runtime seed, i.e., the computations are the '
                             'same across different executions if the seed is '
                             'the same. If not specified, a random seed '
                             'is sampled')
    parser.add_argument('--train_test_split_seed', default=1, type=int,
                        help='Split seed, i.e., the training-test split is the'
                             ' same across different executions if the seed is'
                             ' the same')
    parser.add_argument('--val_test_split_seed', default=1, type=int,
                        help='Split seed, i.e., the validation-test split is '
                             'the same across different executions if the seed'
                             ' is the same')
    parser.add_argument('--train_test_split_size', default=0.7, type=float,
                        help='The percentage of the whole dataset to be in the'
                             'training set (the remaining part will be further'
                             'splitted into validation and test set according '
                             'to --val_test_split_size')
    parser.add_argument('--val_test_split_size', default=0.5, type=float,
                        help='The percentage of the non-training set to be in '
                             'the validation set, the remaining part will be '
                             'in the test set (see also '
                             '--train_test_split_size)')

    parser.add_argument('--source', default='Wrist_d435', type=str)

    parser.add_argument('--dataset_type', default='SingleSourceImage',
                        type=str)

    parser.add_argument('--split', default='random', type=str)

    parser.add_argument('--input', default='rgb', type=str)
    parser.add_argument('--output', default='preshape', type=str)

    parser.add_argument('--model', default='cnn', type=str)
    parser.add_argument('--feature_extractor', default='mobilenet_v2', type=str)
    parser.add_argument('--pretrain', default='imagenet', type=str)
    parser.add_argument('--freeze_all_conv_layers', default=False,
                        action='store_true')

    parser.add_argument('--rnn_type', default='lstm', type=str)
    parser.add_argument('--rnn_hidden_size', default=128, type=int)

    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--fc_layers', default=None, type=int,
                        help='The number of fully connected layers between the'
                             ' feature extractor and the final fully '
                             'connected classifier layer')
    parser.add_argument('--fc_neurons', default=None, type=str,
                        help='The number of neurons per each layer, separated '
                             'by comma')

    parser.add_argument('--test_type', default=None, type=str,
                        help='On which test set to evaluate the model. This '
                             'argument is used only in '
                             'src/tools/*model_name*/eval.py')

    parser.add_argument('--video_specs', default=None, type=str,
                        help='To specify the video on which to evaluate the '
                             'model. A video of our dataset is uniquely '
                             'identified by the instance in the video, the '
                             'preshape label and the sequence number, '
                             'therefore a sample usage is: '
                             '--video_specs *instance_name*,*preshape_class*,*rgb_xxxxx*.'
                             ' This argument is used only in '
                             'src/tools/*model_name*/eval_single_video.py')

    parser.add_argument('--show_perframe_preds', default=False,
                        action='store_true',
                        help='If enabled, the video frames and corresponding'
                             ' predictions are shown. This argument is used '
                             'only in '
                             'src/tools/*model_name*/eval_single_video.py')

    parser.add_argument('--train', default=False, action='store_true',
                        help='If enabled, the training set is considered. '
                             'This argument is used only in '
                             'src/tools/show_stats.py and '
                             'src/tools/show_video_paths.py')
    parser.add_argument('--test', default=False, action='store_true',
                        help='If enabled, the test set is considered. '
                             'This argument is used only in '
                             'src/tools/show_stats.py and '
                             'src/tools/show_video_paths.py')

    parser.add_argument('--category', default=None, type=str,
                        help='The category to consider. This argument is used '
                             'only in src/tools/show_video_paths.py')
    parser.add_argument('--instance', default=None, type=str,
                        help='The instance to consider. This argument is used'
                             ' only in src/tools/show_video_paths.py')

    parser.add_argument('--synthetic', default=False, action='store_true',
                        help='Enable it when the dataset specified via '
                             '--dataset_name arguments is a synthetic dataset')

    parser.add_argument('--dataset_name', default=None, type=str,
                        help='The base folder of the dataset. The path must be'
                             ' specified relative to the repo base folder '
                             '(i.e., prosthetic-grasping-experiments folder))')

    parser.add_argument('--from_features', default=False, action='store_true',
                        help='If enabled, pre-extracted features are used, and'
                             ' only the model part after the feature extractor'
                             ' is trained')

    parser.add_argument('--log_dir', type=str, default=None,
                        help='Specify the folder name where to save logs and '
                             'model checkpoint. The complete path will be: '
                             'prosthetic-grasping-experiments/runs/log_dir. '
                             'If a folder with this name already exists, '
                             'it will be deleted and a new one is created.')

    # TODO
    parser.add_argument('--balance', default=None, type=int,
                        help='NOT IMPLEMENTED YET')

    # TODO
    parser.add_argument('--weighted_loss', default=None, type=int,
                        help='A weighted cross-entropy loss is used, where '
                             'each class is weighted according to the '
                             'inverse of its cardinality.'
                             'NOT IMPLEMENTED YET')

    # TODO
    parser.add_argument('--temporal_augmentation', default=False, action='store_true',
                        help='Used only in the temporal model (e.g. rnn). The '
                             'temporal augmentation upsample, downsample, trim'
                             ' and loop the frames of the video.'
                             'NOT IMPLEMENTED YET')

    # TODO
    parser.add_argument('--temporal_sampling', default=False, action='store_true',
                        help='Used only in the non-temporal model (e.g. cnn).'
                             'The temporal sampling randomly sample frames '
                             'from the whole video.'
                             'NOT IMPLEMENTED YET')

    return build_args(check_args(parser.parse_args()))


def check_args(args):
    if not os.path.exists(args.data_info_file_path):
        raise ValueError(RAISE_VALUE_ERROR_STRING.format(
            args.data_info_file_path, '--data_info_file_path',
            '[data/data_info.yaml]')
        )

    if args.checkpoint is not None:
        if not args.checkpoint.endswith('.pth'):
            raise ValueError('Wrong model checkpoint file, it must be a .pth file')
        if not os.path.exists(args.checkpoint):
            raise ValueError('File {} not found. The path must be specified '
                             'relative to the repo base folder '
                             '(i.e., prosthetic-grasping-experiments).'
                             .format(args.checkpoint))
        if args.log_dir is not None and \
                args.log_dir != args.checkpoint.split('/')[-2]:
            raise ValueError('Wrong combination of arguments: the parent '
                             'folder of the file specified in --checkpoint '
                             'must match the folder name specified in '
                             '--log_dir, e.g., '
                             '--checkpoint runs/log_dir_name/best_model.pth '
                             '--log_dir log_dir_name. All the paths must be '
                             'specified relative to the repo base folder '
                             '(i.e., prosthetic-grasping-experiments folder).')

    if args.source not in SOURCE:
        raise ValueError(
            RAISE_VALUE_ERROR_STRING.format(args.source, '--source', SOURCE)
        )

    if args.train_test_split_size < 0 or args.train_test_split_size > 1:
        raise ValueError(
            'Wrong value for --train_test_split_size argument: it must be in'
            'range (0, 1)'
        )

    if args.val_test_split_size < 0 or args.val_test_split_size > 1:
        raise ValueError(
            'Wrong value for --val_test_split_size argument: it must be in'
            'range (0, 1)'
        )

    if args.dataset_type not in DATASET_TYPE:
        raise ValueError(
            RAISE_VALUE_ERROR_STRING.format(args.dataset_type,
                                            '--dataset_type', DATASET_TYPE)
        )

    if args.split not in SPLIT:
        raise ValueError(
            RAISE_VALUE_ERROR_STRING.format(args.split, '--split', SPLIT)
        )

    if args.synthetic and args.split != 'random':
        raise ValueError('Wrong combination of arguments: --synthetic and '
                         '--split=={}. The synthetic dataset can have '
                         'only --split==random'.format(args.split))

    if args.input not in INPUT:
        raise ValueError(
            RAISE_VALUE_ERROR_STRING.format(args.input, '--input', INPUT)
        )
    if args.output not in OUTPUT:
        raise ValueError(
            RAISE_VALUE_ERROR_STRING.format(args.output, '--output', OUTPUT)
        )

    if args.model not in MODEL:
        raise ValueError(
            RAISE_VALUE_ERROR_STRING.format(args.model, '--model', MODEL)
        )
    if args.model == 'cnn':
        if not args.dataset_type.endswith('Image'):
            raise ValueError('Wrong combination of arguments: --model==cnn and'
                             ' --dataset_type=={}. When using --model==cnn, a '
                             'dataset type ending with the Image suffix has '
                             'to be used.'.format(args.dataset_type))
    elif args.model == 'cnn_rnn':
        if not args.dataset_type.endswith('Video'):
            raise ValueError('Wrong combination of arguments: --model==cnn_rnn'
                             'and --dataset_type=={}. When using '
                             '--model==cnn_rnn, a dataset type ending with the'
                             ' Image suffix has '
                             'to be used.'.format(args.dataset_type))
    else:
        raise ValueError('Not yet implemented for --model {}'
                         .format(args.model))
    if args.feature_extractor not in FEATURE_EXTRACTOR:
        raise ValueError(
            RAISE_VALUE_ERROR_STRING.format(args.feature_extractor,
                                            '--feature_extractor',
                                            FEATURE_EXTRACTOR)
        )
    if args.pretrain not in PRETRAIN:
        raise ValueError(
            RAISE_VALUE_ERROR_STRING.format(args.pretrain, '--pretrain',
                                            PRETRAIN)
        )

    if args.pretrain == 'from_scratch' and args.freeze_all_conv_layers:
        raise ValueError('Wrong combination of arguments: '
                         '--pretrain==from_scratch and '
                         '--freeze_all_conv_layers. It makes no sense to '
                         'freeze all conv layers when training from scratch')

    if args.from_features and not args.freeze_all_conv_layers:
        raise ValueError('Wrong combination of arguments: '
                         '--freeze_all_conv_layers and --from_features. When '
                         'starting from features, also '
                         '--freeze_all_conv_layers has to be set. '
                         'You can not start from features if '
                         'you want to train the conv layers.')

    if args.rnn_type not in RNN_TYPE:
        raise ValueError(
            RAISE_VALUE_ERROR_STRING.format(args.rnn_type, '--rnn_type',
                                            RNN_TYPE)
        )

    if args.fc_layers is not None and args.fc_neurons is not None:
        if args.model != 'cnn':
            raise ValueError('Wrong combination of arguments: --fc_layers and '
                             '--fc_neurons arguments can be used '
                             'only with --model==cnn')

        if args.fc_layers != len(args.fc_neurons.split(',')):
            raise ValueError('Wrong usage of --fc_layers and --fc_neurons. '
                             'Sample usage: --num_layers 3 '
                             '--num_neurons 1024,2048,1024')
    elif args.fc_layers is not None or args.fc_neurons is not None:
        raise ValueError('Wrong combination of arguments: --fc_layers and '
                         '--fc_neurons must be used together or not used at '
                         'all')

    if args.test_type is not None and args.test_type not in TEST_TYPE:
        raise ValueError(
            RAISE_VALUE_ERROR_STRING.format(args.test_type, '--test_type',
                                            TEST_TYPE)
        )

    if args.dataset_name is None:
        raise ValueError('No dataset specified, you have to specify the '
                         'dataset name via --dataset_name argument.'
                         'The dataset name is the dataset base folder.'
                         'An example path to the dataset base folder is the '
                         'following: {}'
                         .format(os.path.join(os.getcwd(),
                                              'data',
                                              'real',
                                              'frames',
                                              'iHannesDataset'
                                              )
                                 )
                         )

    if args.log_dir is not None and args.checkpoint is not None and \
            args.log_dir != args.checkpoint.split('/')[-2]:
        raise ValueError('Wrong combination of arguments: the parent folder of'
                         ' the file specified in --checkpoint must match the '
                         'folder name specified in --log_dir , e.g., '
                         '--checkpoint log_dir_name/best_model.pth '
                         '--log_dir log_dir_name. All the paths must be '
                         'specified relative to the repo base folder.')

    return args


def build_args(args):
    if args.seed is None:
        args.seed = random.randint(0, 2**16)

    with open(args.data_info_file_path, 'r') as f:
        args.data_info = yaml.full_load(f)

    # Add background class
    args.data_info['preshapes'].insert(0, 'no_grasp')
    args.data_info['grasp_types'].insert(0, 'no_grasp')
    args.data_info['instances'].insert(0, 'no_grasp')
    args.data_info['wrist_pss'].insert(0, 'no_grasp')
    args.data_info['preshape_wrist_pss'].insert(0, 'no_grasp')

    args.num_classes = len(args.data_info[args.output + 's'])

    synthetic_or_real = 'synthetic' if args.synthetic else 'real'
    args.dataset_base_folder = os.path.join(
        os.getcwd(), 'data', synthetic_or_real, 'frames', args.dataset_name
    )

    if not os.path.isdir(args.dataset_base_folder):
        raise ValueError('The dataset folder does not exist: {}'
                         .format(args.dataset_base_folder))

    if args.from_features:
        features_folder = os.path.join(os.getcwd(), 'data', synthetic_or_real,
                                       'features', args.feature_extractor,
                                       args.dataset_name)
        if not os.path.isdir(features_folder):
            raise ValueError('The dataset features folder does not exist: {}'
                             .format(features_folder))

    return args
