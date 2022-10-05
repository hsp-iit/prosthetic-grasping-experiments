'''
python3 src/tools/show_video_paths.py --from_features \
    --freeze_all_conv_layers --dataset_name iHannesDataset \
    --dataset_type SingleSourceVideo --model cnn_rnn --train
python3 src/tools/show_video_paths.py --from_features \
    --freeze_all_conv_layers --dataset_name iHannesDataset \
    --dataset_type SingleSourceVideo --model cnn_rnn --test
python3 src/tools/show_video_paths.py --from_features \
    --freeze_all_conv_layers --dataset_name iHannesDataset \
    --dataset_type SingleSourceVideo --model cnn_rnn --test --test_type test_seated
python3 src/tools/show_video_paths.py --from_features \
    --freeze_all_conv_layers --dataset_name iHannesDataset \
    --dataset_type SingleSourceVideo --model cnn_rnn --category dispenser
python3 src/tools/show_video_paths.py --from_features \
    --freeze_all_conv_layers --dataset_name iHannesDataset \
    --dataset_type SingleSourceVideo --model cnn_rnn --instance 001_chips_can
'''
import os
import sys

sys.path.append(os.getcwd())
from src.utils.pipeline import load_dataset
from src.configs.arguments import parse_args


def main(args):
    if args.train == args.test == False:
        raise ValueError('--train and --test are both set to False. '
                         'Activate one among them!')
    if args.train == args.test == True:
        raise ValueError('--train and --test are both set to True. '
                         'You can only activate one among them!')
    if args.dataset_type != 'SingleSourceVideo':
        raise NotImplementedError('Wrong --dataset_type value. Currently '
                                  'only SingleSourceVideo is supported.')

    args.batch_size = 1
    dataloader = load_dataset(args)
    phases = list(dataloader.keys())

    grasp_types_name_list = args.data_info['grasp_types']

    for p in phases:
        print('===== {} ====='.format(p))
        for _, grasp_type, _, _, _, video_path in dataloader[p]:
            # video_path.shape (batch_size)
            for v_p, g_t in zip(video_path, grasp_type):
                if args.category is not None and args.category not in v_p:
                    continue
                if args.instance is not None and args.instance not in v_p:
                    continue

                g_t_idx = g_t.max()
                g_t_name = grasp_types_name_list[g_t_idx]
                print('{:150s} with GRASP TYPE: {}'.format(v_p, g_t_name))
        print()


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
