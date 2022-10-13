'''
python3 src/tools/show_stats.py --model cnn_rnn \
--dataset_type SingleSourceVideo --train --dataset_name iHannesDataset \
--from_features --freeze_all_conv_layers
python3 src/tools/show_stats.py --model cnn_rnn \
--dataset_type SingleSourceVideo --test --dataset_name iHannesDataset \
--from_features --freeze_all_conv_layers
python3 src/tools/show_stats.py --model cnn_rnn \
--dataset_type SingleSourceVideo --test --test_type test_different_velocity \
--dataset_name iHannesDataset --from_features --freeze_all_conv_layers
'''
import os
import copy
import sys

sys.path.append(os.getcwd())
from src.utils.stats import plot_histogram_grasp_types, \
    plot_histogram_preshapes, plot_histogram_instances,\
    plot_matrix_instance_grasp_type_numvideos, \
    plot_matrix_instance_preshape_numvideos
from src.configs.arguments import parse_args
from src.configs.conf import BASE_DIR


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

    plot_histogram_grasp_types(copy.deepcopy(args))
    plot_histogram_preshapes(copy.deepcopy(args))

    plot_matrix_instance_grasp_type_numvideos(copy.deepcopy(args))
    plot_matrix_instance_preshape_numvideos(copy.deepcopy(args))

    plot_histogram_instances(copy.deepcopy(args))


if __name__ == '__main__':
    cur_base_dir = os.getcwd()
    cur_base_dir = os.path.basename(cur_base_dir)
    if cur_base_dir != BASE_DIR:
        raise Exception(
            'Wrong base dir, this file must be run from {} directory.'
            .format(BASE_DIR)
        )

    main(parse_args())
