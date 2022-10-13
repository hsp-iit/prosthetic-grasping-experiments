import os
import itertools
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils.pipeline import load_dataset


def _autolabel(rects, values):
    '''
    To put the value on top of its corresponding column
    '''
    for ii, rect in enumerate(rects):
        height = rect.get_height()
        # attach some text labels
        plt.text(rect.get_x() + rect.get_width() / 2., 1.0 * height, '%s' % (values[ii]), ha='center', va='bottom')


def plot_histogram_grasp_types(args, figsize=(6, 4), show_image=True):
    '''
    Plot the number of videos per grasp_type
    '''
    dataloader = load_dataset(args)

    if args.train:
        phases = ['train', 'val']
    else:
        phases = ['test']

    for phase in phases:
        idx_no_grasp = 0
        grasp_types_name_list = copy.deepcopy(args.data_info['grasp_types'])
        grasp_type_to_count = {g_t: 0 for g_t in grasp_types_name_list}

        for _, grasp_type, _, _, _, _ in dataloader[phase]:
            # grasp.type.shape (batch_size, num_frames_in_video)
            for b in range(grasp_type.shape[0]):
                # Convert from per frame label to single video label
                appo = torch.unique(grasp_type[b])
                grasp_type_video = appo[appo != idx_no_grasp]
                if grasp_type_video.shape != (1,):
                    raise RuntimeError('Something went wrong with the '
                                       'labels vector received: it '
                                       'should contain only two '
                                       'indexes, the background and '
                                       'class indexes.')
                grasp_type_video = grasp_type_video.item()

                grasp_type_name = grasp_types_name_list[grasp_type_video]
                grasp_type_to_count[grasp_type_name] += 1

        # Remove the grasp types that have no videos, i.e., count == 0
        grasp_types_to_remove = []
        for g_t, count in grasp_type_to_count.items():
            if count == 0:
                grasp_types_to_remove.append(g_t)
        for g_t_to_remove in grasp_types_to_remove:
            del grasp_type_to_count[g_t_to_remove]
            grasp_types_name_list.remove(g_t_to_remove)

        # For each grasp_type generate a color: the more elements a grasp_type
        # has, the darker the color will be
        num_grasp_types = len(grasp_type_to_count)
        colors = plt.get_cmap('Blues')(np.linspace(0, 1, num_grasp_types))[::-1]
        ordered_grasp_type_to_count = sorted(
            grasp_type_to_count.items(), key=lambda x: x[1], reverse=True
        )
        ordered_grasp_types = [grasp_type_and_count[0] for grasp_type_and_count in ordered_grasp_type_to_count]
        new_seq_colors = np.zeros_like(colors)
        for idx, grasp_type in enumerate(ordered_grasp_types):
            new_seq_colors[grasp_types_name_list.index(grasp_type)] = colors[idx]

        plt.figure(figsize=figsize)
        rects = plt.bar(np.arange(len(grasp_types_name_list)),
                        grasp_type_to_count.values(),
                        color=new_seq_colors)
        _autolabel(rects, list(grasp_type_to_count.values()))
        plt.xticks(np.arange(len(grasp_types_name_list)),
                   grasp_types_name_list,
                   rotation=90)
        plt.xlabel('Grasp type')
        plt.ylabel('# of videos')
        count = 0
        for v in grasp_type_to_count.values():
            count += v
        plt.title('Phase: '+phase+',   tot. num. of videos: '+str(count))
        plt.tight_layout()
        hist_dir = os.path.join('stats', 'histogram')
        if not os.path.isdir(hist_dir):
            os.makedirs(hist_dir)
        plt.savefig(os.path.join(hist_dir, 'grasp_types_phase_'+phase+'.png'))
        if show_image:
            plt.show()
        plt.close()


def plot_histogram_preshapes(args, figsize=(6, 4), show_image=True):
    '''
    Plot the number of videos per preshape
    '''
    dataloader = load_dataset(args)

    if args.train:
        phases = ['train', 'val']
    else:
        phases = ['test']

    for phase in phases:
        idx_no_grasp = 0
        preshapes_name_list = copy.deepcopy(args.data_info['preshapes'])
        preshape_to_count = {g_t: 0 for g_t in preshapes_name_list}

        for _, _, preshape, _, _, _ in dataloader[phase]:
            # preshape.shape (batch_size, num_frames_in_video)
            for b in range(preshape.shape[0]):
                # Convert from per frame label to single video label
                appo = torch.unique(preshape[b])
                preshape_video = appo[appo != idx_no_grasp]
                if preshape_video.shape != (1,):
                    raise RuntimeError('Something went wrong with the '
                                       'labels vector received: it '
                                       'should contain only two '
                                       'indexes, the background and '
                                       'class indexes.')
                preshape_video = preshape_video.item()

                preshape_name = preshapes_name_list[preshape_video]
                preshape_to_count[preshape_name] += 1

        # Remove the preshape that have no videos, i.e., count == 0
        preshapes_to_remove = []
        for g_t, count in preshape_to_count.items():
            if count == 0:
                preshapes_to_remove.append(g_t)
        for g_t_to_remove in preshapes_to_remove:
            del preshape_to_count[g_t_to_remove]
            preshapes_name_list.remove(g_t_to_remove)

        # For each preshape generate a color: the more elements a preshape
        # has, the darker the color will be
        num_preshapes = len(preshape_to_count)
        colors = plt.get_cmap('Blues')(np.linspace(0, 1, num_preshapes))[::-1]
        ordered_preshape_to_count = sorted(
            preshape_to_count.items(), key=lambda x: x[1], reverse=True
        )
        ordered_preshapes = [preshape_and_count[0] for preshape_and_count in ordered_preshape_to_count]
        new_seq_colors = np.zeros_like(colors)
        for idx, preshape in enumerate(ordered_preshapes):
            new_seq_colors[preshapes_name_list.index(preshape)] = colors[idx]

        plt.figure(figsize=figsize)
        rects = plt.bar(np.arange(len(preshapes_name_list)),
                        preshape_to_count.values(),
                        color=new_seq_colors)
        _autolabel(rects, list(preshape_to_count.values()))
        plt.xticks(np.arange(len(preshapes_name_list)),
                   preshapes_name_list,
                   rotation=90)
        plt.xlabel('Pre-shape')
        plt.ylabel('# of videos')
        count = 0
        for v in preshape_to_count.values():
            count += v
        plt.title('Phase: '+phase+',   tot. num. of videos: '+str(count))
        plt.tight_layout()
        hist_dir = os.path.join('stats', 'histogram')
        if not os.path.isdir(hist_dir):
            os.makedirs(hist_dir)
        plt.savefig(os.path.join(hist_dir, 'preshapes_phase_'+phase+'.png'))
        if show_image:
            plt.show()
        plt.close()


def plot_histogram_instances(args, figsize=(6, 4), show_image=True):
    '''
    Plot the number of videos per instance
    '''
    dataloader = load_dataset(args)

    if args.train:
        phases = ['train', 'val']
    else:
        phases = ['test']

    for phase in phases:
        idx_no_grasp = 0
        instances_name_list = copy.deepcopy(args.data_info['instances'])
        instance_to_count = {g_t: 0 for g_t in instances_name_list}

        for _, _, _, instance, _, _ in dataloader[phase]:
            # instance.shape (batch_size, num_frames_in_video)
            for b in range(instance.shape[0]):
                # Convert from per frame label to single video label
                appo = torch.unique(instance[b])
                instance_video = appo[appo != idx_no_grasp]
                if instance_video.shape != (1,):
                    raise RuntimeError('Something went wrong with the '
                                       'labels vector received: it '
                                       'should contain only two '
                                       'indexes, the background and '
                                       'class indexes.')
                instance_video = instance_video.item()

                instance_name = instances_name_list[instance_video]
                instance_to_count[instance_name] += 1

        # Remove the instances that have no videos, i.e., count == 0
        instances_to_remove = []
        for g_t, count in instance_to_count.items():
            if count == 0:
                instances_to_remove.append(g_t)
        for g_t_to_remove in instances_to_remove:
            del instance_to_count[g_t_to_remove]
            instances_name_list.remove(g_t_to_remove)

        # For each instance generate a color: the more elements a instance
        # has, the darker the color will be
        num_instances = len(instance_to_count)
        colors = plt.get_cmap('Blues')(np.linspace(0, 1, num_instances))[::-1]
        ordered_instance_to_count = sorted(
            instance_to_count.items(), key=lambda x: x[1], reverse=True
        )
        ordered_instances = [instance_and_count[0] for instance_and_count in ordered_instance_to_count]
        new_seq_colors = np.zeros_like(colors)
        for idx, instance in enumerate(ordered_instances):
            new_seq_colors[instances_name_list.index(instance)] = colors[idx]

        plt.figure(figsize=figsize)
        rects = plt.bar(np.arange(len(instances_name_list)),
                        instance_to_count.values(),
                        color=new_seq_colors)
        _autolabel(rects, list(instance_to_count.values()))
        plt.xticks(np.arange(len(instances_name_list)),
                   instances_name_list,
                   rotation=90)
        plt.xlabel('Instance')
        plt.ylabel('# of videos')
        count = 0
        for v in instance_to_count.values():
            count += v
        plt.title('Phase: '+phase+',   tot. num. of videos: '+str(count))
        plt.tight_layout()
        hist_dir = os.path.join('stats', 'histogram')
        if not os.path.isdir(hist_dir):
            os.makedirs(hist_dir)
        plt.savefig(os.path.join(hist_dir, 'instances_phase_'+phase+'.png'))
        if show_image:
            plt.show()
        plt.close()


def plot_matrix_instance_grasp_type_numvideos(args,
                                              figsize=(6, 4),
                                              show_image=True):
    '''
    Plot a matrix instance vs. grasp type where each cell has
    the number of videos
    '''
    dataloader = load_dataset(args)

    if args.train:
        phases = ['train', 'val']
    else:
        phases = ['test']

    for phase in phases:
        instances_name_list = copy.deepcopy(args.data_info['instances'])
        grasp_types_name_list = copy.deepcopy(args.data_info['grasp_types'])
        instance_and_grasp_type_to_count = {}
        for i in instances_name_list:
            instance_and_grasp_type_to_count[i] = {}
            for p in grasp_types_name_list:
                instance_and_grasp_type_to_count[i][p] = 0

        for _, grasp_type, _, instance, _, _ in dataloader[phase]:
            # shape (batch_size, num_frames_in_video)

            # Convert from per frame label to single video label
            # WARNING: this works only if the no_grasp class has index 0
            grasp_type = grasp_type.max(dim=1)[0]
            instance = instance.max(dim=1)[0]

            for grsp_idx, inst_idx in zip(grasp_type, instance):
                grsp_name = grasp_types_name_list[grsp_idx.item()]
                inst_name = instances_name_list[inst_idx.item()]

                instance_and_grasp_type_to_count[inst_name][grsp_name] += 1

        # Remove the instances having no videos
        for inst_name in instances_name_list:
            count = 0
            for grsp_name in instance_and_grasp_type_to_count[inst_name]:
                count += instance_and_grasp_type_to_count[inst_name][grsp_name]
            if count == 0:
                del instance_and_grasp_type_to_count[inst_name]

        instances = list(instance_and_grasp_type_to_count.keys())
        grasp_types = set()
        for inst_name in instance_and_grasp_type_to_count:
            for grsp_name in instance_and_grasp_type_to_count[inst_name]:
                if grsp_name == 'no_grasp':
                    continue
                grasp_types.add(grsp_name)
        matrix = np.zeros((len(instances), len(grasp_types)))
        for idx_o, o in enumerate(instances):
            for idx_p, p in enumerate(grasp_types):
                matrix[idx_o, idx_p] = instance_and_grasp_type_to_count[o][p]

        matrix = np.transpose(matrix)
        fig = plt.figure(figsize=figsize)
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Greys)
        plt.title('Phase: {},   tot. num. of videos: {}'
                  .format(phase, matrix.sum().astype('int32')))
        plt.colorbar()
        x_tick_marks = np.arange(len(instances))
        y_tick_marks = np.arange(len(grasp_types))
        plt.xticks(x_tick_marks, instances, rotation=90)
        plt.yticks(y_tick_marks, grasp_types)

        # in order to write text (the number in this case) inside each cell
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j].astype('int32'), 'd'),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel('Instance')
        plt.ylabel('Grasp type')

        matr_dir = os.path.join('stats', 'matrix')
        if not os.path.isdir(matr_dir):
            os.makedirs(matr_dir)
        plt.savefig(
            os.path.join(matr_dir, 'instance_grasp_type_#videos_phase_'+phase+'.png'),
            dpi=fig.dpi
        )
        if show_image:
            plt.show()
        plt.close()


def plot_matrix_instance_preshape_numvideos(args,
                                            figsize=(6, 4), 
                                            show_image=True):
    '''
    Plot a matrix instance vs. preshape where each cell has 
    the number of videos
    '''
    dataloader = load_dataset(args)

    if args.train:
        phases = ['train', 'val']
    else:
        phases = ['test']

    for phase in phases:
        instances_name_list = copy.deepcopy(args.data_info['instances'])
        preshapes_name_list = copy.deepcopy(args.data_info['preshapes'])
        instance_and_preshape_to_count = {}
        for i in instances_name_list:
            instance_and_preshape_to_count[i] = {}
            for p in preshapes_name_list:
                instance_and_preshape_to_count[i][p] = 0

        for _, _, preshape, instance, _, _ in dataloader[phase]:
            # shape (batch_size, num_frames_in_video)

            # Convert from per frame label to single video label
            # WARNING: this works only if the no_grasp class has index 0
            preshape = preshape.max(dim=1)[0]
            instance = instance.max(dim=1)[0]

            for prsh_idx, inst_idx in zip(preshape, instance):
                prsh_name = preshapes_name_list[prsh_idx.item()]
                inst_name = instances_name_list[inst_idx.item()]

                instance_and_preshape_to_count[inst_name][prsh_name] += 1

        # Remove the instances having no videos
        for inst_name in instances_name_list:
            count = 0
            for prsh_name in instance_and_preshape_to_count[inst_name]:
                count += instance_and_preshape_to_count[inst_name][prsh_name]
            if count == 0:
                del instance_and_preshape_to_count[inst_name]

        instances = list(instance_and_preshape_to_count.keys())
        preshapes = set()
        for inst_name in instance_and_preshape_to_count:
            for prsh_name in instance_and_preshape_to_count[inst_name]:
                if prsh_name == 'no_grasp':
                    continue
                preshapes.add(prsh_name)
        matrix = np.zeros((len(instances), len(preshapes)))
        for idx_o, o in enumerate(instances):
            for idx_p, p in enumerate(preshapes):
                matrix[idx_o, idx_p] = instance_and_preshape_to_count[o][p]

        matrix = np.transpose(matrix)
        fig = plt.figure(figsize=figsize)
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Greys)
        plt.title('Phase: {},   tot. num. of videos: {}'
                  .format(phase, matrix.sum().astype('int32')))
        plt.colorbar()
        x_tick_marks = np.arange(len(instances))
        y_tick_marks = np.arange(len(preshapes))
        plt.xticks(x_tick_marks, instances, rotation=90)
        plt.yticks(y_tick_marks, preshapes)

        # in order to write text (the number in this case) inside each cell
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j].astype('int32'), 'd'),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel('Instance')
        plt.ylabel('Preshape')

        matr_dir = os.path.join('stats', 'matrix')
        if not os.path.isdir(matr_dir):
            os.makedirs(matr_dir)
        plt.savefig(
            os.path.join(matr_dir, 'instance_preshape_#videos_phase_'+phase+'.png'),
            dpi=fig.dpi
        )
        if show_image:
            plt.show()
        plt.close()
