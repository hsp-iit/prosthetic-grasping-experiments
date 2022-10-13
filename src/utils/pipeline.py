import random
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from src.datasets.SingleSourceVideoDataset import SingleSourceVideoDataset
from src.datasets.SingleSourceImageDataset import SingleSourceImageDataset
from src.datasets.features.SingleSourceImageDataset import \
    FeaturesSingleSourceImageDataset
from src.datasets.features.SingleSourceVideoDataset import \
    FeaturesSingleSourceVideoDataset
from src.models.cnn_rnn import CNN_RNN
from src.models.cnn import CNN


class MyRandomCrop(transforms.RandomCrop):
    
    '''
    This class is needed to perform the same random crop along all the frames of a video
    '''

    def __init__(self, size, padding=0, pad_if_needed=False):
        super(MyRandomCrop, self).__init__(size, padding, pad_if_needed)
        self.crop_indices = []

    def __call__(self, img, resample):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill,
                        self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill,
                        self.padding_mode)

        if resample:
            # get novel cropping coordinates
            self.crop_indices = self.get_params(img, self.size)
        i, j, h, w = self.crop_indices

        return F.crop(img, i, j, h, w)


class MyCompose(transforms.Compose):

    def __init__(self, transforms):
        super(MyCompose, self).__init__(transforms)

    def __call__(self, img, resample):
        for t in self.transforms:
            if isinstance(t, MyRandomCrop):
                img = t(img, resample)
            else:
                img = t(img)
        return img


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_dataset(args):
    if args.from_features:
        if args.dataset_type == 'SingleSourceVideo':
            whole_dataset = FeaturesSingleSourceVideoDataset(args)
        elif args.dataset_type == 'SingleSourceImage':
            whole_dataset = FeaturesSingleSourceImageDataset(args)
        else:
            raise ValueError('Not yet implemented for --dataset_type {}'
                             .format(args.dataset_type))

    else:
        if args.pretrain == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            resize_side_size = 256
            crop_size = 224
        else:
            raise ValueError('Not yet implemented for --pretrain {}'
                             .format(args.pretrain))

        transform = {}
        if args.dataset_type == 'SingleSourceVideo':
            transform['train'] = MyCompose([
                transforms.Resize(resize_side_size),
                MyRandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            transform['test'] = transforms.Compose([
                transforms.Resize(resize_side_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            whole_dataset = SingleSourceVideoDataset(args, transform)

        elif args.dataset_type == 'SingleSourceImage':
            transform['train'] = transforms.Compose([
                transforms.Resize(resize_side_size),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            transform['test'] = transforms.Compose([
                transforms.Resize(resize_side_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            whole_dataset = SingleSourceImageDataset(args, transform)

        else:
            raise ValueError('Not yet implemented for --dataset_type {}'
                             .format(args.dataset_type))

    dataloaders = {}
    if args.test_type is None:
        datasets = whole_dataset.split_in_train_val_test_sets()

        dataloaders['train'] = DataLoader(datasets['train'], args.batch_size,
                                          True, num_workers=args.num_workers,
                                          drop_last=False)
        dataloaders['val'] = DataLoader(datasets['val'], args.batch_size,
                                        False, num_workers=args.num_workers,
                                        drop_last=False)
    elif args.test_type == 'test_same_person':
        datasets = whole_dataset.split_in_train_val_test_sets()

        dataloaders['test'] = DataLoader(datasets['test'], args.batch_size,
                                         False, num_workers=args.num_workers,
                                         drop_last=False)
    else:
        dataloaders['test'] = DataLoader(whole_dataset, args.batch_size,
                                         False, num_workers=args.num_workers,
                                         drop_last=False)

    return dataloaders


def load_model(args):
    if args.model == 'cnn_rnn':
        model = CNN_RNN(args)
    elif args.model == 'cnn':
        model = CNN(args)
    else:
        raise ValueError('Not yet implemented for --model {}'
                         .format(args.model))

    return model


def check_arguments_matching(path_to_argument_file, sys_argv):
    # Check if the arguments are the same at training and test time.
    # From the first and second parameters we can retrieve respectively
    # training and test arguments.
    # If some argument regarding the model specs is wrongly
    # changed when lauching the evaluation pipeline, an error must be raised.

    ARGS_TO_IGNORE = ['--test_type', '--checkpoint', '--show_perframe_preds']

    # 1: construct back the training arguments
    with open(os.path.join(path_to_argument_file, 'log.txt'), 'r') as f:
        while f:
            line = f.readline()
            if 'python' not in line:
                continue
            launch_command = line.split(' ')
            training_arguments = launch_command[2:]
            training_arg_2_value = {}
            for i, val in enumerate(training_arguments):
                appo = val.strip() if type(val) is str else val
                # WARNING: THE CODE BELOW WORKS ONLY IF BOOLEAN ARGUMENTS ARE
                #          DEFINED AS: default=False, action='store_true'
                if '--' in val:
                    training_arg_2_value[appo] = True
                else:
                    # overwrite with the correct value
                    training_arg_2_value[training_arguments[i-1]] = appo
            break

    # 2: get current argument value pairs
    test_arg_2_value = {}
    test_arguments = sys_argv[1:]
    for i, val in enumerate(test_arguments):
        appo = val.strip() if type(val) is str else val
        # WARNING: THE CODE BELOW WORKS ONLY IF BOOLEAN ARGUMENTS ARE
        #          DEFINED AS: default=False, action='store_true'
        if '--' in val:
            test_arg_2_value[appo] = True
        else:
            # overwrite with the correct value
            test_arg_2_value[test_arguments[i-1]] = appo

    # 3: compare them with the arguments launched in this evaluation pipeline
    for arg, value in training_arg_2_value.items():
        if arg in ARGS_TO_IGNORE:
            continue
        if test_arg_2_value[arg] != value:
            print(test_arg_2_value[arg], len(test_arg_2_value[arg]))
            print(value, len(value))
            raise Exception('Unmatch argument value between training'
                            ' and test phases: argument '+arg+' has '
                            'value '+value.strip()+' at training time but '
                            +test_arg_2_value[arg].strip()+' at test time.')


def temporal_augmentation(frames, grasp_types, preshapes, instances):
    NUM_FRAMES = frames.shape[1]
    assert frames.shape[1] == grasp_types.shape[1] == preshapes.shape[1] == \
        instances.shape[1] == NUM_FRAMES, \
        'size mismatch in the number of frames between data and labels'
    # frames.shape (batch_size, num_frames, 3, 224, 224)
    # grasp_types.shape==preshapes.shape==instances.shape==(batch_size, num_frames)
    frames_shape, grasp_types_shape, preshapes_shape, instances_shape = \
        frames.shape, grasp_types.shape, preshapes.shape, instances.shape

    new_frames = []
    new_grasp_types = []
    new_preshapes = []
    new_instances = []
    for batch_idx in range(frames.shape[0]):
        do_trimming = random.uniform(0, 1) > 0.5
        do_sampling = random.uniform(0, 1) > 0.5
        cur_vid_frames = frames[batch_idx].detach().clone()
        cur_vid_grasp_types = grasp_types[batch_idx].detach().clone()
        cur_vid_preshapes = preshapes[batch_idx].detach().clone()
        cur_vid_instances = instances[batch_idx].detach().clone()

        if do_trimming:
            idx = random.randint(0, cur_vid_frames.shape[0] // 2)
            cur_vid_frames = cur_vid_frames[idx:]
            cur_vid_grasp_types = cur_vid_grasp_types[idx:]
            cur_vid_preshapes = cur_vid_preshapes[idx:]
            cur_vid_instances = cur_vid_instances[idx:]
        if do_sampling:
            sample_rate = random.randint(2, 6)
            cur_vid_frames = cur_vid_frames[::sample_rate]
            cur_vid_grasp_types = cur_vid_grasp_types[::sample_rate]
            cur_vid_preshapes = cur_vid_preshapes[::sample_rate]
            cur_vid_instances = cur_vid_instances[::sample_rate]

        if cur_vid_frames.shape[0] < NUM_FRAMES:
            # i.e. enter here if the current video has been previously performed trimming or sampling or both:
            #  in this case we need to pad the video until it has NUM_FRAMES frames, the video is padded by
            #  looping it until NUM_FRAMES
            padded_frames = []
            padded_grasp_types = []
            padded_preshapes = []
            padded_instances = []
            for _ in range(0, NUM_FRAMES + cur_vid_frames.shape[0],
                           cur_vid_frames.shape[0]):
                padded_frames.append(cur_vid_frames.detach().clone())
                padded_grasp_types.append(cur_vid_grasp_types.detach().clone())
                padded_preshapes.append(cur_vid_preshapes.detach().clone())
                padded_instances.append(cur_vid_instances.detach().clone())
            padded_frames = torch.cat(padded_frames, dim=0)
            padded_grasp_types = torch.cat(padded_grasp_types, dim=0)
            padded_preshapes = torch.cat(padded_preshapes, dim=0)
            padded_instances = torch.cat(padded_instances, dim=0)
            padded_frames = padded_frames[:NUM_FRAMES]
            padded_grasp_types = padded_grasp_types[:NUM_FRAMES]
            padded_preshapes = padded_preshapes[:NUM_FRAMES]
            padded_instances = padded_instances[:NUM_FRAMES]
        else:
            padded_frames = cur_vid_frames
            padded_grasp_types = cur_vid_grasp_types
            padded_preshapes = cur_vid_preshapes
            padded_instances = cur_vid_instances

        new_frames.append(padded_frames)
        new_grasp_types.append(padded_grasp_types)
        new_preshapes.append(padded_preshapes)
        new_instances.append(padded_instances)

    new_frames = torch.stack(new_frames)
    new_grasp_types = torch.stack(new_grasp_types)
    new_preshapes = torch.stack(new_preshapes)
    new_instances = torch.stack(new_instances)

    assert new_frames.shape == frames_shape and \
        new_grasp_types.shape == grasp_types_shape and \
        new_preshapes.shape == preshapes_shape and \
        new_instances.shape == instances_shape, \
        'Error during the temporal data augmentation process: ' \
        'the shape must remain the same'
    return new_frames, new_grasp_types, new_preshapes, new_instances


"""
The class below is taken from
https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py 
"""
class EarlyStopping:
    
    """Early stops the training if validation loss doesn't improve after a
    given patience."""

    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss
                            improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss
                            improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify
                           as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta

    def __call__(self, val_loss):
        early_stop = False
        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return early_stop
