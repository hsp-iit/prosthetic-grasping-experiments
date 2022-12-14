import os
import glob
import random
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from src.configs import conf


class FeaturesSingleSourceVideoDataset(Dataset):

    def __init__(self, args):
        self._features_suffix = args.feature_extractor
        self._source = args.source
        self._data_info = args.data_info
        self._input = args.input

        self._split = args.split
        self._train_test_split_seed = args.train_test_split_seed
        self._train_test_split_size = args.train_test_split_size
        self._val_test_split_size = args.val_test_split_size
        self._is_dataset_split = False
        self._test_type = args.test_type

        self._instance_and_preshape_2_videos_path = None

        # Currently, all the video of the iHannesDataset are 90 frames long,
        # except for the [test_seated, test_different_background_1,
        # test_different_background_2] test sets which are 45 frames long.
        if args.test_type in ['test_seated', 'test_different_background_1', 'test_different_background_2']:
            self._NUM_FRAMES_IN_VIDEO = 45
        else:
            self._NUM_FRAMES_IN_VIDEO = 90

        # sample path: iHannesDataset/category/instance/preshape/source/rgb*/features.npy

        if args.split == 'random':
            # In --split==random policy, for each preshape of each instance,
            # we put the --train_test_split_size percentage of the whole
            # dataset in the training set and the remaining part in another set.
            # Then, considering this other set, the --val_test_split_size
            # percentage will be in the validation set and the remaining in the
            # test set.

            if args.input == 'rgb':
                videos_iterator = glob.iglob(
                    os.path.join(args.dataset_base_folder,
                                 '*',
                                 '*',
                                 '*',
                                 self._source,
                                 args.input.lower()+'*')
                )

                self._instance_and_preshape_2_videos_path = {}
                for inst in self._data_info['instances']:
                    self._instance_and_preshape_2_videos_path[inst] = {}
                    for prsh in self._data_info['preshapes']:
                        # Each element is a tuple containing the list of paths
                        # and the length of the list
                        self._instance_and_preshape_2_videos_path[inst][prsh] = (0, [])

                for video_path in videos_iterator:
                    folders = pathlib.Path(video_path).parts
                    preshape = folders[-3]
                    instance = folders[-4]
                    category = folders[-5]
                    if (preshape not in self._data_info['preshapes'] or
                            instance not in self._data_info['instances'] or
                            category not in self._data_info['categories']):
                        # print('Skipping {} since preshape or instance or '
                        #       'category is missing in {}'
                        #       .format(video_path, args.data_info_file_path))
                        continue

                    # Depending on the mode (i.e. train or test), a different
                    # set has to be considered
                    metadata_path = video_path\
                        .replace(self._source, 'metadata')\
                        .replace('rgb', 'seq')
                    metadata_path = os.path.join(metadata_path, 'data.log')
                    with open(metadata_path, 'r') as metadata_file:
                        metadata_row = metadata_file.readline().split(' ')
                        testtype_or_other = metadata_row[-1].strip()

                        if args.test_type in [None, 'test_same_person']:
                            # We are in the training phase or testing on the
                            # same_person set
                            if testtype_or_other in conf.TEST_TYPE:
                                continue
                        else:
                            # We are testing on a set different from the
                            # same_person test set
                            if testtype_or_other != args.test_type:
                                continue

                    cur_list_len, list_videos_path = self._instance_and_preshape_2_videos_path[instance][preshape]
                    cur_list_len += 1
                    # Fastest way to add an element to a list
                    list_videos_path += video_path,
                    self._instance_and_preshape_2_videos_path[instance][preshape] = (cur_list_len, list_videos_path)

            else:
                raise ValueError('Not yet implemented for --input {}'
                                 .format(args.input))

            # Cleaning: remove all the instance-preshape pairs
            # having no videos
            for inst in self._data_info['instances']:
                for prsh in self._data_info['preshapes']:
                    cur_list_len = self._instance_and_preshape_2_videos_path[inst][prsh][0]
                    if cur_list_len == 0:
                        del self._instance_and_preshape_2_videos_path[inst][prsh]
            for inst in self._data_info['instances']:
                if len(self._instance_and_preshape_2_videos_path[inst]) == 0:
                    del self._instance_and_preshape_2_videos_path[inst]

            # The instance_and_preshape_2_videos_path contains all the videos
            # path of the considered dataset. If test_type is None or
            # test_type=='test_same_person', these videos will be subsequently
            # split into training, validation and tests sets according to the
            # policy defined in --split. To this purpose,
            # the self.split_in_train_val_test_sets function must be called.

        else:
            raise ValueError('Not yet implemented for --split {}'
                             .format(args.split))

    def split_in_train_val_test_sets(self):
        if self._test_type not in [None, 'test_same_person']:
            raise RuntimeError('This function is meant to be called only '
                               'when we are in training phase or testing on '
                               'the same_person test set')

        if self._split == 'random':
            train_idxs = []
            val_idxs = []
            test_idxs = []

            count_elems = 0
            for inst in self._instance_and_preshape_2_videos_path:
                for prsh in self._instance_and_preshape_2_videos_path[inst]:
                    cur_list_len = \
                        self._instance_and_preshape_2_videos_path[inst][prsh][0]

                    if cur_list_len < 3:
                        raise RuntimeError('We need to have at least 3 videos '
                                           'for each preshape of each instance'
                                           ', such that we have a video in '
                                           'each set (i.e., training, '
                                           'validation and test)')

                    # Later we will do a shuffle, therefore we must be sure
                    # that the order of the elements in the list is
                    # always the same to ensure a reproducible shuffle.
                    # Hence, we preliminarily order the list.
                    # However, maybe the elements order (when feeding the
                    # list as input to shuffle) is always the same, hence
                    # .sort() might be useless... [TO DOUBLE-CHECK] TODO
                    cur_seqs_path = \
                        self._instance_and_preshape_2_videos_path[inst][prsh][1]
                    cur_seqs_path.sort()
                    random.Random(self._train_test_split_seed) \
                        .shuffle(cur_seqs_path)

                    if cur_list_len == 3:
                        train_idxs += count_elems,
                        count_elems += 1

                        val_idxs += count_elems,
                        count_elems += 1

                        test_idxs += count_elems,
                        count_elems += 1
                    else:
                        cur_train_elems_len = int(
                            cur_list_len * self._train_test_split_size
                        )
                        cur_train_elems_idxs = \
                            np.arange(cur_train_elems_len) + count_elems
                        train_idxs += cur_train_elems_idxs.tolist()
                        count_elems += cur_train_elems_len

                        cur_test_elems_len = cur_list_len - cur_train_elems_len

                        cur_val_elems_len = int(
                            cur_test_elems_len * self._val_test_split_size
                        )
                        cur_val_elems_idxs = \
                            np.arange(cur_val_elems_len) + count_elems
                        val_idxs += cur_val_elems_idxs.tolist()
                        count_elems += cur_val_elems_len

                        cur_test_elems_len -= cur_val_elems_len

                        cur_test_elems_idxs = \
                            np.arange(cur_test_elems_len) + count_elems
                        test_idxs += cur_test_elems_idxs.tolist()
                        count_elems += cur_test_elems_len

            datasets = {}
            if self._test_type is None:
                datasets['train'] = Subset(self, train_idxs)
                datasets['val'] = Subset(self, val_idxs)
            elif self._test_type == 'test_same_person':
                datasets['test'] = Subset(self, test_idxs)
            else:
                raise ValueError('Not yet implemented for --test_type {}'
                                 .format(self._split))

            self._is_dataset_split = True

            return datasets
        else:
            raise ValueError('Not yet implemented for --split {}'
                             .format(self._split))

    def train(self, is_train=True):
        # Function not needed when training on pre-extracted features.
        # We declared this function just for sake of coherence
        # with datasets.SingleSourceImageDataset class
        pass

    def eval(self):
        # Function not needed when training on pre-extracted features.
        # We declared this function just for sake of coherence
        # with datasets.SingleSourceImageDataset class
        pass

    def __getitem__(self, item):
        if self._test_type in [None, 'test_same_person']:
            if not self._is_dataset_split:
                raise RuntimeError('Dataset not split: you have to call '
                                   'split_in_train_val_test_sets() function '
                                   'and use the returned Subsets.')
        else:
            if self._is_dataset_split:
                raise RuntimeError('We are considering --test_type=={},'
                                   'therefore the dataset class does not have '
                                   'to be split: remove '
                                   'split_in_train_val_test_sets() function '
                                   'call'
                                   .format(self._test_type))

        # Retrieve video path
        count_elems = 0
        flag = False
        for inst in self._instance_and_preshape_2_videos_path:
            if flag:
                break
            for prsh in self._instance_and_preshape_2_videos_path[inst]:
                cur_list_len = \
                    self._instance_and_preshape_2_videos_path[inst][prsh][0]

                if item < count_elems + cur_list_len:
                    item -= count_elems
                    video_path = \
                        self._instance_and_preshape_2_videos_path[inst][prsh][1][item]
                    flag = True
                    break

                count_elems += cur_list_len

        if not flag:
            raise RuntimeError('Something went wrong: no video path found '
                               'at the index {}'.format(item))

        # Load features
        if self._input == 'rgb':
            new_folders = os.path.join('features', self._features_suffix)
            old_path = pathlib.Path(video_path)
            index_to_replace = old_path.parts.index('frames')
            video_path_features = os.path.join(*old_path.parts[:index_to_replace],
                                               new_folders,
                                               *old_path.parts[index_to_replace+1:])

            video_features = np.load(
                os.path.join(video_path_features, 'features.npy')
            )
            video_features = torch.tensor(video_features).to(dtype=torch.float32)
            # video_features.shape (num_frames_in_video, feat_vect_dim)
            num_frames = video_features.shape[0]
            if num_frames != self._NUM_FRAMES_IN_VIDEO:
                raise RuntimeError('This dataset class is expected to work '
                                   'with video {} frames long, but the video '
                                   '{} has {} frames'
                                   .format(self._NUM_FRAMES_IN_VIDEO,
                                           video_path,
                                           num_frames))
        else:
            raise ValueError('Not yet implemented for --input {}'
                             .format(self._input))

        # Load metadata
        metadata = {}
        appo = video_path.replace(self._source, 'metadata').replace('rgb', 'seq')
        with open(os.path.join(appo, 'data.log')) as metadata_file:
            metadata_row = metadata_file.readline()
            # sample row:
            #  frame_id timestamp_in timestamp_out category instance grasp_type preshape elevation approach ojbAzimuth objElevation
            metadata_row = metadata_row.split(' ')
            metadata['category'] = metadata_row[3]
            metadata['instance'] = metadata_row[4]
            metadata['grasp_type'] = metadata_row[5]
            metadata['preshape'] = metadata_row[6]
            if metadata_row[-1].strip() not in conf.TEST_TYPE:
                metadata['elevation'] = int(metadata_row[7].split('_')[1])
                metadata['approach'] = int(metadata_row[8].split('_')[1])
                metadata['objAzimuth'] = int(metadata_row[9].split('_')[1])
                metadata['objElevation'] = int(
                    metadata_row[10].replace('\n', '').split('_')[1]
                )

        # WARNING: Currently, training videos can be only 90 frames long,
        # at 30 fps, hence 3 seconds video.
        # Therefore the subsequent labeling is valid only for that video
        # frames length.
        # LABEL: 2.5s [grasp]  -  0.5s [no_grasp]
        #
        # BUT, in some test sets the videos are 45 frames long, in this case
        # the whole sequence is labeled with the grasp class
        if self._NUM_FRAMES_IN_VIDEO not in [90, 45]:
            raise RuntimeError('The labeling scheme below works only with '
                               'videos which are 90 or 45 frames long')
        idx_grasp_type = self._data_info['grasp_types'].index(metadata['grasp_type'])
        idx_preshape = self._data_info['preshapes'].index(metadata['preshape'])
        idx_instance = self._data_info['instances'].index(metadata['instance'])
        idx_no_grasp = 0

        grasp_type = torch.tensor(idx_grasp_type).repeat(num_frames)
        preshape = torch.tensor(idx_preshape).repeat(num_frames)
        instance = torch.tensor(idx_instance).repeat(num_frames)

        if self._NUM_FRAMES_IN_VIDEO == 90:
            grasp_type[75:] = idx_no_grasp
            preshape[75:] = idx_no_grasp
            instance[75:] = idx_no_grasp
        elif self._NUM_FRAMES_IN_VIDEO == 45:
            pass
        else:
            raise NotImplementedError

        return video_features, grasp_type, preshape, instance, metadata, video_path

    def __len__(self):
        count_elems = 0
        for inst in self._instance_and_preshape_2_videos_path:
            for prsh in self._instance_and_preshape_2_videos_path[inst]:
                count_elems += self._instance_and_preshape_2_videos_path[inst][prsh][0]

        return count_elems
