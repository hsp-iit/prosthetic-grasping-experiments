import itertools
import io
from collections import OrderedDict

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import cv2


def accuracy(scores, targets, topk=(1, 3), divide_by_batch_size=False):
    # code from: https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        # scores.shape==(batch_size, num_classes)
        _, y_pred = scores.topk(k=maxk, dim=1)  # y_pred.shape==(batch_size, maxk)
        y_pred = y_pred.t()

        # targets.shape==(batch_size,)
        targets = targets.unsqueeze(0).expand_as(y_pred) # replicate ground truth label until maxk

        correct = (y_pred == targets)

        # get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.sum(dim=0, keepdim=True)
            topk_acc = (tot_correct_topk / batch_size) if divide_by_batch_size else tot_correct_topk
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


def per_class_accuracy(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    per_class_accuracies = cm.diagonal()

    return per_class_accuracies


def per_instance_preshape_accuracy(y_true, y_pred, instances, data_info, 
                                   figsize=(16, 10), show_image=True, 
                                   writer=None, 
                                   title='Per-instance-preshape accuracy'):
    preshapes_names = data_info['preshapes']
    instances_names = data_info['instances']

    ground_truth_instance_preshape = np.zeros(
        (len(data_info['preshapes']), len(data_info['instances']))
    )
    for i, idx_preshape in enumerate(y_true):
        idx_instance = instances[i]
        ground_truth_instance_preshape[idx_preshape][idx_instance] += 1

    correct_instance_preshape = np.zeros(
        (len(data_info['preshapes']), len(data_info['instances']))
    )
    for i, idx_preshape in enumerate(y_pred):
        if y_pred[i] != y_true[i]:
            continue
        idx_instance = instances[i]
        correct_instance_preshape[idx_preshape][idx_instance] += 1

    accuracy_instance_preshape = correct_instance_preshape / ground_truth_instance_preshape

    figure = plt.figure(figsize=figsize)
    plt.imshow(accuracy_instance_preshape, interpolation='nearest', 
               cmap=plt.cm.Greys)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(instances_names))
    y_tick_marks = np.arange(len(preshapes_names))
    plt.xticks(x_tick_marks, instances_names, rotation=90)
    plt.yticks(y_tick_marks, preshapes_names)

    # in order to write text (the number in this case) inside each cell
    thresh = np.nanmax(accuracy_instance_preshape) / 2.
    for i, j in itertools.product(range(accuracy_instance_preshape.shape[0]), range(accuracy_instance_preshape.shape[1])):
        plt.text(j, i,
                 '{:.1f}%'.format(accuracy_instance_preshape[i, j] * 100) if not np.isnan(accuracy_instance_preshape[i, j]) else 'nan',
                 horizontalalignment="center",
                 color="white" if accuracy_instance_preshape[i, j] > thresh or np.isnan(accuracy_instance_preshape[i, j]) else "black",
                 fontsize='x-small')

    plt.tight_layout()
    plt.xlabel('Instance')
    plt.ylabel('Preshape')
    plt.grid(False)
    if show_image:
        plt.show()
    plt.close()
    if writer is None:
        raise ValueError('Unable to save confusion matrix to tensorboard: '
                         'you have to pass the SummaryWriter object (via '
                         'the writer parameter) responsible for writing to'
                         ' tensorboard.')
    writer.add_figure('matrix/'+title.replace(' ', '_'), figure)
    writer.close()


def per_idx_frame_accuracy(y_true, y_pred, writer, output, num_videos, num_frames_per_video):
    '''
    On the horizontal axis there is the idx of the frame, i.e. temporal evolution of the video,
    and on the vertical axis there is the per-index-frame accuracy, e.g.
    for i to NUM_FRAMES_PER_VIDEO
        accuracy = correctly_predicted_frames_w/_index_i  /  tot_num_of_frames_w/_index_i
    '''
    assert y_true.shape==y_pred.shape==(num_videos, num_frames_per_video)

    for idx_frame in range(y_true.shape[1]):
        per_idx_frame_acc = y_true[:, idx_frame]==y_pred[:, idx_frame]
        per_idx_frame_acc = per_idx_frame_acc.sum() / len(per_idx_frame_acc) * 100
        writer.add_scalar('Per_idx_frame_accuracy_'+output+'/test', per_idx_frame_acc, idx_frame)
    writer.close()


def per_instance_grasp_type_accuracy(y_true, y_pred, instances, data_info, 
                                     figsize=(10, 10), show_image=True,
                                     writer=None, 
                                     title='Per-instance-grasp_type accuracy'):
    grasp_types_names = data_info['grasp_types']
    instances_names = data_info['instances']

    ground_truth_instance_grasp_type = np.zeros(
        (len(data_info['grasp_types']), len(data_info['instances']))
    )
    for i, idx_grasp_type in enumerate(y_true):
        idx_instance = instances[i]
        ground_truth_instance_grasp_type[idx_grasp_type][idx_instance] += 1

    correct_instance_grasp_type = np.zeros(
        (len(data_info['grasp_types']), len(data_info['instances']))
    )
    for i, idx_grasp_type in enumerate(y_pred):
        if y_pred[i] != y_true[i]:
            continue
        idx_instance = instances[i]
        correct_instance_grasp_type[idx_grasp_type][idx_instance] += 1

    accuracy_instance_grasp_type = correct_instance_grasp_type / ground_truth_instance_grasp_type

    figure = plt.figure(figsize=figsize)
    plt.imshow(accuracy_instance_grasp_type, interpolation='nearest', cmap=plt.cm.Greys)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(instances_names))
    y_tick_marks = np.arange(len(grasp_types_names))
    plt.xticks(x_tick_marks, instances_names, rotation=90)
    plt.yticks(y_tick_marks, grasp_types_names)

    # in order to write text (the number in this case) inside each cell
    thresh = np.nanmax(accuracy_instance_grasp_type) / 2.
    for i, j in itertools.product(range(accuracy_instance_grasp_type.shape[0]), range(accuracy_instance_grasp_type.shape[1])):
        plt.text(j, i,
                 '{:.1f}%'.format(accuracy_instance_grasp_type[i, j] * 100) if not np.isnan(accuracy_instance_grasp_type[i, j]) else 'nan',
                 horizontalalignment="center",
                 color="white" if accuracy_instance_grasp_type[i, j] > thresh or np.isnan(accuracy_instance_grasp_type[i, j]) else "black",
                 fontsize='x-small')

    plt.xlabel('Instance')
    plt.ylabel('Grasp type')
    plt.grid(False)
    plt.tight_layout(rect=[0, 0.4, 1, 0.95])
    if show_image:
        plt.show()
    plt.close()
    if writer is None:
        raise ValueError('Unable to save confusion matrix to tensorboard: '
                         'you have to pass the SummaryWriter object (via '
                         'the writer parameter) responsible for writing to '
                         'tensorboard.')
    writer.add_figure('matrix/'+title.replace(' ', '_'), figure)
    writer.close()


def _plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to numpy array
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    image = cv2.imdecode(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _save_fig_to_tensorboard(fig, writer, title):
    fig = _plot_to_image(fig)
    writer.add_image(title, np.transpose(fig, (2, 0, 1)), 0)
    writer.close()


def plot_confusion_matrix(y_true,
                          y_pred,
                          classes,
                          normalize=True,
                          figsize=(5, 5),
                          title='confusion matrix',
                          cmap=plt.cm.Greys,
                          show_image=True,
                          save_fig_to_tensorboard=True,
                          writer=None):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    figure = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    appo = 'Normalized ' if normalize else 'Unnormalized '
    plt.title(appo+title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = np.nanmax(cm) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.tight_layout(rect=[0, 0.4, 1, 0.95])
    if show_image:
        plt.show()
    plt.close()
    if save_fig_to_tensorboard:
        writer.add_figure('matrix/'+(appo+title).replace(' ', '_'), figure)
        writer.close()


def per_idx_frame_preds_and_accuracy(scores, y_pred, cls_list, figsize=(6, 4), title=None, path_to_save_fig=None):
    '''
    This function is used in src/tools/*/*/video/eval_single_video.py
    It shows a plot where on x-axis there is the frame index and there are two y-axes: one to show the predicted
    class and the other to show the score of the predicted class.
    NOTE THAT this is plot contains stats about a single video, for similar stats but regarding all of the videos
    of the dataset see per_idx_frame_accuracy
    :param scores: shape (num_frames, num_classes)
    :param y_pred: shape (num_frames)
    '''
    scores = scores.max(dim=1)[0]     # take the score of the predicted class
    y_pred_cls_names = []
    for idx_cls in y_pred:
        y_pred_cls_names.append(cls_list[idx_cls.item()])

    f, axes = plt.subplots(2, figsize=figsize)
    axes[0].scatter(np.arange(len(y_pred)), y_pred_cls_names)
    axes[0].set_ylabel('Predicted class')
    axes[1].plot(np.arange(len(y_pred)), scores)
    axes[1].set_ylabel('Confidence')

    f.suptitle(title)
    plt.savefig(path_to_save_fig)
    plt.show()
    plt.close()


def perframe_mAP(scores, targets, list_classes, smooth=False):
    ###################################################################################################################
    # Function taken from https://github.com/xumingze0308/TRN.pytorch/blob/master/lib/utils/eval_utils.py
    ###################################################################################################################
    ###################################################################################################################
    # We follow (Shou et al., 2017) and adopt their per-frame evaluation method of THUMOS'14 datset.
    # Source: https://bitbucket.org/columbiadvmm/cdc/src/master/THUMOS14/eval/PreFrameLabeling/compute_framelevel_mAP.m
    ###################################################################################################################
    '''
    scores.shape==(num_samples, num_classes),   targets.shape==(num_samples,)
    where num_samples==(num_frames_in_video * num_videos)
    '''

    # Simple temporal smoothing via NMS of 5-frames window
    if smooth:
        prob = np.copy(scores)
        prob1 = prob.reshape(1, prob.shape[0], prob.shape[1])
        prob2 = np.append(prob[0, :].reshape(1, -1), prob[0:-1, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob3 = np.append(prob[1:, :], prob[-1, :].reshape(1, -1), axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob4 = np.append(prob[0:2, :], prob[0:-2, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        prob5 = np.append(prob[2:, :], prob[-2:, :], axis=0).reshape(1, prob.shape[0], prob.shape[1])
        probsmooth = np.squeeze(np.max(np.concatenate((prob1, prob2, prob3, prob4, prob5), axis=0), axis=0))
        scores = np.copy(probsmooth)

    # Compute AP
    result = {}
    result['AP'] = OrderedDict()
    for idx_cls, cls in enumerate(list_classes):
        appo = average_precision_score((targets==idx_cls).astype(np.int), scores[:, idx_cls])
        if np.isnan(appo):
            # the average precision score for the actual class is not considered since there
            #  are no ground truth in this class
            continue
        result['AP'][cls] = appo

    # Compute mAP considering also background class,
    result['mAP_all_cls'] = np.mean(list(result['AP'].values()))
    # Compute mAP without considering background class, i.e. 'no_grasp' class
    valid_APs = [result['AP'][cls] for cls in result['AP'].keys() if cls != 'no_grasp']
    result['mAP_valid_cls'] = np.mean(valid_APs)

    return result
