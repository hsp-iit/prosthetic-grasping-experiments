import glob
import os

import numpy as np
import cv2
import torch
from PIL import Image


def show_video_predictions(scores,
                           targets,
                           video_path,
                           path_to_save_dir,
                           classes_list):
    # scores.shape (num_frames_in_video, num_classes)
    # targets.shape (num_frames_in_video)
    print('\n\nLoading video and printing predictions . . .')
    image_paths = glob.glob(os.path.join(video_path, '*.jpg'))
    image_paths.sort()
    TOP_K = 3
    values, indices = torch.topk(scores, TOP_K, dim=1)
    for idx, im_p in enumerate(image_paths):
        open_cv_frame = cv2.imread(im_p)

        open_cv_frame = cv2.copyMakeBorder(open_cv_frame, 90, 0, 30, 30, borderType=cv2.BORDER_CONSTANT, value=0)

        target_label = classes_list[targets[idx].item()]
        cv2.putText(open_cv_frame,
                    target_label,
                    (400, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    1)

        for k in range(TOP_K):
            k_score = values[idx, k]
            k_pred_label = classes_list[indices[idx, k].item()]

            if k == 0:
                color = (0, 255, 0) if k_pred_label==target_label else (0, 0, 255)
            else:
                color = (255, 255, 255)
            cv2.putText(open_cv_frame,
                        k_pred_label,
                        (0, 20+(30*k)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        1)
            cv2.putText(open_cv_frame,
                        '{:<2.1f}%'.format(k_score.item() * 100),
                        (210, 20+(30*k)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1)

        frame_path = os.path.join(path_to_save_dir, str(idx)+'.jpg')
        cv2.imwrite(frame_path, open_cv_frame)

    print('. . . video saved at ' + os.path.join(path_to_save_dir))