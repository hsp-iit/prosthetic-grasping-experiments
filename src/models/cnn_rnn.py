import torch
import torch.nn as nn
from torchvision import models


class CNN_RNN(nn.Module):

    def __init__(self, args):
        super(CNN_RNN, self).__init__()
        self._feature_extractor = None

        if args.from_features:
            self._feature_extractor = nn.Identity()
            # Get the feature vector dimension
            feat_vect_dim = None
            if args.feature_extractor == 'mobilenet_v2':
                appo = models.mobilenet_v2(pretrained=False)
                feat_vect_dim = appo.classifier[-1].in_features
            else:
                raise ValueError('Not yet implemented for '
                                 '--feature_extractor {}'
                                 .format(args._feature_extractor))
        else:
            imagenet_pretrained = True if args.pretrain == 'imagenet' else False
            if args.feature_extractor == 'mobilenet_v2':
                self._feature_extractor = models.mobilenet_v2(
                    pretrained=imagenet_pretrained
                )
            else:
                raise ValueError('Not yet implemented for '
                                 '--feature_extractor {}'
                                 .format(args._feature_extractor))

            if args.pretrain == 'imagenet':
                # nothing to do, weights already loaded above
                pass
            else:
                raise ValueError('Not yet implemented for --pretrain {}'
                                 .format(args.pretrain))

            if args.freeze_all_conv_layers:
                for param in self._feature_extractor.parameters():
                    param.requires_grad = False

            feat_vect_dim = self._feature_extractor.classifier[-1].in_features
            self._feature_extractor.classifier = nn.Identity()

        self._rnn_hidden_size = args.rnn_hidden_size
        self._rnn_type = None
        if args.rnn_type == 'lstm':
            # We could use also torch.nn.LSTM, I prefer this one since it has
            # more flexibility. In case of changes,
            # also the forward must be changed accordingly
            self._rnn_type = nn.LSTMCell(feat_vect_dim, args.rnn_hidden_size)
        else:
            raise ValueError('Not yet implemented for --rnn_type {}'
                             .format(args.rnn_type))

        self._dropout = nn.Dropout(args.dropout)

        self._classifier = nn.Linear(args.rnn_hidden_size, args.num_classes)

    def forward(self, x):
        # x.shape (batch_size, num_frames_in_video, C, H, W)
        batch_size, num_frames = x.shape[0], x.shape[1]
        h_n = torch.zeros(batch_size, self._rnn_hidden_size).to(
            device=x.device, dtype=x.dtype
        )
        c_n = torch.zeros_like(h_n).to(device=x.device, dtype=x.dtype)

        scores = []
        for n in range(num_frames):
            out = self._feature_extractor(x[:, n])
            h_n, c_n = self._rnn_type(self._dropout(out), (h_n, c_n))

            appo = self._classifier(self._dropout(h_n))
            # appo.shape (batch_size, num_classes)
            scores.append(appo)

        scores = torch.stack(scores, dim=1)
        # scores.shape (batch_size, num_frames, num_classes)

        return scores

    def step(self, x_t, states):
        # Used when instead of passing a whole sequence, only one step of that
        # sequence has to be processed

        # x.shape (batch_size, C, H, W)
        batch_size = x_t.shape[0]
        if states is None:
            h_t = torch.zeros(batch_size, self._rnn_hidden_size).to(
                device=x_t.device, dtype=x_t.dtype
            )
            c_t = torch.zeros_like(h_t).to(device=x_t.device, dtype=x_t.dtype)
        else:
            h_t, c_t = states

        out = self._feature_extractor(x_t)
        h_t, c_t = self._rnn_type(out, (h_t, c_t))
        score = self._classifier(h_t)

        return score, (h_t, c_t)
