import torch.nn as nn
from torchvision import models


class CNN(nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
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

        in_features = feat_vect_dim
        self._fc_layers = []
        if args.fc_layers is not None:
            fc_neurons = args.fc_neurons.split(',')
            for i in range(args.fc_layers):
                # Dropout -> Linear -> ReLU
                self._fc_layers.append(nn.Dropout(args.dropout))
                self._fc_layers.append(nn.Linear(in_features, int(fc_neurons[i])))
                self._fc_layers.append(nn.ReLU())

                in_features = int(fc_neurons[i])
        else:
            self._fc_layers.append(nn.Identity())

        self._fc_layers = nn.Sequential(*self._fc_layers)

        self._classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(in_features, args.num_classes)
        )

    def forward(self, x):
        # x.shape (batch_size, C, H, W)
        out = self._feature_extractor(x)
        out = self._fc_layers(out)
        out = self._classifier(out)

        return out
