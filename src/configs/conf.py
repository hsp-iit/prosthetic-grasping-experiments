SOURCE = ['Head_d435', 'Wrist_d435', 'Wrist_t265']
DATASET_TYPE = ['SingleSourceVideo',  'SingleSourceImage']
SPLIT = ['random']

INPUT = ['rgb']
OUTPUT = ['preshape', 'grasp_type', 'instance']

MODEL = ['cnn_rnn', 'cnn']
FEATURE_EXTRACTOR = ['mobilenet_v2']
PRETRAIN = ['imagenet']
RNN_TYPE = ['lstm', 'rnn']

TEST_TYPE = ['test_same_person', 'test_different_velocity', 'test_from_ground',
             'test_seated', 'test_different_background_1',
             'test_different_background_2', 'test_new_instances',
             'test_new_categories',
             'test_prosthesis', 'test_prosthesis_new_instances',
             'test_prosthesis_new_light_condition',
             'test_prosthesis_new_categories']


RAISE_VALUE_ERROR_STRING = 'Value {} is not defined for {} argument. ' \
                           'It must be one among {}'
