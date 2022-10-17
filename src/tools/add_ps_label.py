import os
import glob
import argparse


instance_and_grasp_type_2_ps_value = {}

instance_and_grasp_type_2_ps_value['055_baseball'] = {}
instance_and_grasp_type_2_ps_value['055_baseball']['power_sphere'] = 0

instance_and_grasp_type_2_ps_value['shap_ball'] = {}
instance_and_grasp_type_2_ps_value['shap_ball']['power_sphere'] = 0

instance_and_grasp_type_2_ps_value['tennis_ball'] = {}
instance_and_grasp_type_2_ps_value['tennis_ball']['power_sphere'] = 0

instance_and_grasp_type_2_ps_value['033_spatula'] = {}
instance_and_grasp_type_2_ps_value['033_spatula']['adducted_thumb'] = 90
instance_and_grasp_type_2_ps_value['033_spatula']['prismatic_4fingers'] = 0

instance_and_grasp_type_2_ps_value['048_hammer'] = {}
instance_and_grasp_type_2_ps_value['048_hammer']['small_diameter'] = 90

instance_and_grasp_type_2_ps_value['003_cracker_box'] = {}
instance_and_grasp_type_2_ps_value['003_cracker_box']['sphere_4fingers'] = 0
instance_and_grasp_type_2_ps_value['003_cracker_box']['medium_wrap'] = 90

instance_and_grasp_type_2_ps_value['mellin'] = {}
instance_and_grasp_type_2_ps_value['mellin']['sphere_4fingers'] = 0
instance_and_grasp_type_2_ps_value['mellin']['medium_wrap'] = 90

instance_and_grasp_type_2_ps_value['pudding'] = {}
instance_and_grasp_type_2_ps_value['pudding']['sphere_4fingers'] = 0
instance_and_grasp_type_2_ps_value['pudding']['medium_wrap'] = 90

instance_and_grasp_type_2_ps_value['blue_brush'] = {}
instance_and_grasp_type_2_ps_value['blue_brush']['adducted_thumb'] = 90

instance_and_grasp_type_2_ps_value['orange_brush'] = {}
instance_and_grasp_type_2_ps_value['orange_brush']['adducted_thumb'] = 90

instance_and_grasp_type_2_ps_value['010_potted_meat_can'] = {}
instance_and_grasp_type_2_ps_value['010_potted_meat_can']['sphere_4fingers'] = 0
instance_and_grasp_type_2_ps_value['010_potted_meat_can']['medium_wrap'] = 90

instance_and_grasp_type_2_ps_value['006_mustard_bottle'] = {}
instance_and_grasp_type_2_ps_value['006_mustard_bottle']['large_diameter'] = 90
instance_and_grasp_type_2_ps_value['006_mustard_bottle']['tripod'] = 0

instance_and_grasp_type_2_ps_value['019_pitcher_base'] = {}
instance_and_grasp_type_2_ps_value['019_pitcher_base']['adducted_thumb'] = 90
instance_and_grasp_type_2_ps_value['019_pitcher_base']['prismatic_4fingers'] = 0

instance_and_grasp_type_2_ps_value['021_bleach_cleanser'] = {}
instance_and_grasp_type_2_ps_value['021_bleach_cleanser']['large_diameter'] = 90
instance_and_grasp_type_2_ps_value['021_bleach_cleanser']['tripod'] = 0

instance_and_grasp_type_2_ps_value['black_dispenser'] = {}
instance_and_grasp_type_2_ps_value['black_dispenser']['adducted_thumb'] = 90
instance_and_grasp_type_2_ps_value['black_dispenser']['tripod'] = 0
instance_and_grasp_type_2_ps_value['black_dispenser']['large_diameter'] = 90

instance_and_grasp_type_2_ps_value['magenta_glass'] = {}
instance_and_grasp_type_2_ps_value['magenta_glass']['large_diameter'] = 90
instance_and_grasp_type_2_ps_value['magenta_glass']['sphere_4fingers'] = 0

instance_and_grasp_type_2_ps_value['yellow_glass'] = {}
instance_and_grasp_type_2_ps_value['yellow_glass']['large_diameter'] = 90
instance_and_grasp_type_2_ps_value['yellow_glass']['sphere_4fingers'] = 0

instance_and_grasp_type_2_ps_value['011_banana'] = {}
instance_and_grasp_type_2_ps_value['011_banana']['tripod'] = 0
instance_and_grasp_type_2_ps_value['011_banana']['prismatic_4fingers'] = 0

instance_and_grasp_type_2_ps_value['025_mug'] = {}
instance_and_grasp_type_2_ps_value['025_mug']['prismatic_2fingers'] = 90
instance_and_grasp_type_2_ps_value['025_mug']['large_diameter'] = 90
instance_and_grasp_type_2_ps_value['025_mug']['sphere_4fingers'] = 0

instance_and_grasp_type_2_ps_value['white_mug'] = {}
instance_and_grasp_type_2_ps_value['white_mug']['prismatic_2fingers'] = 90
instance_and_grasp_type_2_ps_value['white_mug']['large_diameter'] = 90
instance_and_grasp_type_2_ps_value['white_mug']['sphere_4fingers'] = 0

instance_and_grasp_type_2_ps_value['029_plate'] = {}
instance_and_grasp_type_2_ps_value['029_plate']['adducted_thumb'] = 90

instance_and_grasp_type_2_ps_value['cyan_plate'] = {}
instance_and_grasp_type_2_ps_value['cyan_plate']['adducted_thumb'] = 90

instance_and_grasp_type_2_ps_value['orange_plate'] = {}
instance_and_grasp_type_2_ps_value['orange_plate']['adducted_thumb'] = 90

instance_and_grasp_type_2_ps_value['white_plate'] = {}
instance_and_grasp_type_2_ps_value['white_plate']['adducted_thumb'] = 90

instance_and_grasp_type_2_ps_value['brown_ringbinder'] = {}
instance_and_grasp_type_2_ps_value['brown_ringbinder']['adducted_thumb'] = 90

instance_and_grasp_type_2_ps_value['maglia_ringbinder'] = {}
instance_and_grasp_type_2_ps_value['maglia_ringbinder']['adducted_thumb'] = 90

instance_and_grasp_type_2_ps_value['018_plum'] = {}
instance_and_grasp_type_2_ps_value['018_plum']['power_sphere'] = 0

instance_and_grasp_type_2_ps_value['strawberry'] = {}
instance_and_grasp_type_2_ps_value['strawberry']['power_sphere'] = 0

instance_and_grasp_type_2_ps_value['red_wood_block'] = {}
instance_and_grasp_type_2_ps_value['red_wood_block']['tripod'] = 0

instance_and_grasp_type_2_ps_value['yellow_wood_block'] = {}
instance_and_grasp_type_2_ps_value['yellow_wood_block']['tripod'] = 0

instance_and_grasp_type_2_ps_value['030_fork'] = {}
instance_and_grasp_type_2_ps_value['030_fork']['prismatic_4fingers'] = 0

instance_and_grasp_type_2_ps_value['031_spoon'] = {}
instance_and_grasp_type_2_ps_value['031_spoon']['prismatic_4fingers'] = 0

instance_and_grasp_type_2_ps_value['032_knife'] = {}
instance_and_grasp_type_2_ps_value['032_knife']['prismatic_4fingers'] = 0

instance_and_grasp_type_2_ps_value['037_scissors'] = {}
instance_and_grasp_type_2_ps_value['037_scissors']['adducted_thumb'] = 90
instance_and_grasp_type_2_ps_value['037_scissors']['prismatic_4fingers'] = 0

instance_and_grasp_type_2_ps_value['040_large_marker'] = {}
instance_and_grasp_type_2_ps_value['040_large_marker']['prismatic_4fingers'] = 0

instance_and_grasp_type_2_ps_value['026_sponge'] = {}
instance_and_grasp_type_2_ps_value['026_sponge']['adducted_thumb'] = 90
instance_and_grasp_type_2_ps_value['026_sponge']['power_disk'] = 0

instance_and_grasp_type_2_ps_value['inverted_color_sponge'] = {}
instance_and_grasp_type_2_ps_value['inverted_color_sponge']['adducted_thumb'] = 90
instance_and_grasp_type_2_ps_value['inverted_color_sponge']['power_disk'] = 0

instance_and_grasp_type_2_ps_value['001_chips_can'] = {}
instance_and_grasp_type_2_ps_value['001_chips_can']['large_diameter'] = 90
instance_and_grasp_type_2_ps_value['001_chips_can']['sphere_4fingers'] = 0

instance_and_grasp_type_2_ps_value['colors'] = {}
instance_and_grasp_type_2_ps_value['colors']['large_diameter'] = 90
instance_and_grasp_type_2_ps_value['colors']['sphere_4fingers'] = 0

instance_and_grasp_type_2_ps_value['leather_wallet'] = {}
instance_and_grasp_type_2_ps_value['leather_wallet']['adducted_thumb'] = 90
instance_and_grasp_type_2_ps_value['leather_wallet']['power_disk'] = 0

instance_and_grasp_type_2_ps_value['sweet_years_wallet'] = {}
instance_and_grasp_type_2_ps_value['sweet_years_wallet']['adducted_thumb'] = 90
instance_and_grasp_type_2_ps_value['sweet_years_wallet']['power_disk'] = 0


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_name', type=str, default=None)
    # parser.add_argument('--synthetic', action='store_true')
    # args = parser.parse_args()

    # DATASET_NAME = os.path.join(
    #     'data', 'synthetic' if args.synthetic else 'real', 'frames', 
    #     args.dataset_name
    # )

    # metadatas_path = os.path.join(DATASET_NAME, '*', '*', '*', 
    #                               'metadata', 'seq*', 'data.log') 
    # files = glob.glob(metadatas_path)
    # if files == []:
    #     raise Exception('No metadata files found at path ' + metadatas_path)

    # for f in files:
    #     with open(f, 'r') as metadata:
    #         line = metadata.readline().split(' ')
    #         instance = line[4]
    #         grasp_type = line[5]
        
    #     ps_value = instance_and_grasp_type_2_ps_value[instance][grasp_type]

    #     with open(f, 'a') as metadata:
    #         metadata.writelines('\npronation-supination ' + str(ps_value))
