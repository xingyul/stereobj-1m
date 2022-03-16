'''
    Merge JSON prediction files of all object classes
'''
import argparse
import json
import glob
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='', help='Input directory [default: ]')
parser.add_argument('--output_dir', default='', help='Output directory [default: ]')
parser.add_argument('--split', default='test', help='Dataset split [default: test]')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.system('mkdir -p {}'.format(args.output_dir))


if __name__ == "__main__":

    classes = ['blade_razor', 'hammer', 'microplate', 'needle_nose_pliers', 'pipette_0.5_10',
            'pipette_100_1000', 'pipette_10_100', 'screwdriver', 'side_cutters',
            'sterile_tip_rack_10', 'sterile_tip_rack_1000', 'sterile_tip_rack_200',
            'tape_measure', 'tube_rack_1.5_2_ml', 'tube_rack_50_ml', 'wire_stripper', 'wrench']

    combined_dict = {'split': args.split, 'pred': {}}

    for cls_type in classes:
        if cls_type not in combined_dict:
            combined_dict['pred'][cls_type] = {}

        filename = os.path.join(args.input_dir, cls_type + '.json')
        with open(filename, 'r') as f:
            pose_pred = json.load(f)

        combined_dict['pred'][cls_type] = pose_pred['pred']

    save_filename = os.path.join(args.output_dir, 'merged.json')
    print(save_filename)
    with open(save_filename, 'w') as f:
        json.dump(combined_dict, f, indent=1)



