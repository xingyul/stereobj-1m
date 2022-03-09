'''
    Single-GPU training.
'''
import argparse
import json
import glob
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='', help='Input directory [default: ]')
parser.add_argument('--output_dir', default='', help='Output directory [default: ]')
parser.add_argument('--cls_type', default='', help='Object class of interest [default: ]')
parser.add_argument('--split', default='test', help='Dataset split [default: test]')
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.system('mkdir -p {}'.format(args.output_dir))


if __name__ == "__main__":
    all_saved_jsons = glob.glob(os.path.join(args.input_dir, '*.json'))

    combined_dict = {'pred': {}}

    for idx, filename in enumerate(all_saved_jsons):
        with open(filename, 'r') as f:
            pose_pred = json.load(f)

        sequence_id = os.path.basename(filename).split('__')[0]
        frame_id = os.path.basename(filename).split('.')[0].split('__')[1]
        if sequence_id not in combined_dict['pred']:
            combined_dict['pred'][sequence_id] = {}
        combined_dict['pred'][sequence_id][frame_id] = pose_pred

    combined_dict['cls_type'] = args.cls_type
    combined_dict['split'] = args.split

    save_filename = os.path.join(args.output_dir, '{}.json'.format(args.cls_type))
    print(save_filename)
    with open(save_filename, 'w') as f:
        json.dump(combined_dict, f, indent=1)



