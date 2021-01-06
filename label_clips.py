import argparse
import logging
import glob
import json
import os
import shutil
import sys
from pathlib import Path

from annotator import Annotator

logger = logging.getLogger('label-clips')


def label_clips(clips_dir, labels_file, n_views):
    # Initialise the annotator
    annotator = Annotator(
        [
            {'name': 'talking', 'color': (0, 255, 0)},
            {'name': 'not_talking', 'color': (0, 0, 255)}
        ],
        clips_dir, sort_files_list=True, N_show_approx=n_views, screen_ratio=16/9,
        image_resize=1, loop_duration=None, annotation_file=str(labels_file)
    )

    # call annotation tool
    annotator.main()

def copy_clips(export_clips_dir, clips_dir, labels_file):
    # Class subdirectories creation or cleaning
    for label in ['talking', 'not_talking']:
        class_directory = os.path.join(export_clips_dir, label)

        if not os.path.exists(class_directory):
            os.makedirs(class_directory)
        elif os.listdir(class_directory):
            ans = None
            while ans not in {'y', 'n'}:
                ans = input('Directory {} is not empty. Do you want to remove its contents? y or n: '.format(class_directory))
            else:
                if ans == 'y':
                    logger.info('Removing contents of directory {}'.format(class_directory))
                    contents = glob.glob(os.path.join(class_directory,'*'))
                    for c in contents:
                        try:
                            os.remove(c)
                        except:
                            logger.warning('Failed to remove file {}'.format(c))

    # Copy clips to class subdirectories
    with open(labels_file, 'r') as fin:
        annot_list = json.loads(fin.read())

    for annot in annot_list:
        src = annot['video']
        dst = os.path.join(export_clips_dir, annot['label'])

        logger.debug('Copying file {} to {}'.format(src, dst))

        shutil.copy(src, dst)


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description='Labeling helper. Runs muvilab on target specified directory and manage labeled clips.'
    )

    parser.add_argument('--log-level',
        help='logger level (see Python logging docs)',
        default='INFO'
    )
    parser.add_argument('--log-verbose',
        default=False,
        help='False: logs to clips-labeling.log, True: also prints to stdout.'
    )

    parser.add_argument('clips_dir',
        help='Directory that contains clips to label',
        type=Path
    )
    parser.add_argument('labels_file',
        help='Path to json file to read/write labels',
        type=Path
    )

    parser.add_argument('--num-views',
        help='Approximate number of display windows shown per page.',
        type=int,
        default=20
    )
    parser.add_argument('--export-path',
        help='If a path is specified, at the end of the labeling session, the clips will be copied to the target path, creating one directory for each class.',
        type=Path,
    )

    args = parser.parse_args()

    # Logging configuration
    logging.basicConfig(filename='clips-labeling.log', level=args.log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    if args.log_verbose:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    return args.clips_dir, args.labels_file, args.num_views, args.export_path

if __name__ == "__main__":

    clips_dir, labels_file, n_views, copy_clips_dir = parse_cli_args()

    label_clips(clips_dir, labels_file, n_views)

    if copy_clips_dir is not None:
        copy_clips(copy_clips_dir, clips_dir, labels_file)

