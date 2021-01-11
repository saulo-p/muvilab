import argparse
import logging
import glob
import os
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from annotator import Annotator

logger = logging.getLogger('video-to-clips')

def read_pts(filename: Union[str, bytes, Path]) -> np.ndarray:
    """Read a .PTS landmarks file into a numpy array
    https://stackoverflow.com/questions/59591181/does-python-have-a-standard-pts-reader-or-parser
    """
    with open(filename, 'rb') as f:
        # process the PTS header for n_rows and version information
        rows = version = None
        for line in f:
            if line.startswith(b"//"):  # comment line, skip
                continue
            header, _, value = line.strip().partition(b':')
            if not value:
                if header != b'{':
                    raise ValueError("Not a valid pts file")
                if version != 1:
                    raise ValueError(f"Not a supported PTS version: {version}")
                break
            try:
                if header == b"n_points":
                    rows = int(value)
                elif header == b"version":
                    version = float(value)  # version: 1 or version: 1.0
                elif not header.startswith(b"image_size_"):
                    # returning the image_size_* data is left as an excercise
                    # for the reader.
                    raise ValueError
            except ValueError:
                raise ValueError("Not a valid pts file")

        # if there was no n_points line, make sure the closing } line
        # is not going to trip up the numpy reader by marking it as a comment
        points = np.loadtxt(f, max_rows=rows, comments="}")

    if rows is not None and len(points) < rows:
        raise ValueError(f"Failed to load all {rows} points")
    return points

def show_mouth_roi(image, src_path):
    MOUTH_OUTER_IDXS = [i for i in range(48,60)]

    landmarks_filename = os.path.join(src_path, 'annot', '0'*(6-len(str(globals()['frame_id']))) + str(globals()['frame_id']) + '.pts')
    globals()['frame_id'] += 1
    try:
        landmarks_np = read_pts(landmarks_filename)
    except Exception as e:
        print(e)
        return image

    landmarks_mouth = landmarks_np[MOUTH_OUTER_IDXS, :]

    (x, y, w, h) = cv2.boundingRect(np.array(landmarks_mouth, dtype=np.int))

    BORDER = 10
    roi = image[(y - BORDER):(y + h + BORDER), (x - BORDER):(x + w + BORDER)]


    roi = cv2.resize(roi,(64, 32), interpolation=cv2.INTER_LINEAR)

    return roi


def create_clips(input_videos, output_dir, clips_length, clips_overlap, crop_mouth_roi):
    # Define preprocessing pipeline
    if crop_mouth_roi:
        preprocessing = [
            {
                'functor': show_mouth_roi,
                'args': list(),
                'kwargs': dict()
            }
        ]
    else:
        preprocessing = list()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif os.listdir(output_dir):
        # If directory is not empty, ask user if content should be deleted.
        ans = None
        while ans not in {'y', 'n'}:
            ans = input('Directory {} is not empty. Do you want to remove its contents? y or n: '.format(output_dir))
        else:
            if ans == 'y':
                logger.info('Removing contents of directory %s', output_dir)
                contents = glob.glob(os.path.join(output_dir,'*'))
                for c in contents:
                    try:
                        os.remove(c)
                    except:
                        logger.warning('Failed to remove file %s', c)

    for vid in input_videos:
        if crop_mouth_roi:
            globals()['frame_id'] = 1
            preprocessing[0]['args'] = [ os.path.dirname(vid) ]
        try:
            Annotator.video_to_clips(vid, output_dir, clips_length, overlap=clips_overlap, preprocessing_pipeline=preprocessing)
        except Exception as e:
            logger.error('Error processing video {}: '.format(vid), e)
        else:
            logger.info('Video %s converted to clips succesfully.', vid)



def parse_cli_args():
    parser = argparse.ArgumentParser(
        description='Creates short clips from input video files.'
    )

    parser.add_argument('--log-level',
        help='logger level (see Python logging docs)',
        default='INFO'
    )
    parser.add_argument('--log-verbose',
        default=False,
        help='False: logs to clips-labeling.log, True: also prints to stdout.'
    )

    parser.add_argument('input_videos_file',
        help='File that contains list of paths to input videos.',
        type=Path
    )
    parser.add_argument('output_dir',
        help='Path to output directory where clips will be stored.',
        type=Path
    )

    parser.add_argument('--clips-duration',
        help='Duration [seconds] of the output clips.',
        type=float,
        default=1.
    )
    parser.add_argument('--clips-frame-count',
        help='Number of frames of the output clips.\nClip duration is recommended but frames takes precedence for debug needs.',
        type=int
    )
    parser.add_argument('--clips-overlap',
        help='Amount of overlap between consecutive clips of same video [0: no overlap, 1: total overlap[',
        type=float,
        default=0.
    )

    parser.add_argument('--crop-mouth',
        help='Whether to create clips of mouth ROIs only. 0: no crop, 1: crop',
        type=int,
        default=0
    )

    args = parser.parse_args()

    # Logging configuration
    logging.basicConfig(filename='clips-labeling.log', level=args.log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    if args.log_verbose:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    # Args postprocessing
    clips_length = args.clips_frame_count if args.clips_frame_count is not None else args.clips_duration

    with open(args.input_videos_file, 'r') as fin:
        videos = fin.read().splitlines()

    return videos, args.output_dir, clips_length, args.clips_overlap, bool(args.crop_mouth)

if __name__ == "__main__":

    videos_list, output_dir, clips_length, clips_overlap, do_crop_mouth = parse_cli_args()

    logger.info('Generating Clips...\nVideos: {}\nOutput directory: {}\nClip Length: {}\nClips Overlap: {}'.format(
        videos_list, output_dir, clips_length, clips_overlap))

    create_clips(videos_list, output_dir, clips_length, clips_overlap, do_crop_mouth)
