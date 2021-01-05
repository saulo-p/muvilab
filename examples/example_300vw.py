import os
import sys
from pathlib import Path
from typing import Union
import cv2
import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)
from annotator import Annotator


## Preprocessing pipeline


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
    landmarks_filename = os.path.join(src_path, 'annot', '0'*(6-len(str(globals()['frame_id']))) + str(globals()['frame_id']) + '.pts')
    try:
        landmarks_np = read_pts(landmarks_filename)
    except Exception as e:
        print(e)
        return image

    landmarks_mouth = landmarks_np[MOUTH_OUTER_IDXS, :]

    (x, y, w, h) = cv2.boundingRect(np.array(landmarks_mouth, dtype=np.int))

    BORDER = 10
    roi = image[(y - BORDER):(y + h + BORDER), (x - BORDER):(x + w + BORDER)]

    globals()['frame_id'] += 1

    roi = cv2.resize(roi,(64, 32), interpolation=cv2.INTER_LINEAR)

    return roi


PATH_300VW = '/home/steveml/Dev/mouth-state-detection/data/300VW'
IDX_300VW = sys.argv[1]
MOUTH_OUTER_IDXS = [i for i in range(48,60)]
CLIPS_DIR = os.path.join(ROOT_DIR, 'local_clips')

# Create the folders
if not os.path.exists(CLIPS_DIR):
    os.mkdir(CLIPS_DIR)


# Initialise the annotator
annotator = Annotator([
        {'name': 'talking', 'color': (0, 255, 0)},
        {'name': 'not_talking', 'color': (0, 0, 255)}],
        CLIPS_DIR, sort_files_list=True, N_show_approx=40, screen_ratio=16/9,
        image_resize=1, loop_duration=None, annotation_file='labels.json')

preprocessing = [
    {
        'functor': show_mouth_roi,
        'args': [os.path.join(PATH_300VW, str(IDX_300VW))],
        'kwargs': dict()
    }
]

# Split the video into clips
print('Generating clips from the video...')
frame_id = 1
annotator.video_to_clips(os.path.join(PATH_300VW, str(IDX_300VW), 'vid.avi'), CLIPS_DIR, clip_length=1., overlap=0, resize=1, preprocessing_pipeline=preprocessing)

# Run the annotator
annotator.main()

