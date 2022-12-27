'''
extract_mouth_batch.py
    This script will extract mouth crop of every single video inside source directory
    while preserving the overall structure of the source directory content.

Usage:
    python extract_mouth_batch.py [source directory] [pattern] [target directory] [face predictor path]

    pattern: *.avi, *.mpg, etc

Example:
    python scripts/extract_mouth_batch.py evaluation/samples/GRID/ *.mpg TARGET/ common/predictors/shape_predictor_68_face_landmarks.dat

    Will make directory TARGET and process everything inside evaluation/samples/GRID/ that match pattern *.mpg.
'''
import numpy as np
from LipNet2.videos import Video
import os, fnmatch, sys, errno
from skimage import io
from imageio import imread
import shutil

 #sys.argv[1]

#if __name__ == '__main__':
def refine_grid():
    SOURCE_PATH = '/Users/kadkhodm/Desktop/Research/inpainting/original/video_mouth_crop/'
    failure_cnt = 0
    spkrs = os.listdir(SOURCE_PATH)
    for spkr in spkrs:
        if os.path.isdir(os.path.join(SOURCE_PATH, spkr)):
            folders = os.listdir(os.path.join(SOURCE_PATH, spkr))
            for folder in folders:
                if os.path.isdir(os.path.join(SOURCE_PATH,spkr,folder)):
                    image_names = [name for name in os.listdir(os.path.join(SOURCE_PATH,spkr,folder)) if name.endswith('.png')]
                    image_list = [imread(os.path.join(SOURCE_PATH,spkr,folder,image)) for image in image_names]
                    file_size_list = [image.shape for image in image_list]
                    if all([file_size==(50, 100, 3) for file_size in file_size_list]):
                        continue
                    else:
                        failure_cnt += 1
                        print(f"folder {folder} for speaker {spkr} has invalid frame sizes")
                        print(f"number of invalid folder sizes {failure_cnt}")
                        inval_path = os.path.join(SOURCE_PATH,spkr,folder)
                        shutil.rmtree(inval_path)

def refine_grid_video_frames():
    SOURCE_PATH = '/Users/kadkhodm/Desktop/Research/inpainting/original/video_mouth_crop/'
    failure_cnt = 0
    spkrs = os.listdir(SOURCE_PATH)
    for spkr in spkrs:
        if os.path.isdir(os.path.join(SOURCE_PATH, spkr)):
            folders = os.listdir(os.path.join(SOURCE_PATH, spkr))
            for folder in folders:
                if os.path.isdir(os.path.join(SOURCE_PATH,spkr,folder)):
                    image_names = [name for name in os.listdir(os.path.join(SOURCE_PATH,spkr,folder)) if name.endswith('.png')]
                    if len(image_names)==75:
                        continue
                    else:
                        failure_cnt += 1
                        print(f"folder {folder} for speaker {spkr} has invalid frame sizes")
                        print(f"number of invalid folder sizes {failure_cnt}")
                        inval_path = os.path.join(SOURCE_PATH,spkr,folder)
                        shutil.rmtree(inval_path)
