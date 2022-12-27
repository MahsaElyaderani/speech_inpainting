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

from LipNet2.videos import Video
import os, fnmatch, sys, errno
from skimage import io

SOURCE_PATH = '/Users/kadkhodm/Desktop/Research/inpainting/original/video_normal/' #sys.argv[1]
SOURCE_EXTS = '*.mpg'#sys.argv[2]
TARGET_PATH = '/Users/kadkhodm/Desktop/Research/inpainting/original/video_mouth_crop/'#sys.argv[3]
FACE_PREDICTOR_PATH = '/Users/kadkhodm/Desktop/Research/inpainting/predictors/shape_predictor_68_face_landmarks.dat'#sys.argv[4]

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def mouth_crop():
    failure_cnt = 0
    spkrs = os.listdir(SOURCE_PATH)
    for spkr in spkrs:
        if os.path.isdir(os.path.join(SOURCE_PATH, spkr)):
            files = os.listdir(os.path.join(SOURCE_PATH, spkr))
            if os.path.exists(os.path.join(TARGET_PATH, spkr)):
                print(f"target_dir {os.path.join(TARGET_PATH, spkr)} exists")
                continue
            else:
                for file in files:
                    if file.endswith('.mpg'):
                        try:
                            filepath = os.path.join(SOURCE_PATH,spkr,file)
                            print("Processing: {}".format(filepath))
                            video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH).from_video(filepath)

                            # filepath_wo_ext = os.path.splitext(filepath)[0]
                            filepath_wo_ext = filepath.split("video_normal/")[-1].split(".mpg")[0] #
                            target_name = filepath.split("video_normal/")[-1].split('/')[1].split(".mpg")[0]
                            target_dir = os.path.join(TARGET_PATH, filepath_wo_ext)
                            # spkr_folder = filepath.split("video_normal/")[-1].split('/')[0]

                            mkdir_p(target_dir)

                            i = 0
                            for frame in video.mouth:
                                io.imsave(os.path.join(target_dir, target_name+"_{0:03d}.png".format(i)), frame)
                                i += 1
                        except ValueError:
                            failure_cnt = failure_cnt + 1
                            print("failure counter: ", failure_cnt)
                            print("could not crop the video file in path: ", filepath)