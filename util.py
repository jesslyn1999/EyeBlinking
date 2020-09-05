import numpy as np
import subprocess
import json
import shlex
import os
import time
import ray


def euclidean_distance(leftx, lefty, rightx, righty):
    return np.sqrt((leftx - rightx) ** 2 + (lefty - righty) ** 2)


def find_face_area(face):
    return abs((face.left() - face.right()) * (face.bottom() * face.top()))


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_duration(file_path_with_file_name):
    cmd = 'ffprobe -show_entries format=duration -v quiet -of csv="p=0"'

    args = shlex.split(cmd)
    args.append(file_path_with_file_name)

    ffprobe_output = subprocess.check_output(args).decode('utf-8')
    ffprobe_output = json.loads(ffprobe_output)
    return ffprobe_output


def get_fps(file_path_with_file_name):
    cmd = 'ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 ' \
          '-show_entries stream=r_frame_rate'

    args = shlex.split(cmd)
    args.append(file_path_with_file_name)

    ffprobe_output_fps = subprocess.check_output(args).decode('utf-8').strip().split('/')
    return int(ffprobe_output_fps[0]) / int(ffprobe_output_fps[1])


def get_split_filename(filename):
    # param: '/root/dir/sub/file.ext'  out: ('file.ext', ('file', '.ext'))
    base_filename = os.path.basename(filename)
    return base_filename, os.path.splitext(base_filename)


def is_video_file(filename):
    return get_split_filename(filename)[1][1] in ['.mp4']


def make_dir(dir_name):
    return os.mkdir(dir_name)


def get_video_list_in_dir(dir_path):
    result = []
    for root, dirs, files in os.walk(dir_path, topdown=True):
        result.extend(list(filter(lambda file: is_video_file(file), files)))
        result = list(map(lambda file: os.path.join(root, file), result))
    return result


def get_time():
    return time.ctime().replace(":", "")


if __name__ == "__main__":
    exit(0)
