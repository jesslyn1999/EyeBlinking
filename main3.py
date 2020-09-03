import cv2
import dlib
import sys
import math
from util import *
from Logger import Logger
import argparse
from pathlib import Path
from graph import save_ear_graph, save_microsleep_perclos_graph


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    dim = (640, 480)
    return cv2.resize(image, dim, interpolation=inter)


def get_EAR(frame, eye_points, facial_landmarks):
    left_point = [facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y]
    right_point = [facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y]
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # Drawing horizontal and vertical line
    hor_line = cv2.line(frame, (left_point[0], left_point[1]), (right_point[0], right_point[1]), (255, 0, 0), 3)
    ver_line = cv2.line(frame, (center_top[0], center_top[1]), (center_bottom[0], center_bottom[1]), (255, 0, 0), 3)

    hor_line_length = euclidean_distance(left_point[0], left_point[1], right_point[0], right_point[1])
    ver_line_length = euclidean_distance(center_top[0], center_top[1], center_bottom[0], center_bottom[1])

    return ver_line_length / hor_line_length


def main():
    parser = argparse.ArgumentParser(description='Eye Blinking Detection Rate.')
    parser.add_argument('--model', help='Path to eye detection from face parameter model.',
                        default='model/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--dir_video', help='Path to directory input video files.')
    parser.add_argument('--output_dir', help='Path to directory video generated. ex: ./output',
                        default='./output')
    parser.add_argument('--batch_time', help='integer to define eye blinking time batch in seconds',
                        default=1200)
    args = parser.parse_args()

    model = args.model
    input_dir_video = args.dir_video
    output_dir = args.output_dir
    batch_time = int(args.batch_time)

    if not (input_dir_video and Path(input_dir_video).is_dir()):
        print("--(!)Dir video path is not valid")
        exit(0)

    input_videos = get_video_list_in_dir(input_dir_video)

    if not input_videos:
        print("--(!)Dir video path has no video .mp4")
        exit(0)

    if not Path(model).is_file():
        print("--(!)Model file path is not valid")
        exit(0)

    if not Path(output_dir).is_dir():
        print("--(!)Output dir path is not valid")
        print("--(.)Making dir path...")
        make_dir(output_dir)

    if not type(batch_time) == int:
        print("--(!)Batch_time must be in integer")

    cap_test = None
    sample_video = None
    for video in input_videos:
        cap_test = cv2.VideoCapture(video)
        if not cap_test.isOpened:
            continue
        sample_video = video
        break

    if not sample_video:
        print("--(!)Video can't be processed at all")
        exit(0)

    output_dir = output_dir + '/' + get_split_filename(input_dir_video)[0] + " " + get_time()
    make_dir(output_dir)

    print("--(!)Output folder dir: ", output_dir)
    original_stdout = sys.stdout
    sys.stdout = Logger(output_dir)

    output_vid_filename = output_dir + '/video' + get_split_filename(sample_video)[1][1]
    output_ear_graph_filename = output_dir + '/EAR_graph.png'
    output_graph_filename = output_dir + '/perclos_microsleep_graph.png'

    fps_sample = get_fps(sample_video)

    width_out = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_out = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_test.release()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_vid_filename, fourcc, fps_sample, (width_out, height_out))
    font = cv2.FONT_HERSHEY_SIMPLEX

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

    eye_blink_signal_ear = []
    eye_blink_signal_time = []

    global previous_ratio, blink_counter, blinking_time_batch, \
        curr_batch, microsleep_counter_batch, microsleep_time_batch, \
        total_process_time, total_process_frame

    total_process_time = 0
    total_process_frame = 0
    blinking_time_batch = 0
    curr_batch = 0
    microsleep_counter_batch = 0
    microsleep_time_batch = 0
    blink_counter = 0
    previous_ratio = 100
    perclos = []
    microsleep = []

    def process_video(vid_filename):
        global previous_ratio, blink_counter, blinking_time_batch, \
            curr_batch, microsleep_counter_batch, microsleep_time_batch, \
            total_process_time, total_process_frame

        fps = get_fps(vid_filename)
        duration = get_duration(vid_filename)
        total_frame = math.ceil(fps * duration)

        cap = cv2.VideoCapture(vid_filename)

        if not cap.isOpened:
            print('--(!)Error opening video capture: {}'.format(vid_filename))
            return

        print('--(*)Success in opening video capture: {}'.format(vid_filename))

        curr_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                total_process_frame += total_frame
                total_process_time += duration
                break

            curr_frame += 1

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(frame_gray)

            blinking_ratio_rounded = -1

            curr_process_time = total_process_time + curr_frame / total_frame * duration

            # print("Curr process_time : ", curr_process_time, batch_time, curr_process_time // batch_time, curr_batch)

            if curr_process_time // batch_time > curr_batch:
                curr_batch = curr_process_time // batch_time
                perclos.append(blinking_time_batch / batch_time * 100)
                microsleep.append(microsleep_counter_batch)
                microsleep_counter_batch = 0
                microsleep_time_batch = 0
                blinking_time_batch = 0

            if not faces:
                previous_ratio = 100
                microsleep_counter_batch = 0
                microsleep_time_batch = 0
                blinking_time_batch = 0
            else:
                faces_area_max_idx = np.argmax(map(find_face_area, faces))

                face = faces[faces_area_max_idx]

                landmarks = predictor(frame_gray, face)
                left_eye_ratio = get_EAR(frame, [36, 37, 38, 39, 40, 41], landmarks)
                right_eye_ratio = get_EAR(frame, [42, 43, 44, 45, 46, 47], landmarks)

                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

                blinking_ratio_1 = blinking_ratio * 100
                blinking_ratio_2 = np.round(blinking_ratio_1)
                blinking_ratio_rounded = blinking_ratio_2 / 100

                eye_blink_signal_ear.append(blinking_ratio)
                eye_blink_signal_time.append(curr_process_time)

                if 0 <= blinking_ratio < 0.20:
                    if previous_ratio > 0.20:
                        blink_counter = blink_counter + 1
                    else:
                        microsleep_time_batch = 1 / fps
                        blinking_time_batch += 1 / fps
                else:
                    if previous_ratio <= 0.20:
                        if 0 < microsleep_time_batch * 1000 < 500:
                            microsleep_counter_batch += 1
                    microsleep_time_batch = 0

                previous_ratio = blinking_ratio

            frame = cv2.putText(frame, str(blinking_ratio_rounded), (30, 50), font, 2, (0, 0, 255), 5)
            out.write(frame)

            if cv2.waitKey(10) == 27:  # ESC key
                break

        cap.release()

    total_input_videos = len(input_videos)
    for idx, video in enumerate(input_videos):
        sys.stdout = original_stdout
        print("Progress: {}/{}".format(idx + 1, total_input_videos))
        sys.stdout = Logger(output_dir)
        process_video(video)

    perclos.append(blinking_time_batch / batch_time * 100)
    microsleep.append(microsleep_counter_batch)

    out.release()
    cv2.destroyAllWindows()

    print('\n\n\nMICROSLEEP: {}\n\nPERCLOS: {}'.format(microsleep, perclos))

    # plot for EAR ratio
    save_ear_graph([eye_blink_signal_time, eye_blink_signal_ear], output_ear_graph_filename)

    # plot for microsleep and perclos
    x_labels = np.arange(0, len(microsleep) * batch_time, batch_time)
    data = [microsleep, perclos]
    print('\nX_Labels: {}'.format(x_labels))
    save_microsleep_perclos_graph(data, x_labels, output_graph_filename)


if __name__ == "__main__":
    main()
