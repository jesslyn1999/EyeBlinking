import cv2
import dlib
import sys
import math
from util import *
from Logger import Logger
import argparse
from pathlib import Path
from graph import save_ear_graph, save_microsleep_perclos_graph
from collections import deque
import threading
import json


exit_flag = 0
queueLock = threading.Lock()
video_work_queue = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

out_processed_results = []

global_idx_video_out = 0

total_input_videos = 0
left_processed_videos = 0

file_stdout = sys.stdout
original_stdout = sys.stdout


class LocalThread(threading.Thread):

    def __init__(self, threadId, name, q):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.q = q

    def run(self):
        print("--(*){}: Starting".format(self.name))
        process_video(self.name, self.q)
        print("--(*){}: Exiting ".format(self.name))


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


def get_ear(frame, eye_points, facial_landmarks):
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


def process_video(thread_name, q):
    global global_idx_video_out, left_processed_videos
    while not exit_flag:
        vid_filename = None
        idx_out = global_idx_video_out
        global_idx_video_out += 1

        queueLock.acquire()
        if video_work_queue:
            vid_filename = q.pop()
        queueLock.release()

        if vid_filename:
            fps = get_fps(vid_filename)
            duration = get_duration(vid_filename)
            total_frame = math.ceil(fps * duration)

            cap = cv2.VideoCapture(vid_filename)

            if not cap.isOpened:
                print('--(!){}: Error opening video capture-{}: {}'.format(thread_name, idx_out, vid_filename))
                return

            print('--(*){}: Success in opening video capture-{}: {}'.format(thread_name, idx_out, vid_filename))

            curr_frame = 0
            vid_ear_scores = []
            vid_time_labels = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                curr_frame += 1
                blinking_ratio_rounded = -1
                blinking_ratio = -1

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(frame_gray)

                # print("{}: Curr frame : {}/{}".format(thread_name, curr_frame, total_frame))

                if faces:
                    faces_area_max_idx = np.argmax(map(find_face_area, faces))
                    face = faces[faces_area_max_idx]

                    landmarks = predictor(frame_gray, face)
                    left_eye_ratio = get_ear(frame, [36, 37, 38, 39, 40, 41], landmarks)
                    right_eye_ratio = get_ear(frame, [42, 43, 44, 45, 46, 47], landmarks)

                    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                    blinking_ratio_rounded = np.round(blinking_ratio * 100) / 100

                vid_ear_scores.append(blinking_ratio)
                vid_time_labels.append(curr_frame / total_frame * duration)

                # frame = cv2.putText(frame, str(blinking_ratio_rounded), (30, 50),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                if cv2.waitKey(10) == 27:  # ESC key
                    break

            cap.release()
            queueLock.acquire()
            left_processed_videos += 1
            sys.stdout = original_stdout
            print("{}: Progress: {}/{}".format(thread_name, left_processed_videos,
                                               total_input_videos))
            sys.stdout = file_stdout
            queueLock.release()

            out_processed_results.append({"idx": idx_out, "time_labels": vid_time_labels,
                                          "ear_scores": vid_ear_scores})


def main(input_processed_result=None):
    global video_work_queue, exit_flag, total_input_videos, file_stdout, original_stdout, \
        out_processed_results

    parser = argparse.ArgumentParser(description='Eye Blinking Detection Rate.')
    parser.add_argument('--model', help='Path to eye detection from face parameter model.',
                        default='model/shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--dir_video', help='Path to directory input video files.')
    parser.add_argument('--output_dir', help='Path to directory video generated. ex: ./output',
                        default='./output')
    parser.add_argument('--batch_time', help='integer to define eye blinking time batch in seconds',
                        default=1200)
    parser.add_argument('--thread_num', help='define thread_number',
                        default=15)
    parser.add_argument('--processed_result', help='json file path of processed_result existed')
    args = parser.parse_args()

    model = args.model
    input_dir_video = args.dir_video
    output_dir = args.output_dir
    batch_time = int(args.batch_time)
    thread_num = int(args.thread_num)
    input_processed_result_path = args.processed_result
    input_processed_result = None

    if input_processed_result_path and Path(input_processed_result_path).is_file():
        input_dir_video = os.path.dirname(input_processed_result_path)

    output_dir = output_dir + '/' + get_split_filename(input_dir_video)[0] + " " + get_time()
    make_dir(output_dir)

    print("--(*)Output folder dir: ", output_dir)
    file_stdout = Logger(output_dir)
    sys.stdout = file_stdout

    output_ear_graph_filename = output_dir + '/EAR_graph.png'
    output_graph_filename = output_dir + '/perclos_microsleep_graph.png'
    output_processed_result = output_dir + '/processed_result.json'

    if input_processed_result_path:
        with open(input_processed_result_path) as json_file:
            input_processed_result = json.load(json_file)

    if not input_processed_result:
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

        # output_vid_filename = output_dir + '/video' + get_split_filename(sample_video)[1][1]

        fps_sample = get_fps(sample_video)

        width_out = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_out = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_test.release()

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        total_input_videos = len(input_videos)
        video_work_queue = deque(input_videos)
        video_work_queue.reverse()
        threads = []

        sys.stdout = original_stdout
        print("--(*)Progress: {}/{}".format(0, total_input_videos))
        sys.stdout = file_stdout

        for i in range(thread_num):
            temp_thread = LocalThread(i, "Thread-{}".format(i), video_work_queue)
            temp_thread.start()
            threads.append(temp_thread)

        # Wait for queue to empty
        while video_work_queue:
            pass

        exit_flag = 1

        # Wait for all threads to complete
        for t in threads:
            t.join()

        sys.stdout = original_stdout
        print("--(*)Write processed results to: ", output_processed_result)
        sys.stdout = file_stdout
        with open(output_processed_result, "w+") as outfile:
            json.dump(out_processed_results, outfile)

    # out = cv2.VideoWriter(output_vid_filename, fourcc, fps_sample, (width_out, height_out))

    def process_report_results(processed_results):
        total_process_time = 0
        final_time_labels = []
        final_ear_ratios = []
        perclos = []
        microsleep = []

        curr_batch = 0
        closing_time_batch = 0
        microsleep_counter_batch = 0
        microsleep_time_batch = 0
        blink_counter = 0
        previous_ratio = 100

        processed_results.sort(key=lambda x: x.get('idx'))

        for result in processed_results:
            time_labels = list(map(lambda x: x + total_process_time, result.get('time_labels')))
            final_time_labels.extend(time_labels)
            ear_scores = result.get('ear_scores')
            final_ear_ratios.extend(ear_scores)

            for idx in range(len(ear_scores)):
                # print("processing result: ", idx)

                if time_labels[idx] // batch_time > curr_batch:
                    curr_batch = time_labels[idx] // batch_time
                    perclos.append(closing_time_batch / batch_time * 100)
                    microsleep.append(microsleep_counter_batch)
                    closing_time_batch = 0
                    microsleep_counter_batch = 0
                    microsleep_time_batch = 0

                if ear_scores[idx] == -1:
                    previous_ratio = 100
                    closing_time_batch = 0
                    microsleep_counter_batch = 0
                    microsleep_time_batch = 0
                else:
                    if 0 <= ear_scores[idx] < 0.20:  # close eye
                        if previous_ratio > 0.20:
                            blink_counter = blink_counter + 1
                        else:
                            delta_time = time_labels[idx] - (time_labels[idx - 1] if time_labels and idx > 0 else 0)
                            microsleep_time_batch += delta_time
                            closing_time_batch += delta_time
                    else:  # open eye
                        if previous_ratio <= 0.20 and 0 < microsleep_time_batch * 1000 < 500:
                            microsleep_counter_batch += 1
                        microsleep_time_batch = 0

                    previous_ratio = ear_scores[idx]

            if time_labels:
                total_process_time = time_labels[-1]

        perclos.append(closing_time_batch / batch_time * 100)
        microsleep.append(microsleep_counter_batch)

        return {'time_labels': final_time_labels, 'ear_ratios': final_ear_ratios,
                'microsleep': microsleep, 'perclos': perclos}

    sys.stdout = original_stdout
    print("--(*)Eye detection is done for all videos")
    sys.stdout = file_stdout

    if input_processed_result:
        out_processed_results = input_processed_result
    detail_results = process_report_results(out_processed_results)
    # out.release()
    cv2.destroyAllWindows()

    print('--(*)Finish program at: ', get_time())

    microsleep_g, perclos_g = detail_results.get('microsleep'), detail_results.get('perclos')
    ear_ratios, time_labels_x = detail_results.get('ear_ratios'), detail_results.get('time_labels')

    print('\n\n\nMICROSLEEP: {}\n\nPERCLOS: {}'.format(microsleep_g, perclos_g))

    # plot for EAR ratio
    save_ear_graph([time_labels_x, ear_ratios], output_ear_graph_filename)

    # plot for microsleep and perclos
    x_labels = np.arange(0, len(microsleep_g) * batch_time, batch_time)
    data = [microsleep_g, perclos_g]
    print('\nX_Labels: {}'.format(x_labels))
    save_microsleep_perclos_graph(data, x_labels, output_graph_filename)


if __name__ == "__main__":
    main()
