from pytorchyolo.detect import detect_image

import sys
import cv2


def process_video(in_path, out_path, model, tracker, resolution=416, conf_thres=0.85, nms_thres=0.5, verbose=True):
    """Draw bboxes and objects' ids on the provided video and save the video in a separate file.

    Args:
        in_path (str): path to the video to process.
        out_path (str): path to the file to save the processed video (*.avi format).
        model: pytorchyolo-darknet model.
        tracker: sorttracker-Sort tracker.
        resolution (int): Size of each image dimension for yolo.
        conf_thres (float): bounding boxes with confidence below conf_thres won't be applied.
        nms_thres (float): IOU threshold for non-maximum suppression.
        verbose (bool): if True prints progress to stdout.
    """
    vid_in = cv2.VideoCapture(in_path)
    vid_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'),
                              vid_in.get(cv2.CAP_PROP_FPS), (int(vid_in.get(3)), int(vid_in.get(4))))

    if verbose:
        frame_total = count_frames(in_path)
        frame_i = 1

    while True:
        _, frame_ = vid_in.read()
        if frame_ is None:
            break
        else:
            detections = detect_image(model, frame_, resolution, conf_thres, nms_thres)
            detections = detections[detections[:, 5] == 0, :]  # leave only detected persons (drop other classes)
            detections = tracker.update(detections[:, :-1])  # pass detections w/o class predictions

            for d in detections:
                draw_bbx(frame_, d)
                draw_id(frame_, d)

            vid_out.write(frame_)

            if verbose:
                print(f"\r{frame_i} of {frame_total} processed", file=sys.stdout, end="")
                frame_i += 1


def draw_bbx(frame_, detection):
    """Draws bounding boxes and object id on the provided frame.

    Args:
        frame_ (nd.array): RGB image of shape HxWxC
        detection (nd.array): [x1, y1, x2, y2, id]
    """
    # bounding box
    pt1 = detection[:2].astype(int)
    pt2 = detection[2:4].astype(int)
    cv2.rectangle(img=frame_, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=1)

    # label
    label = str(int(detection[4]))
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

    pt2 = pt1[0] + label_size[0] + 3, pt1[1] + label_size[1] + 4
    cv2.rectangle(img=frame_, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=-1)
    cv2.putText(frame_, label, (pt1[0], pt1[1] + label_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)


def draw_id(frame_, detection):
    """Draws object id on the provided frame.

    Args:
        frame_ (nd.array): RGB image of shape HxWxC
        detection (nd.array): [x1, y1, x2, y2, id]
    """

    # id and its background
    id_ = str(int(detection[4]))
    background_size = cv2.getTextSize(id_, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

    # bounding box
    pt1 = detection[:2].astype(int)
    pt2 = pt1[0] + background_size[0] + 3, pt1[1] + background_size[1] + 4

    cv2.rectangle(img=frame_, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=-1)
    cv2.putText(frame_, id_, (pt1[0], pt1[1] + background_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)


def count_frames(vid_path):
    """Count the number of frames in the provided video."""
    frames = 0
    vid = cv2.VideoCapture(vid_path)
    while True:
        _, frame = vid.read()
        if frame is None:
            break
        else:
            frames += 1
    return frames
