from pytorchyolo.models import load_model
from sorttracker.tracker import Sort
from run.utils import process_video

import os
import sys
from pathlib import Path


if __name__ == "__main__":

    model_path = r"../config/yolov3.cfg"
    weights_path = r"../weights/yolov3.weights"
    in_dir = r"../input"
    out_dir = r"../output/yolov3_sort"
    vid_file = "campus4-c2.avi"

    os.makedirs(out_dir, exist_ok=True)

    model = load_model(model_path, weights_path)
    for p in model.parameters():
        p.requires_grad = False

    in_path = str(Path(in_dir) / vid_file)
    out_path = str(Path(out_dir) / vid_file)

    process_video(in_path, out_path, model, Sort())
    print(f" | {vid_file} processing completed", file=sys.stdout)
