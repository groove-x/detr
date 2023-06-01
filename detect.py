# -*- coding: utf-8 -*-

# !pip install -q git+https://github.com/huggingface/transformers.git

# !pip install -q timm

"""## Prepare the image using DetrFeatureExtractor

Let's use the image of the two cats chilling on a couch once more. It's part of the [COCO](https://cocodataset.org/#home) object detection validation 2017 dataset.
"""

from typing import List
import requests

from PIL import Image
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np


"""Let's first apply the regular image preprocessing using `DetrFeatureExtractor`. The feature extractor will resize the image (minimum size = 800, max size = 1333), and normalize it across the channels using the ImageNet mean and standard deviation."""

from transformers import DetrFeatureExtractor
from transformers import DetrForObjectDetection

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, prob, boxes):
    global model
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("last_detected.jpg")

def plot_results_pillow(pil_img, prob, boxes):
    global model
    pil_img2 = pil_img.copy()
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(pil_img2)
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        c = [int(255 * a) for a in c]
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)], outline=tuple(c), width=3
        )

        cl = p.argmax()
        text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        # print(f"{text=}")
        draw.text((xmin, ymin), text, "red")
        # draw.textbbox((xmin, ymin), text)
    return pil_img2

def plot_results_opencv(cvimg: np.ndarray, prob, boxes) -> np.ndarray:
    global model
    cvimg2 = cvimg.copy()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        c = [int(255 * a) for a in c]
        cv2.rectangle(cvimg2,
            pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), color=tuple(c), thickness=3
        )

        cl = p.argmax()
        text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        cv2.putText(cvimg2, text, org=(int(xmin), int(ymin)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6, color=(0, 0, 255), thickness=2)
        # draw.textbbox((xmin, ymin), text)
    return cvimg2

def detect_image(im):
    global model
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    encoding = feature_extractor(im, return_tensors="pt")
    encoding.keys()

    print(encoding['pixel_values'].shape)

    """## Forward pass

    Next, let's send the pixel values and pixel mask through the model. We use the one with a ResNet-50 backbone here (it obtains a box AP of 42.0 on COCO validation 2017).
    """

    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    outputs = model(**encoding)

    """Let's visualize the results!"""

    # keep only predictions of queries with 0.9+ confidence (excluding no-object class)
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # rescale bounding boxes
    target_sizes = torch.tensor(im.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    pil_img2 = plot_results_pillow(im, probas[keep], bboxes_scaled)
    pil_img2.save("last_detected_pillow.jpg")

    cvimg = pil2cv(im)
    cvimg2 = plot_results_opencv(cvimg, probas[keep], bboxes_scaled)
    cv2.imwrite("last_detected_opencv.jpg", cvimg2)

def cv2pil(image: np.ndarray):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image.copy()


def pil2cv(image) -> np.ndarray:
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def count_frames_in_video(video):
    i = 0
    while True:
        ret, _ = video.read()
        if not ret:
            break
        i += 1
    return i


def select_frames_in_video(video, num_frames: int, N: int) -> List[np.ndarray]:
    rotate = False
    if num_frames > N:
        selected_idx = [int(num_frames * i / N) for i in range(N)]
    else:
        selected_idx = [i for i in range(N)]

    frames = []
    for i in selected_idx:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            break
        if rotate:
            frame = np.rot90(frame, 3)
        frames.append(frame)
    return frames


def detect_movie(video_path: str):
    # global video, num_frames, im
    video = cv2.VideoCapture(video_path)
    out_video_path = f"by_detr_{Path(video_path).stem}.mp4"
    FPS = 15
    writer = None
    # Extract frames from the video
    num_frames = count_frames_in_video(video)
    print(f"{num_frames=}")
    video = cv2.VideoCapture(video_path)
    N = 400
    frames = select_frames_in_video(video, num_frames, N)
    global model
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    for i, frame in enumerate(frames):
        im = Image.fromarray(frame)
        print(f"{i} / {len(frames)}")
        W, H = im.width, im.height

        if writer is None:
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_video_path, codec, FPS, (W, H))

        encoding = feature_extractor(im, return_tensors="pt")
        outputs = model(**encoding)

        # keep only predictions of queries with 0.9+ confidence (excluding no-object class)
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        # rescale bounding boxes
        target_sizes = torch.tensor(im.size[::-1]).unsqueeze(0)
        postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
        bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

        cvimg = frame
        cvimg2 = plot_results_opencv(cvimg, probas[keep], bboxes_scaled)
        writer.write(cvimg2)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    SAMPLE_URL = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    parser = argparse.ArgumentParser(description="DETR detection")
    group = parser.add_argument_group('input_type')
    group.add_argument("--path", help="path to image")
    group.add_argument("--url", help="URL to image")
    group.add_argument("--video", help="path to video")

    args = parser.parse_args()
    if args.url:
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        im = Image.open(requests.get(url, stream=True).raw)
    elif args.path:
        path = Path(args.path)
        im = Image.open(str(path))

    if args.url or args.path:
        detect_image(im)
    elif args.video:
        video_path = args.video
        detect_movie(video_path)
