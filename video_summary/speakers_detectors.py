from transformers import DetrFeatureExtractor, DetrForObjectDetection
import torch
from PIL import Image
import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

d = './images/face_sample_1.PNG'
image = Image.open(d).convert(mode='RGB')

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

locations = []
scores = []
images = []
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    # let's only keep detections with score > 0.9
    if score > 0.9 and model.config.id2label[label.item()] == 'person':
        locations.append(box)
        scores.append(round(score.item(), 3))
        images.append(image.crop(box=box))
        # what kind of other objects might be interesting for us?
        """
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
        """

# https://huggingface.co/hustvl/yolos-tiny
# https://huggingface.co/facebook/detr-resnet-50


# image: (4, 454, 1332)    # (3, 800, 1066)
# mean: (3, 1, 1) / (3,)
# std: (3, 1, 1) / (3,)
# image.ndim: 3


# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB
# PIL.PngImagePlugin.PngImageFile image mode=RGBA
