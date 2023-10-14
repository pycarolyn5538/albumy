import requests
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

def get_file_contents(filename):
    """ Given a filename,
        return the contents of that file
    """
    try:
        with open(filename, 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)


API_TOKEN = get_file_contents('/Users/goudanhan/Downloads/albumy/apikey_tagging.txt')
API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
headers = {"Authorization": API_TOKEN}


def tagging_query(filename):
    # for file in filename:
        # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        print(filename)
        image = Image.open(filename)

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        object_list = set()
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            object_list.add(model.config.id2label[label.item()])
        return list(object_list)