import io
import os
import json
import torch
from PIL import Image
from torchvision.models import vit_b_32
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from ts.torch_handler.base_handler import BaseHandler
from time import time
import time as tm
from datetime import datetime

class VisionTransformerHandler(BaseHandler):
    image_processing = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self):
        super(VisionTransformerHandler, self).__init__()
        self.model = vit_b_32(weights=False)
        self.initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        serialized_file = context.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        state_dict = torch.load(model_pt_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        mapping_file_path = os.path.join(model_dir, "imagenet_class_index.json")
        self.mapping = self.load_label_mapping(mapping_file_path)

        self.initialized = True

    def preprocess(self, data):
        images = []
        request_ids = []
        preprocess_entry_time = time()
        
        for row in data:
            image_file_path = row['body']['path']
            image = Image.open(image_file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            try:
                image = self.image_processing(image)
                images.append(image)
                request_ids.append(row['body']['request_id'])
            except RuntimeError as e:
                continue

        print(f"Current batch size being processed: {len(images)}")
        if len(images) == 0:
            return None, None, None

        preprocess_exit_time = time()
        return torch.stack(images).to(self.device), request_ids, preprocess_entry_time, preprocess_exit_time

    def inference(self, data, *args, **kwargs):
        img, request_ids, preprocess_entry_time, preprocess_exit_time = data
        # tm.sleep(0.05)
        outputs = self.model(img)
        return outputs, request_ids, preprocess_entry_time, preprocess_exit_time

    def postprocess(self, inference_output):
        outputs, request_ids, preprocess_entry_time, preprocess_exit_time = inference_output
        processing_time = time() - preprocess_exit_time
        _, predicted_indices = torch.max(outputs, 1)
        predicted_labels = [self.mapping[str(index)][1] for index in predicted_indices.cpu().numpy()]
        return [
            {
                'id': request_id,
                'label': predicted_labels[i],
                'processing_time': processing_time,
                'timestamp_B': preprocess_entry_time
            }
            for i, request_id in enumerate(request_ids)
        ]

    @staticmethod
    def load_label_mapping(mapping_file_path):
        with open(mapping_file_path) as f:
            mapping = json.load(f)
        return mapping

_service = VisionTransformerHandler()

