from functools import partial

import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


class ModelCompilation(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.device = "cuda:0"
        if not torch.cuda.is_available():
            self.device = "cpu"
        match model_name:
            case "detr1":
                self.model = self.create_model_detr(num_classes=5, num_queries=20)
                self.model.load_state_dict(torch.load('models/detr1.pth'))
            case "detr2":
                self.model = self.create_model_detr(num_classes=2, num_queries=200)
                self.model.load_state_dict(torch.load('models/detr_uav_vsai_result.pth'))
            case "detr3":
                self.model = self.create_model_detr(num_classes=3, num_queries=200)
                self.model.load_state_dict(torch.load('models/detr_drone_v3.pth'))
            case "detr4":
                self.model = self.create_model_detr(num_classes=2, num_queries=300)
                self.model.load_state_dict(torch.load('models/detr_uav.pth'))


    def create_model_detr(self, num_classes, num_queries):
        model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=False)
        in_features = model.class_embed.in_features
        model.class_embed = nn.Linear(in_features=in_features, out_features=num_classes)
        model.num_queries = num_queries
        model.query_embed = nn.Embedding(num_queries, 256)
        return model.to(self.device)

    def forward(self, x):
        self.model.eval()
        y = x.to(self.device)
        return self.model(y)
