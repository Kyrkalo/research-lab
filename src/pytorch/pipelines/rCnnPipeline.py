import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class RCnnPipeline:
    def __init__(self, configs=None):
        self.device = configs["device"] if configs and "device" in configs else self.device
        self.config = configs if configs else self.config
        pass

    def get_model_instance_segmentation(self):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config["num_classes"])

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            self.config["num_classes"]
        )
        return model
    
    def setup(self):
        self.model = self.get_model_instance_segmentation()
        self.model.to(self.device)
        
        return self

    def run(self):
        return self