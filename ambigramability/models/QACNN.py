import sys, os
from omegaconf import OmegaConf

from torchvision import models
import torch
import torch.nn as nn

class QACnn(nn.Module):
    def __init__(self, cnn_path, base_cnn_type='resnet:34', model_cfg={'id': 3, 'num_classes': 52}, num_classes=None) -> None:
        super().__init__()
        self.num_classes = num_classes
        input_feature_size = 512
        hidden_feature_size = input_feature_size
        
        ## load pre-train model directly.
        if cnn_path.split('.')[-1] == 'pkl':
            self.cnn_model = torch.load(cnn_path)
        ## load weights and reconstruct pre-train model.
        elif cnn_path.split('.')[-1] == 't7':
            sys.path.append("../../")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.cnn_model = compose_net(OmegaConf.create(model_cfg))
            self.cnn_model = self.cnn_model.to(device)
            self.cnn_model.load_state_dict(
                torch.load(cnn_path)['net']
            )
        else:
            raise NotImplementedError("Undefined extension. select `.pkl` or `.t7`.")

        for param in self.cnn_model.parameters():
            param.requires_grad = False


        if base_cnn_type.split(':')[0] == 'resnet':
            self.cnn_model.fc = nn.Linear(self.cnn_model.fc.in_features, input_feature_size)
        elif base_cnn_type.split(':')[0] == 'vgg':
            self.cnn_model.classifier = nn.Linear(25088, input_feature_size)
        else:
            raise NotImplementedError("Undefined base cnn model type. select `resnet` or `vgg`.")
        
        self.normal_embed = nn.Linear(num_classes, input_feature_size)
        for param in self.normal_embed.parameters():
            param.requires_grad = False

        
        self.classifier = nn.Sequential(
            nn.Linear(input_feature_size * 2, hidden_feature_size),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(hidden_feature_size, hidden_feature_size),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(hidden_feature_size, hidden_feature_size),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(hidden_feature_size, 2),
        )
        
    def forward(self, img, label):
        img_features = self.cnn_model(img)
        label = torch.nn.functional.one_hot(torch.squeeze(label), num_classes=self.num_classes).float()
        label_emb = self.normal_embed(label)
        
        if len(label_emb.shape) == 1:
            label_emb = torch.unsqueeze(label_emb, 0)
        
        features = torch.cat([img_features, label_emb], dim=1)
        out = self.classifier(features)
        return out
    
    
    
class VQAModel(torch.nn.Module):
    def __init__(self, cnn_path, num_classes, embedding_size, hidden_size_ch, dropout, base_cnn_type='resnet:34', model_cfg={}):
        super(VQAModel, self).__init__()
        
        ## load pre-train model directly.
        if cnn_path.split('.')[-1] == 'pkl':
            self.cnn_model = torch.load(cnn_path)
        ## load weights and reconstruct pre-train model.
        elif cnn_path.split('.')[-1] == 't7':
            sys.path.append("../../")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.cnn_model = compose_net(OmegaConf.create(model_cfg))
            self.cnn_model = self.cnn_model.to(device)
            self.cnn_model.load_state_dict(
                torch.load(cnn_path)['net']
            )
        else:
            raise NotImplementedError("Undefined extension. select `.pkl` or `.t7`.")
        
        self.normal_embed = nn.Linear(num_classes, embedding_size)
        self.freeze_model()
        
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        
        if base_cnn_type.split(':')[0] == 'resnet':
            self.cnn_model.fc = nn.Linear(self.cnn_model.fc.in_features, self.embedding_size)
        elif base_cnn_type.split(':')[0] == 'vgg':
            self.cnn_model.classifier = nn.Linear(25088, self.embedding_size)
        else:
            raise NotImplementedError("Undefined base cnn model type. select `resnet` or `vgg`.")


        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(2 * self.embedding_size, hidden_size_ch),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size_ch, hidden_size_ch),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size_ch, 2)
        )

    def forward(self, image, label):
        """
        image           : [batch_size, 3, 224, 224]
        label           : [batch_size, 1]
        """
        label = torch.nn.functional.one_hot(torch.squeeze(label), num_classes=self.num_classes).float()
        embedded_label = self.normal_embed(label)
        embedded_image = self.cnn_model(image)  # [batch_size, embedding_size]
        embedding = torch.cat((embedded_label, embedded_image), dim=1)  # [batch_size, 2 * embedding_size]

        output = self.classification_head(embedding)  # [batch_size, d_out]

        return output
    
    def freeze_model(self):
        for param in self.cnn_model.parameters():
            param.requires_grad = False
            
        for parm in self.normal_embed.parameters():
            param.requires_grad = False



def compose_net(model_cfg: OmegaConf):
    """Automatically compose PyTorch Model from configs.

    Args:
        model_cfg (OmegaConf): model configs.
        
    Returns:
        net: PyTorch Model.
    """
    NUM_CLASSES = 52
    # NUM_CLASSES = model_cfg.num_classes
    
    # torch.hub._validate_not_a_forked_repo=lambda a,b,c: True  # <- enable it when running on docker.
    MODEL_LIST = [
        torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=False),
        torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=False),
        torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False),
        torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False),
    ]
    
    net = MODEL_LIST[model_cfg.id]
    if model_cfg.id >= 0 and model_cfg.id <= 1:
        net.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, NUM_CLASSES),
        )
    elif model_cfg.id >= 2 and model_cfg.id <= 6:
        net.fc = torch.nn.Linear(net.fc.in_features, NUM_CLASSES)
    print(net)
    
    return net