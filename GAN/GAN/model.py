import torch
import torch.nn as nn
import timm

class Encoder(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        backbone = timm.create_model(backbone_name, pretrained=True)
        backbone.head = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embed_dim = self.backbone.num_features

    def forward(self, x):
        x = self.backbone(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=512, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if isinstance(drop, tuple):
            drop_probs = drop
        else:
            drop_probs = (drop, drop)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DetModel(nn.Module):
    def __init__(self, backbone_name='vit_small_patch16_224', num_objects=5):
        super().__init__()
        self.backbone = Encoder(backbone_name)
        self.num_objects = num_objects
        # out_dim = num_objects * (4 + 1)
        # self.head = Mlp(self.backbone.embed_dim, out_features=out_dim)

        bbox_dim = num_objects * 4
        conf_dim = num_objects
        # Separate heads for bounding boxes and confidence scores
        self.bbox_head = Mlp(self.backbone.embed_dim, out_features=bbox_dim)
        self.conf_head = Mlp(self.backbone.embed_dim, out_features=conf_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        # x = self.head(x)
        # x = x.reshape(B,self.num_objects,-1)

        bbox = torch.sigmoid(self.bbox_head(x)).reshape(B, self.num_objects, 4)
        conf = torch.sigmoid(self.conf_head(x)).reshape(B, self.num_objects, 1)
        
        return torch.cat([bbox, conf], dim=-1)
    

class DetLineModel(nn.Module):
    def __init__(self, backbone_name='vit_small_patch16_224', out_dim=4):
        super().__init__()
        self.backbone = Encoder(backbone_name)
        self.head = Mlp(self.backbone.embed_dim, out_features=out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x = self.head(x)
        x = self.sigmoid(x)
        x = x.reshape(B, -1)
        
        return x

if __name__=='__main__':
    model = DetLineModel('vit_base_patch16_224')
    dummy_input = torch.rand((2,3,224,224))
    out = model(dummy_input)
    print(out)
    print(out.shape)