import torch
from torch import nn
from torchvision.models import resnet18, resnet34

torch_models = {
    "resnet18": resnet18,
    "resnet32": resnet34,
}


class FaceProcess(nn.Module):

    def __init__(self, model_name, out_dim=128, pretrained=False):

        super(FaceProcess, self).__init__()

        self.model = torch_models[model_name](pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_dim)

    def init_weights(self):
        self.model.fc.weight.data.normal_(0.0, 0.02)
        self.model.fc.bias.data.fill_(0)

    def forward(self, anchor, positive, negative):
        b, _, _, _ = anchor.size()
        x = torch.cat([anchor, positive, negative])
        f_vector = self.model(x)
        anchor_vec, positive_vec, negative_vec = torch.split(f_vector, (b, b, b))
        return anchor_vec, positive_vec, negative_vec

    def predict(self, faces):
        f_vector = self.model(faces)

        return f_vector

if __name__ == '__main__':
    pass
