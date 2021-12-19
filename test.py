import os

import torch
from torchvision import transforms
import numpy as np


from torch.utils.data import DataLoader
from utils import load_model, get_mean_dist
from datasets.lfw_reader import TripletLossLFWDataset
from models.simple_model import FaceProcess
from config import DATA_PATH, MODEL_NAME, EMBEDDING_SIZE
from tqdm import tqdm
from torchvision.datasets import LFWPairs

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

THRESHOLD = 9.5

def test(threshold, model, dataloader):
    model.eval()
    with torch.no_grad():
        binary_score = 0
        count = 0

        targets = []
        predicts = []

        for img1, img2, target in dataloader:
            b, _, _, _ = img1.size()
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)

            x = torch.cat([img1, img2])
            vectors = model.predict(x)
            vec1, vec2 = torch.split(vectors, (b, b))

            vec1_numpy = vec1.cpu().detach().numpy()
            vec2_numpy = vec2.cpu().detach().numpy()

            dist = np.linalg.norm(vec2_numpy - vec1_numpy, axis=1)

            predicts += (dist < threshold).astype(np.int32).tolist()
            targets += target.numpy().tolist()

        binary_score = np.sum(np.array(predicts) == np.array(targets))
        binary_score = 100 * binary_score / len(predicts)

    return binary_score



if __name__ == '__main__':
    model = FaceProcess(MODEL_NAME, EMBEDDING_SIZE).to(DEVICE)

    data_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_data = LFWPairs(DATA_PATH, "test", "funneled", download=True,
                                      transform=data_transform)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    for name in os.listdir(r"pretrained_model"):
        results = []
        print(name)

        load_model(model, os.path.join(r"pretrained_model", name))
        model.eval()

        for threshold in np.arange(0.1, 4.5, 0.1):
            res = test(threshold, model, test_dataloader)
            results.append(res)
        print(results)

    print("Thresholds {}".format(results))