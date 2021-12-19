import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from models.simple_model import FaceProcess
from torch.utils.data import DataLoader

import torch.nn.functional as F
from utils import save_model, get_mean_dist, load_model
from datasets.lfw_reader import TripletLossLFWDataset
from torchvision import transforms
from config import BATCH_SIZE, ITERATION_COUNT, LR, DATA_PATH, MODEL_NAME, PRETRAINED, MARGIN, EMBEDDING_SIZE
from torch.nn.modules.distance import PairwiseDistance
from torchvision.datasets import LFWPairs
from test import test
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# TODO: CosineSimilarity Loss
# TODO: Select triplets: Semi-Hard Negative triplet and Hard Negative triplet

def make_plot(arr, label):

    from matplotlib import pyplot as plt

    plt.clf()
    plt.plot(arr)
    plt.ylabel(label)
    plt.xlabel("Batches")
    plt.savefig(f"{label}.png")

losses = []
def train(epoch,
          model,
          loss_fn,
          optimizer,
          dataloader,
          margin,
          use_semihard_negatives=False):
    global losses

    # l2_distance = PairwiseDistance(p=2)
    model.train()
    for i, (anchor_img, positive_img, negative_img) in enumerate(dataloader):

        anchor_img = anchor_img.to(DEVICE)
        positive_img = positive_img.to(DEVICE)
        negative_img = negative_img.to(DEVICE)

        anchor_embeddings, positive_embeddings, negative_embeddings = \
            model(anchor_img, positive_img, negative_img)

        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        if (i + 1) % 20 == 0:
            print("Epoch {}, Step [{}/{}], Loss {:.4f}".format(epoch, i, len(dataloader), float(loss)))


def test_different_thresholds(model, dataloader):
    thresholds = np.arange(0.05, 1.5, 0.15)
    results = []

    for threshold in tqdm(thresholds):
        res = test(threshold, model, dataloader)
        results.append(res)

    print("Results:", results)

def main():
    model = FaceProcess(MODEL_NAME, EMBEDDING_SIZE, PRETRAINED).to(DEVICE)

    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = TripletLossLFWDataset(DATA_PATH,
                                       "train",
                                       "funneled",
                                       download=True,
                                       samples_num=3000,
                                       transform=train_transform)

    test_data = LFWPairs(DATA_PATH, "test", "funneled", download=True,
                         transform=test_transform)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    loss = nn.TripletMarginLoss(margin=MARGIN)
    # loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(), margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, ITERATION_COUNT + 1):
        print(f"Epoch {epoch}")
        train(epoch, model, loss, optimizer, train_dataloader, MARGIN, use_semihard_negatives=False)
        # if epoch % 5 == 0:
        test_different_thresholds(model, test_dataloader)
        if epoch % 1 == 0:
            save_model(model, epoch, f"saves/{MODEL_NAME}_{str(PRETRAINED)}_cos")

    make_plot(losses, "Loss")

if __name__ == '__main__':
    main()


