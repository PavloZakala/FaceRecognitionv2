import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
from torchvision.datasets import LFWPeople, LFWPairs


class TripletLossLFWDataset(LFWPeople):

    def _get_triplets(self):
        data, targets = [], []
        with open(os.path.join(self.root, self.labels_file), 'r') as f:
            lines = f.readlines()
            n_folds, s = (int(lines[0]), 1) if self.split == "10fold" else (1, 0)

            for fold in range(n_folds):
                n_lines = int(lines[s])
                people = [line.strip().split("\t") for line in lines[s + 1: s + n_lines + 1]]
                people_with_more_than_one_images = [(identity, num_imgs) for identity, num_imgs in people if
                                                    int(num_imgs) > 1]
                s += n_lines + 1

                for _ in range(self.samples_num):
                    current_identity, current_num_imgs = self.rng.choice(people_with_more_than_one_images)

                    negative_identity = current_identity
                    while negative_identity == current_identity:
                        negative_identity, _ = self.rng.choice(people)
                    negative_idx = 1

                    current_img_indexes = np.arange(1, int(current_num_imgs) + 1)
                    self.rng.shuffle(current_img_indexes)
                    anchor_idx, positive_idx = current_img_indexes[:2]

                    anchor_img = self._get_path(current_identity, anchor_idx)
                    positive_img = self._get_path(current_identity, positive_idx)
                    negative_img = self._get_path(negative_identity, negative_idx)

                    data.append((anchor_img, positive_img, negative_img))

        return data, targets

    def __init__(self,
                 root: str,
                 split: str = "train",
                 image_set: str = "funneled",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 samples_num=3000,
                 download: bool = False,
                 random_seed=0):
        super(TripletLossLFWDataset, self).__init__(root, split, image_set, transform, target_transform,
                                                    download)

        self.samples_num = samples_num
        self.rng = np.random.default_rng(random_seed)
        self.data, self.targets = self._get_triplets()

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:

        anchor_img = self._loader(self.data[index][0])
        positive_img = self._loader(self.data[index][1])
        negative_img = self._loader(self.data[index][2])

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img


if __name__ == '__main__':
    import cv2
    from torchvision import transforms

    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    # dataset = LFWPeople("data", "train", "funneled", download=True)
    dataset = LFWPairs("data", "test", "funneled", download=True)
    # dataset = TripletLossLFWDataset("data",
    #                                    "train",
    #                                    "funneled",
    #                                    download=True,
    #                                    samples_num=3000,
    #                                    transform=data_transform)
    #
    # for i in range(len(dataset)):
    #     anchor, positive, negative = dataset[i]
    #
    #     faces = np.concatenate([anchor, positive, negative], axis=1)
    #     cv2.imshow("image", faces)
    #     cv2.waitKey()
