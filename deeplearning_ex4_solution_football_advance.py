import cv2
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

def collate_fn(batch):
    items = list(zip(*batch))
    images, labels = items
    final_images = torch.cat(images, dim=0)
    final_labels = []
    for l in labels:
        final_labels.extend(l)
    return final_images, torch.FloatTensor(final_labels)

class FootballDataset(Dataset):
    def __init__(self, root, transform=None):
        self.matches = os.listdir(root)
        self.match_files = [os.path.join(root, match_file) for match_file in self.matches]
        self.transform = transform

        # Use image id in json file to count
        self.from_id = 0
        self.to_id = 0

        # Use image_id to select video
        self.video_select = {}

        # Count total frame in all video
        for path in self.match_files:
            # Extract json file
            json_dir, video_dir = os.listdir(path)
            json_dir, video_dir = os.path.join(path, json_dir), os.path.join(path, video_dir)
            with open(json_dir, "r") as json_file:
                json_data = json.load(json_file)

            self.to_id += len(json_data["images"])
            self.video_select[path] = [self.from_id + 1, self.to_id]
            self.from_id = self.to_id

    def __len__(self):
        return self.to_id

    def __getitem__(self, idx):
        # Choose real index and video of this frame
        for key, value in self.video_select.items():
            if value[0] <= idx + 1 <= value[1]:
                idx = idx % value[0]
                select_path = key

        # Load file
        json_dir, video_dir = os.listdir(select_path)
        json_dir, video_dir = os.path.join(select_path, json_dir), os.path.join(select_path, video_dir)
        json_file = open(json_dir, "r")
        annotations = json.load(json_file)["annotations"]

        # Real frame
        cap = cv2.VideoCapture(video_dir)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        flag, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Take annotations
        annotations = [anno for anno in annotations if anno["image_id"] == idx + 1 and anno["category_id"] == 4]
        box = [annotation["bbox"] for annotation in annotations]
        cropped_images = [frame[int(y):int(y + h), int(x):int(x + w)] for [x, y, w, h] in box]
        if self.transform:
            cropped_images = torch.stack([self.transform(image) for image in cropped_images])

        # Take number of players
        jerseys = [int(annotation["attributes"]["jersey_number"]) for annotation in annotations]

        return cropped_images, jerseys


if __name__ == "__main__":
    index = 800
    batch_size = 8
    transform = Compose([
        ToPILImage(),
        Resize((112, 224)),
        ToTensor(),
    ])
    dataset = FootballDataset("../data/football", transform=transform)
    cropped_images, jerseys = dataset.__getitem__(index)
    train_params = {"batch_size": batch_size,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": 4,
                    "collate_fn": collate_fn}
    data_loader = DataLoader(dataset, **train_params)
    for images, labels in data_loader:
        print(images.shape)
        print(labels.shape)
        print("---------")
