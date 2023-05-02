import os, time
from tqdm import tqdm
import datetime
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
from model import MCSFNet
from metrics import DiceLoss, DiceBCELoss

def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,"images", name) + ".png" for name in data]
    masks = [os.path.join(path,"masks", name) + ".png" for name in data]
    return images, masks


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0

        return image, mask

    def __len__(self):
        return self.n_samples


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    time_taken = []
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(loader)):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            start_time = time.time()
            y_pred, _ = model(x)
            y_pred = torch.sigmoid(y_pred)
            end_time = time.time() - start_time
            time_taken.append(end_time)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss/len(loader)
        epoch_jac = epoch_jac/len(loader)
        epoch_f1 = epoch_f1/len(loader)
        epoch_recall = epoch_recall/len(loader)
        epoch_precision = epoch_precision/len(loader)
        mean_time_taken = np.mean(time_taken)
        mean_fps = 1 / mean_time_taken
        print("Mean FPS: ", mean_fps)
        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def load_data(path, i):
    valid_names_path = f"{path}/" + i + "_test.txt"
    valid_x, valid_y = load_names(path, valid_names_path)

    return  (valid_x, valid_y)

if __name__ == "__main__":
    # """ Training logfile """
    test_log_path = "files/net.txt"
    if os.path.exists(test_log_path):
        print("Log file exists")
    else:
        train_log = open("files/net.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ load_mdoel """
    image_size = 256
    size = (image_size, image_size)
    device = torch.device('cuda')

    # 存放模型的地址
    model_path = "checkpoint/"
    cate = [model_path + x for x in os.listdir(model_path)]
    for folder in cate:
        model = MCSFNet()
        model = model.to(device)
        model.load_state_dict(torch.load(folder))
        filename = ["kvasir", "clinic", "etis", "colon", "cvc300"]
        for i in filename:
            path = "../data/test/"+i
            #
            """ Dataset """
            (valid_x, valid_y) = load_data(path, i)
            data_str = f"model:{folder}---Dataset:{i} --- Valid: {len(valid_x)}"
            print_and_save(test_log_path, data_str)
            valid_dataset = DATASET(valid_x, valid_y, size, transform=None)
            valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )
            loss_fn = DiceBCELoss()

            valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
            data_str = f"Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1-score(Dice): {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
            print_and_save(test_log_path, data_str)
        print('='*40)
