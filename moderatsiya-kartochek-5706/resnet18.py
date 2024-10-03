import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from kan import KAN
import numpy as np
from sklearn.metrics import precision_score, recall_score
import time
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings("ignore")


def seed_everything(seed=7575):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.random.manual_seed(seed)


def set_requires_grad(model, value=False):
    for param in model.parameters():
        param.requires_grad = value


def compute_metric(pred, gt):
    score = f1_score(gt, pred, average='binary')
    return score


def train_model(model, dataloaders, criterion, optimizer,
                phases, num_epochs=3):
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=4, verbose=True, eps=1e-6)
    #    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)
    start_time = time.time()

    acc_history = {k: list() for k in phases}
    loss_history = {k: list() for k in phases}
    f1_history = {k: list() for k in phases}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_f1 = 0
            all_labels = []
            all_preds = []
            # Iterate over data.
            n_batches = len(dataloaders[phase])
            for inputs, labels in tqdm(dataloaders[phase], total=n_batches):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    probas = torch.sigmoid(outputs[:, 1]) >= 0.5
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(probas.cpu().numpy())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_f1 += (compute_metric(preds.cpu().numpy(), labels.data.cpu().numpy()))
                # print(probas.cpu().numpy())
                # print(labels.data.cpu().numpy())
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()
            epoch_acc /= len(dataloaders[phase].dataset)

            epoch_f1 = running_f1
            epoch_f1 /= len(dataloaders[phase].dataset)
            precision = precision_score(all_labels, all_preds, average='binary')
            recall = recall_score(all_labels, all_preds, average='binary')
            f1 = f1_score(all_labels, all_preds, average='binary')

            scheduler.step(epoch_loss)
            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss,
                                                                  epoch_acc, epoch_f1))

            print(f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')
            print(optimizer.param_groups[0]["lr"])
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))

    return model, acc_history


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveConcatPool2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)


def init_model(device, num_classes):
    model = torchvision.models.resnet50(pretrained=True)
    set_requires_grad(model, False)
    model.avgpool = AdaptiveConcatPool2d()
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        KAN([model.fc.in_features*2, 64, num_classes]))
    model = model.to(device)
    return model


# hardcode
MODEL_WEIGHTS = "./baseline_resnet50.pt"
# TRAIN_DATASET = "../data/train/"
# TRAIN_CSV = "../data/private_info/train.csv"
TRAIN_DIR = './data/train/'
VAL_DIR = './data/val/'


class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {'other': 0, 'smoking': 1}
        self.image_paths = []
        self.labels = []

        for cls_name in os.listdir(root_dir):
            cls_folder = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if cls_name == 'other':
                            label = self.classes['other']
                        else:
                            label = self.classes['smoking']

                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(train_dir, val_dir, batch_size=32):
    img_size = 224
    # transform = transforms.Compose([
    #     transforms.Resize((img_size, img_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor(),
    #     # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if len(x.shape) == 2 or x.shape[-1] == 1 else x), # if Gray => make 3 channels
    #     # transforms.Lambda(lambda x: x[:4, :, :]), # if 4 channels => make 3
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomGrayscale(p=0.1),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    train_dataset = BaseDataset(root_dir=train_dir, transform=transform)
    val_dataset = BaseDataset(root_dir=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.classes


if __name__ == "__main__":
    seed_everything()

    train_loader, val_loader, classes = get_data_loaders(TRAIN_DIR, VAL_DIR)

    batch_size = 32
    num_workers = 8
    PRE_EPOCH = 2
    EPOCH = 11
    learning_rate = .001

    loaders = {'train': train_loader, 'val': val_loader}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = init_model(device, num_classes=2)

    #    pretrain_optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate, amsgrad=False)
    # pretrain_optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=1e-2)
    pretrain_optimizer = torch.optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)

    #    train_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    # train_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    # Pretrain
    # запустить предобучение модели на две эпохи
    pretrain_results = train_model(model, loaders, criterion, pretrain_optimizer,
                                   phases=['train', 'val'], num_epochs=PRE_EPOCH)

    # Train
    # запустить дообучение модели
    set_requires_grad(model, True)
    train_results = train_model(model, loaders, criterion, train_optimizer,
                                phases=['train', 'val'], num_epochs=EPOCH)

    torch.save(model.state_dict(), MODEL_WEIGHTS)
