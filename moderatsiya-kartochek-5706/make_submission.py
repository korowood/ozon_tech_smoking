import torch
import torchvision.transforms as transforms
import os
from PIL import Image
# from resnet18 import init_model
import torchvision
from kan import KAN
from torch import nn

MODEL_WEIGHTS = "baseline_resnet50.pt"
TEST_IMAGES_DIR = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveConcatPool2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        return torch.cat([self.avgpool(x), self.maxpool(x)], dim=1)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = init_model(device, 2)
    model = torchvision.models.resnet50(pretrained=False)
    model.avgpool = AdaptiveConcatPool2d()
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        KAN([model.fc.in_features * 2, 64, 2]))
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_image_names = os.listdir(TEST_IMAGES_DIR)
    all_preds = []

    for image_name in all_image_names:
        img_path = os.path.join(TEST_IMAGES_DIR, image_name)
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.sigmoid(output[:, 1]).item() >= 0.5
            all_preds.append(int(pred))

    with open(SUBMISSION_PATH, "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_image_names, all_preds):
            f.write(f"{name}\t{cl_id}\n")
