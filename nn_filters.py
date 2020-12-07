import argparse
import os
import sys
import time

import numpy as np
import random
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import utils
from Pastiche import PasticheModel
from vgg import vgg11
from PIL import Image

manual_seed = 8888
train_size = (480,640)
eval_size = (810,1080)
content_dataset_path = "coco-2017"
style_dataset_path = "filter_images"
log_interval = 50
subset_size = 5000

Mean = [0.5, 0.5, 0.5]
Std = [0.2, 0.2, 0.2]

def batch_norm(batch):
    mean = batch.new_tensor(Mean).view(-1, 1, 1)
    std = batch.new_tensor(Std).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def train(args, device):
    # original degree (degree=1)
    L_c = 10**5  # content loss weight
    L_s = 10**8.5  # style loss weight

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    content_transform = transforms.Compose([
        transforms.Resize(train_size),
        #transforms.CenterCrop(train_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content_dataset = datasets.ImageFolder(content_dataset_path, content_transform)
    index_list = list(range(len(content_dataset)))
    subset_list = random.sample(index_list, subset_size)
    content_dataset = torch.utils.data.Subset(content_dataset, subset_list)
    #print(len(content_dataset))
    style_dataset = [img for img in os.listdir(style_dataset_path)]
    # sort on filter index
    style_dataset = sorted(style_dataset, key=lambda i: i[-5])
    #print(style_dataset)

    train_loader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True)
    num_styles = len(style_dataset)

    PM = PasticheModel(num_styles).to(device)
    optimizer = Adam(PM.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = vgg11().to(device)
    style_transform = transforms.Compose([
        transforms.Resize(train_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style_batch = []

    for i in range(num_styles):
        style = Image.open(style_dataset_path + '/' + style_dataset[i])
        style = style_transform(style)
        style_batch.append(style)

    styles = torch.stack(style_batch).to(device)

    style_features = vgg(batch_norm(styles))
    style_gram = [gram_matrix(i) for i in style_features]
    # degree of the filtering we want to apply
    degree = args.filtering_level

    if degree <=0:
        L_s = 1
    else:
        L_s = L_s * 10 ** min(degree-1, 5)

    for epoch in range(args.epochs):
        PM.train()
        count = 0

        for batch_idx, (x, _) in enumerate(train_loader):

            if len(x) < args.batch_size:
                break

            count += len(x)
            optimizer.zero_grad()

            #style_ids = [i % num_styles for i in range(count - len(x), count)]
            style_ids = []
            for i in range(len(x)):
                id = random.randint(0, num_styles-1)
                style_ids.append(id)

            stylized = PM(x.to(device), style_ids=style_ids)

            stylized = batch_norm(stylized)
            contents = batch_norm(x)

            features_stylized = vgg(stylized.to(device))
            features_contents = vgg(contents.to(device))

            # use the last block to last block to compute high-level content loss
            # content_loss = mse_loss(features_stylized[-1], features_contents[-1])
            # use second to last block to compute high-level content loss
            content_loss = mse_loss(features_stylized[-2], features_contents[-2])
            content_loss = L_c * content_loss

            style_loss = 0

            for ft_y, s_gram in zip(features_stylized, style_gram):
                y_gram = gram_matrix(ft_y)
                style_loss += mse_loss(y_gram, s_gram[style_ids, :, :])
            style_loss = L_s * style_loss

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print("Epoch {}:\t[{}/{}]\tcontent loss: {:.4f}\tstyle loss: {:.4f}\ttotal loss: {:.4f}".format(
                    epoch + 1, batch_idx, len(train_loader),
                    content_loss / (batch_idx + 1),
                    style_loss / (batch_idx + 1),
                    (content_loss + style_loss) / (batch_idx + 1)
                ))

    # save model
    saved_as = '{0}/{1}_filter_level_{2}_epoch{3}.pth'.format(args.save, str(time.ctime()).replace(' ', '_'), args.filtering_level, args.epochs)
    torch.save(PM, saved_as)
    print("\n Model successfully saved as {} ".format(saved_as))

    return PM


def gen_styles(args, device, model):
    content_image = Image.open(os.path.join("images/content_images", args.content_image))
    content_transform = transforms.Compose([
        transforms.Resize(eval_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_images = content_image.expand(len(args.style_id),-1,-1,-1).to(device)
    #print(content_images.shape)
    #content_image = content_image.unsqueeze(0).to(device)
    #print(content_image.shape)

    with torch.no_grad():
        if args.load_model is None or args.load_model == "None" :
            style_model = model
        else:
            style_model = torch.load(args.load_model)
            style_model.eval()
        style_model.to(device)
        output = style_model(content_images, args.style_id).cpu()

    #output = output.squeeze(0)
    for i in range(len(output)):
        save_image(args.output_image + '/' + args.content_image.replace('.jpg', '_') +'filter' + str(args.style_id[i]+1) + '_level'+ str(args.filtering_level) + '.jpg', output[i])


def save_image(filename, image_data):
    img = image_data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(x):
    (b, d, h, w) = x.size()
    features = x.view(b, d, w * h)
    gram = features.bmm(features.transpose(1, 2)) / (d * h * w)
    return gram

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--epochs", type=int, default=2,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="batch size for training")
    parser.add_argument("--gpu", type=int, default=3,
                        help="GPU id. -1 for CPU")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="learning rate")
    parser.add_argument("--save", type=str, default="saved_models",
                        help="path to folder where trained models will be saved.")
    parser.add_argument("--output-image", type=str, default="images/output_images",
                        help="path for saving the output image")
    parser.add_argument("--content-image", type=str, default="hoofer.jpg",
                                 help="name of content image you want to gen_styles")
    parser.add_argument("--load-model", type=str, default=None,
                        help="saved model to be used for stylizing the image if applicable")
    parser.add_argument("--filtering-level", type=float, default=1.0,
                        help="A positive integer for degree of filtering.0 for no filter.")
    parser.add_argument("--style-id", type=list, default=[0,1,2,3,4,5,6,7,8],
                        help="style number id corresponding to the order in training")

    args = parser.parse_args()

    if args.gpu > -1 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
        print("Start running on GPU{}".format(args.gpu))
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Running on CPU only.")

    if args.load_model is None or args.load_model == 'None':
        print("Start training from scratch")
        model = train(args, device)
        model.eval()
        gen_styles(args, device, model)
    else:
        print("Loading pretrained model from {}".format(args.load_model))
        gen_styles(args, device, None)
