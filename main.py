import torch
import pandas as pd
import yaml
import sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.dataset import ImageDataset
from utils.util import make_csv_file, set_seed
from trainer import Trainer


def main():

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = './config/train_config.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    if 'random_seed' in params:
        set_seed(params['random_seed'])
        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transforms_aug= transforms.Compose([  
            transforms.Resize((params['img_size'], params['img_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        transforms.Resize((params['img_size'], params['img_size'])),
        ])


    train_data = pd.read_csv(params['data']['train_data'])
    val_data = pd.read_csv(params['data']['validation_data'])

    train_dataset = ImageDataset(train_data, params, transform=transform, aug_transform=transforms_aug)
    val_dataset = ImageDataset(val_data, params, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=params['batch_size'], shuffle=False)

    print('The number of training data: ', len(train_dataset))
    print('The number of validation data: ', len(val_dataset))

    trainer = Trainer(params, device)
    if params['mode'] == 'train':
        trainer.train(train_loader, val_loader)
    elif params['mode'] == 'test':
        trainer.validate()
        
if __name__ == '__main__':
    main()