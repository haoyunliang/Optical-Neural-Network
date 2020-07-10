import os
import csv
import random
import pathlib
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import onn


os.environ["CUDA_VISIBLE_DEVICES"] = '8'


def main(args):

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    # transform = transforms.Compose([transforms.Resize(size=(200, 200)), transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
    val_dataset = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=112, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=112, shuffle=False, pin_memory=True)

    model = onn.Net()
    model.cuda()

    if args.whether_load_model:
        model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) + args.model_name))
        print('Model : "' + args.model_save_path + str(args.start_epoch) + args.model_name + '" loaded.')
    else:
        if os.path.exists(args.result_record_path):
            os.remove(args.result_record_path)
        else:
            with open(args.result_record_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ['Epoch', 'Train_Loss', "Train_Acc", 'Val_Loss', "Val_Acc", "LR"])

    criterion = torch.nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.num_epochs):

        log = [epoch]

        model.train()

        train_len = 0.0
        train_running_counter = 0.0
        train_running_loss = 0.0

        tk0 = tqdm(train_dataloader, ncols=100, total=int(len(train_dataloader)))
        for train_iter, train_data_batch in enumerate(tk0):

            train_images = train_data_batch[0].cuda()           # (64, 1, 200, 200) float32 1. 0.
            train_labels = train_data_batch[1].cuda()           # (1024, 10) int64 9 0
            train_images = F.pad(train_images, pad=(86, 86, 86, 86))

            train_labels = F.one_hot(train_labels, num_classes=10).float()

            train_images = torch.squeeze(torch.cat((train_images.unsqueeze(-1),
                                                    torch.zeros_like(train_images.unsqueeze(-1))), dim=-1), dim=1)

            train_outputs = model(train_images)

            train_loss_ = criterion(train_outputs, train_labels)
            train_counter_ = torch.eq(torch.argmax(train_labels, dim=1), torch.argmax(train_outputs, dim=1)).float().sum()

            optimizer.zero_grad()
            train_loss_.backward()
            optimizer.step()

            train_len += len(train_labels)
            train_running_loss += train_loss_.item()
            train_running_counter += train_counter_

            train_loss = train_running_loss / train_len
            train_accuracy = train_running_counter / train_len

            tk0.set_description_str('Epoch {}/{} : Training'.format(epoch, args.start_epoch + 1 + args.num_epochs - 1))
            tk0.set_postfix({'Train_Loss': '{:.5f}'.format(train_loss), 'Train_Accuracy': '{:.5f}'.format(train_accuracy)})

        log.append(train_loss)
        log.append(train_accuracy)

        with torch.no_grad():
            # 验证
            model.eval()

            val_len = 0.0
            val_running_counter = 0.0
            val_running_loss = 0.0

            tk1 = tqdm(val_dataloader, ncols=100, total=int(len(val_dataloader)))
            for val_iter, val_data_batch in enumerate(tk1):

                val_images = val_data_batch[0].cuda()  # (64, 1, 200, 200) float32 1. 0.
                val_labels = val_data_batch[1].cuda()  # (1024, 10) int64 9 0
                val_images = F.pad(val_images, pad=(86, 86, 86, 86))
                val_labels = F.one_hot(val_labels, num_classes=10).float()

                val_images = torch.squeeze(torch.cat((val_images.unsqueeze(-1),
                                                        torch.zeros_like(val_images.unsqueeze(-1))), dim=-1), dim=1)

                val_outputs = model(val_images)

                val_loss_ = criterion(val_outputs, val_labels)
                val_counter_ = torch.eq(torch.argmax(val_labels, dim=1), torch.argmax(val_outputs, dim=1)).float().sum()

                val_len += len(val_labels)
                val_running_loss += val_loss_.item()
                val_running_counter += val_counter_

                val_loss = val_running_loss / val_len
                val_accuracy = val_running_counter / val_len

                tk1.set_description_str('Epoch {}/{} : Validating'.format(epoch, args.start_epoch + 1 + args.num_epochs - 1))
                tk1.set_postfix({'Val_Loss': '{:.5f}'.format(val_loss), 'Val_Accuarcy': '{:.5f}'.format(val_accuracy)})

            log.append(val_loss)
            log.append(val_accuracy)

        torch.save(model.state_dict(), (args.model_save_path + str(epoch) + args.model_name))
        print('Model : "' + args.model_save_path + str(epoch) + args.model_name + '" saved.')

        with open(args.result_record_path, 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--whether-load-model', type=bool, default=False, help="是否加载模型继续训练")
    parser.add_argument('--start-epoch', type=int, default=0, help='从哪个epoch继续训练')
    # 数据和模型相关
    parser.add_argument('--model-name', type=str, default='_model.pth')
    parser.add_argument('--model-save-path', type=str, default="./saved_model/")
    parser.add_argument('--result-record-path', type=pathlib.Path, default="./result.csv", help="数值结果记录路径")

    torch.backends.cudnn.benchmark = True
    args_ = parser.parse_args()
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    main(args_)