import argparse
import random
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats
from dataset.mydata import Traindata,Testdata
from model.AVQA import AVQA
from torch.optim import lr_scheduler

# ------------settings------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(filename)s %(levelname)s %(message)s",
                    datefmt="%a %d %b %Y %H:%M:%S",filename='./log/avqa.log')

# ------------args--------------
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size',type=int, default=16)
parser.add_argument('--data_path',type=str, default='./data')
args = parser.parse_args()

# ------------dataset------------
data_dir = args.data_path
train_file = './datalist/train.txt'
test_file = './datalist/test.txt'
train_dataset = Traindata(data_dir, train_file)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = Testdata(data_dir, test_file)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

# ------------setting------------
avqa = AVQA().to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(avqa.parameters(), lr=args.learning_rate,weight_decay=1e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer,[50,100,200,350],gamma=0.5)
train_loss = []
test_loss = []
train_epochs_loss = []
test_epochs_loss = []


def train():
    for epoch in range(args.epochs):
        # ------------train------------
        avqa.train()
        train_epoch_loss = []
        for idx, data in enumerate(train_dataloader, 0):
            audio = data['audio'].float().to(device)
            video = data['video'].float().to(device)
            score = data['score'].float().to(device)
            outputs = avqa(video, audio)
            optimizer.zero_grad()
            loss = loss_fn(score, outputs)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            logging.info('[train]epoch={}/{},{}/{} of train, loss={}'.format(
                epoch+1, args.epochs, idx+1, len(train_dataloader), loss.item()))
        scheduler.step()
        train_epochs_loss.append(np.average(train_epoch_loss))
        torch.cuda.empty_cache()
        # ------------vali------------
        avqa.eval()
        test_epoch_loss = []
        result_list = []
        score_list = []
        with torch.no_grad():
            for idx, data in enumerate(test_dataloader, 0):
                audio = data['audio'].to(device)
                video = data['video'].to(device)
                score = data['score'].to(device)
                outputs = avqa(video,audio)
                loss = loss_fn(outputs, score)
                result_list.append(outputs)
                score_list.append(score)
                test_epoch_loss.append(loss.item())
                test_loss.append(loss.item())
            pred = torch.stack(result_list).view(-1)
            gt = torch.stack(score_list).view(-1)
            plcc = stats.pearsonr(gt.cpu().numpy(), pred.cpu().numpy())[0]
            srocc = stats.spearmanr(gt.cpu().numpy(), pred.cpu().numpy())[0]
            average_loss = np.average(test_epoch_loss)
            test_epochs_loss.append(average_loss)
            logging.info('[valid]epoch={}/{},test,loss={},plcc={},srocc={}'.format(epoch+1,args.epochs,average_loss,plcc,srocc))
            torch.cuda.empty_cache()


def plot():
    # ------------plot------------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
    plt.plot(test_epochs_loss[1:], '-o', label="test_loss")
    plt.title("epochs_loss")
    plt.legend()
    f = plt.gcf()  # 获取当前图像
    f.savefig('./log/avqa.png')
    f.clear()


if __name__ == "__main__":
    train()
    plot()
