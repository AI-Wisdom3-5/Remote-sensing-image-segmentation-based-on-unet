from model import *
from torch.utils.data import DataLoader
from dataset import Mydata
from torch import optim
import time
from torchvision.utils import save_image
import torch

# from unet_test import *
from Unet_better import UNetplusplus
from model import *


save_path = 'save_img'

class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss, self).__init__()

    def forward(self, pred, mask):
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)
        #计算交并集
        overlap = (pred * mask).sum()
        denum = pred.sum() + mask.sum() + 1e-8

        dice = (2*overlap) / denum
        return 1 - dice


def check_loss(load,model):
    loss = 0
    num = 0
    with torch.no_grad():
        for i, (x,y) in enumerate(load):
            inputs = x.to(device,dtype=torch.float32)
            labels = y.to(device,dtype=torch.float32)

            pred_y = model(inputs)
            loss_fc = loss_fun(pred_y, labels)

            loss_fc = loss_fc.detach().cpu()

            _img = inputs[0]
            _labels = labels[0]
            _pred_y = pred_y[0]
            img_test = torch.stack([_img,_labels,_pred_y],dim=0)
            save_image(img_test,f'save_img/unet_test/test{num}.jpg')

            loss += loss_fc
            num +=1
    loss = loss / len(test_loader)
    return loss


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
train_data = Mydata('train','train_labels')
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
test_data = Mydata('test','test_labels')
test_loader=DataLoader(test_data, batch_size=2,shuffle=True)

model = Unet().to(device)
# model = UNetplusplus(num_classes=3).to(device)
# model = Unet_test().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fun = Diceloss()
# loss_fun = smp.losses.DiceLoss(mode='multiclass')

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=10)

best_loss=1
for epoch in range(200):
    loss = 0
    start_time = time.time()
    for i, (x,y) in enumerate(train_loader):
        inputs = x.to(device,dtype=torch.float32)
        labels = y.to(device,dtype=torch.float32)

        pred_y = model(inputs)
        # loss_fc = loss_fun(pred_y, labels)
        loss_fc = loss_fun(pred_y, labels)

        optimizer.zero_grad()
        loss_fc.backward()
        optimizer.step()

        loss_fc = loss_fc.detach().cpu()
        loss += loss_fc


        _img = inputs[0]
        _labels = labels[0]
        _pre = (pred_y[0])
        img = torch.stack([_img,_labels,_pre],dim=0)
        # save_image(img,f'{save_path}/train/train_{epoch}.jpg')
        #unet++保存地址
        save_image(img, f'{save_path}/unet_train/train_{epoch}.jpg')


    loss = loss / len(train_loader)

    val_loss = check_loss(test_loader,model)
    scheduler.step(loss)

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'save_model/test_best_model.pt')
        print(f'save model success >>>>>>val_loss:{best_loss}')
    print(f'epoch:{epoch} time:{time.time()-start_time}train_loss:{loss} val_loss:{val_loss}')
