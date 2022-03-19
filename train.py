import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from models.PoseNet import PoseNet, PoseLoss
from data.DataSource import *
import os
import matplotlib.pyplot as plt
from optparse import OptionParser


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(epochs, batch_size, learning_rate, save_freq, data_dir):
    # train dataset and train loader
    datasource = DataSource(data_dir, train=True)
    train_loader = Data.DataLoader(dataset=datasource, batch_size=batch_size, shuffle=True)

    # load model
    posenet = PoseNet().to(device)

    # loss function
    criterion = PoseLoss(0.3, 0.3, 1., 300, 300, 300)

    # train the network
    optimizer = torch.optim.Adam(nn.ParameterList(posenet.parameters()),
                     lr=learning_rate, eps=1,
                     weight_decay=0.0625,
                     betas=(0.9, 0.999))

    batches_per_epoch = len(train_loader.batch_sampler)
    train_losses = []
    for epoch in range(epochs):
        print("Starting epoch {}:".format(epoch))
        epoch_loss = 0
        posenet.train()
        for step, (images, poses) in enumerate(train_loader):
            b_images = Variable(images, requires_grad=True).to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(poses), requires_grad=True).to(device)

            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("{}/{}: loss = {}".format(step+1, batches_per_epoch, loss))
        train_losses.append(epoch_loss / datasource.__len__())
        # Save state
        if epoch % save_freq == 0:
            save_filename = 'epoch_{}.pth'.format(str(epoch+1).zfill(5))
            save_path = os.path.join('checkpoints', save_filename)
            torch.save(posenet.state_dict(), save_path)
    np.save('train_losses.npy',train_losses)
    plt.figure(1,figsize=(10,10))
    plt.xticks(np.arange(len(train_losses)),np.arange(1,len(train_losses)+1))
    plt.plot(train_losses,'b',label='Training data')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('Train_Losses.png',dpi=400)
    plt.show()

def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', default=200, type='int')
    parser.add_option('--learning_rate', default=0.0001, type='float')
    parser.add_option('--batch_size', default=75, type='int')
    parser.add_option('--save_freq', default=20, type='int')
    parser.add_option('--data_dir', default='data/datasets/KingsCollege/')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    main(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, save_freq=args.save_freq, data_dir=args.data_dir)
