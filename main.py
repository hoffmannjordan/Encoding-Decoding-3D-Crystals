from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import numpy as np
from torch.utils.data.dataset import Dataset
from pymatgen.core.structure import Structure
from dataset import *
from CrystalVAE import *
from Discriminator import *
from Segmentation import *



def loss_MSE(recon_x, x, mu, logvar,epoch):
    BCE = F.mse_loss(recon_x, x, size_average=True)
    weight = 5.0
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + weight*KLD, BCE, weight*KLD

def loss_BCE(recon_x, x, mu, logvar,epoch):
    weight = 0.0
    BCE = F.binary_cross_entropy(recon_x, x, size_average=True)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + weight*KLD, BCE, weight*KLD


def train(args, model, device, train_loader, optimizer, epoch ,UNet, optimizer2,):
        model.train()
        for batch_idx, (data,  target,label, abcangles, species_mat) in enumerate(train_loader):

                data, target = data.to(device), target.to(device)
                species_mat = species_mat.to(device)

                optimizer.zero_grad()

                optimizer2.zero_grad()

                reconstruction, mu, logvar = model(data)
                
                UNetRecon = UNet(reconstruction)
                loss1, bce, kld = loss_MSE(reconstruction, target, mu, logvar,epoch)
                lossBCE, bce2, kld2 = loss_BCE(UNetRecon, species_mat, mu, logvar,epoch)
                loss = loss1  + 0.1*lossBCE     
                loss.backward(retain_graph=True)
                lossBCE.backward()
                optimizer2.step()
                optimizer.step()
                


def test(args, model, device, test_loader, epoch,unet):

        model.eval()
        unet.eval()
        test_loss = 0
        SAVE_UP_TO = 6
        with torch.no_grad():
                i = 0
                counter = 0
                for data, target,  label , abcangles, species_mat in test_loader:
                        # Send the data to the device (GPU)
                        data, target = data.to(device), target.to(device)
                        species_mat = species_mat.to(device)
                        # Compute reconstruction and segmentation
                        reconstruction, mu, logvar  = model(data)
                        species_pred = unet(reconstruction)
                        # Compute recontsruction loss
                        loss , bce , kld  = loss_MSE(reconstruction, target, mu, logvar,1)
                        loss2, bce2, kld2 = loss_BCE(species_pred, species_mat, mu, logvar,epoch)

                        species_pred = species_pred.cpu().numpy()
                        label = label.cpu().numpy()
                        outputNP3 = reconstruction.cpu().numpy()
                        targetNP3 = target.cpu().numpy()
                        species_mat = species_mat.cpu().numpy()
                        

                        if i < SAVE_UP_TO:
                            for ii in range(0,18):
                                PREDICTION = outputNP3[ii]
                                TARGET     = targetNP3[ii]
                                mat1 = PREDICTION.flatten()
                                np.savetxt('/SAVE/DIRECTORY/ElectronDensity_Pred_'+str(counter)+'_'+str(epoch)+'.csv',mat1)
                                mat1 = TARGET.flatten()
                                np.savetxt('/SAVE/DIRECTORY/ElectronDensity_True_'+str(counter)+'_'+str(epoch)+'.csv',mat1)
                                species_mat_True = species_mat[ii].argmax(axis=0)
                                species_mat_Pred = species_pred[ii].argmax(axis=0)
                                np.savetxt('/SAVE/DIRECTORY/Species_True_'+str(counter)+'_'+str(epoch)+'.csv',species_mat_True.flatten().astype(int),fmt='%i')
                                np.savetxt('/SAVE/DIRECTORY/Species_Pred_'+str(counter)+'_'+str(epoch)+'.csv',species_mat_Pred.flatten().astype(int),fmt='%i')
                                counter += 1

                        i += 1
        print("Test loss is: ",loss)



def main():

        parser = argparse.ArgumentParser(description='Repeating Lattice VAE (ReLa)')
        parser.add_argument('--epochs', type=int, default=50, metavar='N',
                                                help='number of epochs to train (default: 50)')
        parser.add_argument('--lr', type=float, default=0.0000001, metavar='LR',
                                                help='learning rate (default: 0.0000001)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                                help='SGD momentum (default: 0.5)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                                                help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                                                help='random seed (default: 1)')

        args = parser.parse_args()
        use_cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        # We used 32 pickle files to load the data.
        CrystalDataset_Train   = CrystalDataset(1,32)
        train_loader = torch.utils.data.DataLoader(CrystalDataset_Train, batch_size=18,
                                                shuffle=True, num_workers=0)
        CrystalDataset_Test = CrystalDataset(0,1)
        test_loader = torch.utils.data.DataLoader(CrystalDataset_Test, batch_size=18,
                                                shuffle=False, num_workers=0)

        model = CVAE().to(device)
        UNet = AttU_Net3D().to(device)

        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer2 = optim.SGD(UNet.parameters(), lr=args.lr, momentum=args.momentum)
        
        for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch,UNet,optimizer2)
                test(args, model, device, test_loader, epoch,UNet)
                if False:
                    if epoch%2==0:
                           torch.save(model.state_dict(),"ReLaDS_"+str(epoch)+".pt")
                           torch.save(UNet.state_dict(), "ReLaDS_U_"+str(epoch)+".pt")

if __name__ == '__main__':
    main()


