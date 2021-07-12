import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import platform
import glob
from argparse import ArgumentParser
from test_function import *
###########################################################################################
# parameter
parser = ArgumentParser(description='MRI-DGDN')
parser.add_argument('--net_name', type=str, default='MRI-DGDN', help='name of net')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=11, help='D,11')
parser.add_argument('--growth-rate', type=int, default=32, help='G,32')
parser.add_argument('--num-layers', type=int, default=8, help='C,8')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {10, 20, 30, 40, 50}')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model_MRI', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log_MRI', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='BrainImages_test', help='name of test set')
parser.add_argument('--run_mode', type=str, default='train', help='train、test')
args = parser.parse_args()
#########################################################################################
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###########################################################################################
# parameter
nrtrain = 800   # number of training blocks
batch_size = 1
Training_data_Name = 'Training_BrainImages_256x256_100.mat'
# define save dir
model_dir = "./%s/%s_layer_%d_denselayer_%d_ratio_%d_lr_%f" % (args.model_dir, args.net_name,args.layer_num, args.num_layers, args.cs_ratio, args.learning_rate)
test_dir = os.path.join(args.data_dir, args.test_name)   # test image dir
filepaths = glob.glob(test_dir + '/*.png')
output_file_name = "./%s/%s_layer_%d_denselayer_%d_ratio_%d_lr_%f.txt" % (args.log_dir, args.net_name, args.layer_num, args.num_layers, args.cs_ratio, args.learning_rate)
#########################################################################################
# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, args.cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask_matrix']
mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']
#########################################################################################
class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, mask):
        x_dim_0 = x.shape[0]
        x_dim_1 = x.shape[1]
        x_dim_2 = x.shape[2]
        x_dim_3 = x.shape[3]
        x = x.view(-1, x_dim_2, x_dim_3, 1)
        y = torch.zeros_like(x)
        z = torch.cat([x, y], 3)
        fftz = torch.fft(z, 2)
        z_hat = torch.ifft(fftz * mask, 2)
        x = z_hat[:, :, :, 0:1]
        x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
        return x
##########################################################################################
# Define initialize parametes
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)
###########################################################################
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)
###########################################################################
class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, 1, kernel_size=1) # output 1 channel

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning
###########################################################################
# Define MRI-RDB Block
class BasicBlock(torch.nn.Module):
    def __init__(self,growth_rate, num_layers):
        super(BasicBlock, self).__init__()

        self.Sp = nn.Softplus()
        self.G = growth_rate
        self.C = num_layers

        self.rdb = RDB(1, self.G, self.C)  # local residual learning

    def forward(self, x, fft_forback, PhiTb, mask, lambda_step):
        x = x - self.Sp(lambda_step) * fft_forback(x, mask)
        x = x + self.Sp(lambda_step) * PhiTb

        x_pred = self.rdb(x)  # local residual learning

        return [x_pred]
#####################################################################################################
# Define Deep Geometric Distillation Network
class DGDN(torch.nn.Module):
    def __init__(self, LayerNo, growth_rate, num_layers):
        super(DGDN, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()
        self.G = growth_rate
        self.C = num_layers

        for i in range(LayerNo):
            onelayer.append(BasicBlock(self.G, self.C))

        self.fcs = nn.ModuleList(onelayer)
        self.fcs.apply(initialize_weights)

        # gradient step
        self.w_mu1 = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu1 = nn.Parameter(torch.Tensor([0.1]))

    def forward(self, PhiTb, mask):

        x = PhiTb
        for i in range(self.LayerNo):
            mu1_ = self.w_mu1 * i + self.b_mu1
            [x] = self.fcs[i](x, self.fft_forback, PhiTb, mask, mu1_)
            if i==((self.LayerNo-1)/2):
                # print(i)
                x_mid = x
        x_final = x

        return [x_final,x_mid]
##################################################################################3
# initial test file
result_dir = os.path.join(args.result_dir, args.test_name)
result_dir = result_dir+'_'+args.net_name+'_ratio_'+ str(args.cs_ratio)+'_epoch_'+str(args.end_epoch)+'/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
###################################################################################
# model
model = DGDN(args.layer_num, args.growth_rate, args.num_layers)
model = nn.DataParallel(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
###################################################################################
print_flag = 1   # print parameter number
if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        # print('Layer %d' % num_count)
        # print(para.size())
####################################################################################
class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length
    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()
    def __len__(self):
        return self.len
#####################################################################################
if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)
#######################################################################################
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if args.start_epoch > 0:   # train stop and restart
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, args.start_epoch)))
#########################################################################################
if args.run_mode == 'train':
    # Training loop
    for epoch_i in range(args.start_epoch+1, args.end_epoch+1):
        model = model.train()
        step = 0
        for data in rand_loader:
            step = step+1
            batch_x = data
            batch_x = batch_x.to(device)
            batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])
            PhiTb = FFT_Mask_ForBack()(batch_x, mask)
            [x_output, x_mid] = model(PhiTb, mask)
            # Compute and print loss
            loss_discrepancy = torch.mean(torch.abs(x_output - batch_x))
            loss_discrepancy_mid = torch.mean(torch.abs(x_mid - batch_x))

            loss_all = loss_discrepancy + loss_discrepancy_mid

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # step %100==0
            if step % 100 == 0:
                output_data = "[%02d/%02d] Step:%.0f | Total Loss: %.6f | Discrepancy Loss: %.6f| Discrepancy mid Loss: %.6f" % \
                              (epoch_i, args.end_epoch, step, loss_all.item(), loss_discrepancy.item(),
                               loss_discrepancy_mid.item())
                print(output_data)

            # Load pre-trained model with epoch number
        model = model.eval()
        PSNR_mean, SSIM_mean, RMSE_mean = test_implement_MRI(filepaths, model, args.cs_ratio, mask, args.test_name, epoch_i,
                                                             result_dir,args.run_mode,device)
        # save result
        output_data = [epoch_i, loss_all.item(), loss_discrepancy.item(), loss_discrepancy_mid.item(), PSNR_mean,
                       SSIM_mean, RMSE_mean]
        output_file = open(output_file_name, 'a')
        for fp in output_data:   # write data in txt
            output_file.write(str(fp))
            output_file.write(',')
        output_file.write('\n')    # line feed
        output_file.close()

        # save model in every epoch
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters

elif args.run_mode=='test':
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, args.end_epoch)))
    # Load pre-trained model with epoch number
    model = model.eval()
    PSNR_mean, SSIM_mean, RMSE_mean = test_implement_MRI(filepaths, model, args.cs_ratio,mask,args.test_name,args.end_epoch,result_dir,
                                              args.run_mode,device)

#########################################################################################
