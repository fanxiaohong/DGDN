import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import cv2
import copy
from time import time
import math
from skimage.measure import compare_ssim as ssim
import torch
import numpy as np

# define device
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
####################################################################################
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
def compute_measure(y_gt, y_pred, data_range):
    pred_rmse = compute_RMSE(y_pred, y_gt)
    return (pred_rmse)

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))
##################################################################################
# test image index
def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)
#########################################################################################
# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)
#######################################################################################
def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]
#######################################################################################
# 重叠区处理
def imread_CS_py_overlap(Iorg,overlap_size):
    block_size = 33
    block_stride = block_size-overlap_size-1
    [row, col] = Iorg.shape
    # print('image_shape',Iorg.shape)
    row_pad = block_stride-np.mod(row-block_size,block_stride)
    # print('row_pad',row_pad)
    col_pad = block_stride-np.mod(col-block_size,block_stride)
    # print('col_pad', col_pad)
    Ipad = np.concatenate((Iorg, np.zeros([row, row_pad])), axis=1)
    # print('Ipad_shape',Ipad.shape)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    # print('Ipad_shape', Ipad.shape)
    # plt.imshow(Ipad)
    # plt.show()
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]
########################################################################################
def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col
########################################################################################
# 重叠区处理
def img2col_py_overlap(Ipad, block_size,overlap_size):
    block_stride = block_size-overlap_size-1   # 根据重叠尺寸和block尺寸计算滑步尺寸
    [row, col] = Ipad.shape
    # print('Ipad_shape',Ipad.shape)
    row_block = (row-block_size)/block_stride+1
    col_block = (col-block_size)/block_stride+1
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_stride):
        for y in range(0, col-block_size+1, block_stride):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # print('img_col',img_col)
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    # print('img_col_shape', img_col.shape)
    return img_col
######################################################################################
def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec
#####################################################################################
# 重叠区重建后图片处理
def col2im_CS_py_overlap(X_col, row, col, row_new, col_new,overlap_size):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    # print('X0_shape',X0_rec.shape)
    block_stride = block_size-overlap_size-1
    gap_single = int((block_size-1-block_stride)/2)
    count = 0
    for x in range(0, row_new-block_size+1, block_stride):
        for y in range(0, col_new-block_size+1, block_stride):
            x_pad = X_col[:, count].reshape([block_size, block_size])
            # print('x',x,'y',y)
            # print(col_new - block_size)
            if x==0 and y==0:  # 原点块
                X0_rec[0:32-gap_single, 0:32-gap_single] = x_pad[:32-gap_single,:32-gap_single]
                # print(1)
            elif x==0 and y!=0 and y!=(col_new-block_size):  # 上边界
                X0_rec[:32-gap_single, y+gap_single:y+32-gap_single] = x_pad[:32-gap_single,gap_single:32-gap_single]
                # print(2)
            elif x == 0 and y == (col_new - block_size):  # 上边界角
                X0_rec[:32-gap_single, y :y + block_size] = x_pad[:32-gap_single, :block_size]
                # print(3)
            elif x!=0 and y==0 and x!=(row_new-block_size):   # 左边界
                X0_rec[x+gap_single:x+32-gap_single, :32-gap_single] = x_pad[gap_single:32-gap_single,:32-gap_single]
                # print(4)
            elif y==0 and x==(row_new-block_size):   # 左边界角
                X0_rec[x:x+block_size, :32-gap_single] = x_pad[:block_size,:32-gap_single]
                # print(5)
            elif x == (row_new-block_size) and  y!=0 and y!=(col_new-block_size):   # 下边界
                X0_rec[x:x+block_size, y+gap_single:y + 32-gap_single] = x_pad[:block_size, gap_single:32-gap_single]
                # print(6)
            elif y == (col_new-block_size) and  x!=0 and x!=(row_new-block_size):   # 右边界
                X0_rec[x+gap_single:x + 32-gap_single, y:y + block_size] = x_pad[gap_single:32-gap_single, :block_size]
                # print(7)
            elif x == (row_new-block_size) and y == (col_new-block_size): # 右下角
                X0_rec[x :x + block_size, y:y + block_size] = x_pad[:block_size, :block_size]
                # print(8)
            else:  # 中心区域面积为16X16
                X0_rec[x+gap_single:x+ gap_single+block_stride, y+gap_single:y + gap_single+block_stride] = \
                    x_pad[gap_single:gap_single+ block_stride, gap_single:gap_single+ block_stride]
                # print(9)
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            count = count + 1
            # plt.imshow(X0_rec)
            # plt.show()
    X_rec = X0_rec[:row, :col]
    # plt.imshow(X_rec)
    # plt.show()
    return X_rec
#####################################################################################
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
###################################################################################
# test implement image, save image
def test_implement(filepaths,Phi, Qinit,model,cs_ratio, test_name,epoch_num,result_dir,overlap_size,run_mode,loss_mode):
    print('\n')
    print("CS Reconstruction Start")
    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    Init_PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    Init_SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    RMSE_total = []
    with torch.no_grad():
        for img_no in range(ImgNum):
            imgName = filepaths[img_no]
            Img = cv2.imread(imgName, 1)
            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()
            Iorg_y = Img_yuv[:, :, 0]
            if overlap_size == 0:
                [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
                Icol = img2col_py(Ipad, 33).transpose() / 255.0
            else:
                [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py_overlap(Iorg_y,overlap_size)
                Icol = img2col_py_overlap(Ipad, 33,overlap_size).transpose() / 255.0
            Img_output = Icol
            start = time()
            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
            if loss_mode == 'ISTAplus':
                [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)
            elif loss_mode == 'Fista':
                [x_output, loss_layers_sym, encoder_st] = model(Phix, Phi,Qinit)
            end = time()
            PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
            initial_result = PhiTPhi.cpu().data.numpy()
            Prediction_value = x_output.cpu().data.numpy()
            if overlap_size == 0:
                X_init = np.clip(col2im_CS_py(initial_result.transpose(), row, col, row_new, col_new), 0, 1)
                X_rec = np.clip(col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new), 0, 1)
                # X_rec = np.clip(col2im_CS_py(initial_result.transpose(), row, col, row_new, col_new), 0, 1)
            else:
                X_init = np.clip(col2im_CS_py_overlap(initial_result.transpose(), row, col, row_new, col_new,overlap_size), 0, 1)
                X_rec = np.clip(col2im_CS_py_overlap(Prediction_value.transpose(), row, col, row_new, col_new,overlap_size), 0, 1)
                # X_rec = np.clip(col2im_CS_py(initial_result.transpose(), row, col, row_new, col_new), 0, 1)
            init_PSNR = psnr(X_init * 255, Iorg.astype(np.float64))
            init_SSIM = ssim(X_init * 255, Iorg.astype(np.float64), data_range=255)
            rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)
            m_reg = compute_measure(Iorg.astype(np.float64)/255, X_rec , 1)
            RMSE_total.append(m_reg)
            if run_mode == 'test':
                print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f, RMSE is %.4f" % (
                img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM,m_reg))
                Img_rec_yuv[:, :, 0] = X_rec * 255
                im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
                # resultName = imgName.replace(data_dir, result_dir)
                img_name = imgName.split('/')
                img_name = str(img_name[2])
                img_name = img_name.split('.')
                img_dir = result_dir+img_name[0]+"_PSNR_%.2f_SSIM_%.4f_RMSE_%.4f.png" % ( rec_PSNR, rec_SSIM,m_reg)
                # cv2.imwrite("%s_PSNR_%.2f_SSIM_%.4f.png" % (resultName, rec_PSNR, rec_SSIM), im_rec_rgb)
                cv2.imwrite(img_dir, im_rec_rgb)
            del x_output
            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            Init_PSNR_All[0, img_no] = init_PSNR
            Init_SSIM_All[0, img_no] = init_SSIM
    print('\n')
    # init_data = "CS ratio is %d, Avg Initial  PSNR/SSIM for %s is %.2f-%.4f/%.4f-%.4f" % (cs_ratio, test_name, np.mean(Init_PSNR_All), np.std(Init_PSNR_All), np.mean(Init_SSIM_All), np.std(Init_SSIM_All))
    output_data = "CS ratio is %d, Avg Proposed PSNR/SSIM/RMSE for %s is %.2f-%.2f/%.4f-%.4f/%.4f-%.4f, Epoch number of model is %d \n" % (
    cs_ratio, test_name, np.mean(PSNR_All), np.std(PSNR_All), np.mean(SSIM_All), np.std(SSIM_All),
    np.array(RMSE_total).mean(), np.array(RMSE_total).std(), epoch_num)
    # print(init_data)
    print(output_data)
    print("CS Reconstruction End")
    return np.mean(PSNR_All), np.mean(SSIM_All),np.array(RMSE_total).mean()
###################################################################################
# test implement
def test_implement_MRI(filepaths, model, cs_ratio, mask, test_name, epoch_num, result_dir,run_mode,loss_mode):
    print('\n')
    print("CS Reconstruction Start")
    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    Init_PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    Init_SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    RMSE_total = []
    with torch.no_grad():
        for img_no in range(ImgNum):
            imgName = filepaths[img_no]
            Iorg = cv2.imread(imgName, 0)
            Icol = Iorg.reshape(1, 1, 256, 256) / 255.0
            Img_output = Icol
            start = time()
            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            PhiTb = FFT_Mask_ForBack()(batch_x, mask)
            if loss_mode == 'ISTAplus':
                [x_output, loss_layers_sym] = model(PhiTb, mask)
            elif loss_mode == 'Fista':
                [x_output, loss_layers_sym, encoder_st] = model(PhiTb, mask)
            elif loss_mode == 'RDB':
                [x_output, x_mid] = model(PhiTb, mask)
            end = time()
            initial_result = PhiTb.cpu().data.numpy().reshape(256, 256)
            Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)
            X_init = np.clip(initial_result, 0, 1).astype(np.float64)
            # X_rec = np.clip(initial_result, 0, 1).astype(np.float64)
            X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)
            init_PSNR = psnr(X_init * 255, Iorg.astype(np.float64))
            init_SSIM = ssim(X_init * 255, Iorg.astype(np.float64), data_range=255)
            rec_PSNR = psnr(X_rec * 255., Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec * 255., Iorg.astype(np.float64), data_range=255)
            m_reg = compute_measure(Iorg.astype(np.float64)/255, X_rec, 1)
            RMSE_total.append(m_reg)
            if run_mode == 'test':
                # print("[%02d/%02d] Run time for %s is %.4f, Initial  PSNR is %.2f, Initial  SSIM is %.4f" % (
                # img_no, ImgNum, imgName, (end - start), init_PSNR, init_SSIM))
                print("[%02d/%02d] Run time for %s is %.4f, Proposed PSNR is %.2f, Proposed SSIM is %.4f, Proposed RMSE is %.4f" % (
                img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM,m_reg))
                im_rec_rgb = np.clip(X_rec * 255, 0, 255).astype(np.uint8)
                img_name = imgName.split('/')
                img_name = str(img_name[2])
                img_name = img_name.split('.')
                img_dir = result_dir + img_name[0] + "_PSNR_%.2f_SSIM_%.4f_RMSE_%.4f.png" % (rec_PSNR, rec_SSIM,m_reg)
                # cv2.imwrite("%s_PSNR_%.2f_SSIM_%.4f.png" % (resultName, rec_PSNR, rec_SSIM), im_rec_rgb)
                cv2.imwrite(img_dir, im_rec_rgb)
            del x_output
            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            Init_PSNR_All[0, img_no] = init_PSNR
            Init_SSIM_All[0, img_no] = init_SSIM
    print('\n')
    # init_data = "CS ratio is %d, Avg Initial  PSNR/SSIM for %s is %.2f-%.4f/%.4f-%.4f" % (cs_ratio, test_name, np.mean(Init_PSNR_All),np.std(Init_PSNR_All),np.mean(Init_SSIM_All),np.std(Init_SSIM_All))
    output_data = "CS ratio is %d, Avg Proposed PSNR/SSIM/RMSE for %s is %.2f-%.2f/%.4f-%.4f/%.4f-%.4f, Epoch number of model is %d \n" % (
        cs_ratio, test_name, np.mean(PSNR_All), np.std(PSNR_All), np.mean(SSIM_All), np.std(SSIM_All),
        np.array(RMSE_total).mean(), np.array(RMSE_total).std(), epoch_num)
    print(output_data)
    print("CS Reconstruction End")
    return np.mean(PSNR_All), np.mean(SSIM_All),np.array(RMSE_total).mean()
##############################################################################################################