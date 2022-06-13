import torch
import models
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import cv2
from einops import rearrange, repeat
from torch.utils.tensorboard import SummaryWriter
import utils.metrics as metrics
from hausdorff import hausdorff_distance
import time
import torch.nn.functional as F
import random
from utils.loss_functions.dice_loss import DC_and_BCE_loss, DC_and_CE_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#  ============================== add the seed to make sure the results are reproducible ==============================

seed_value = 5000  # the number of seed
np.random.seed(seed_value)  # set random seed for numpy
random.seed(seed_value)  # set random seed for python
os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
torch.manual_seed(seed_value)  # set random seed for CPU
torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
torch.backends.cudnn.deterministic = True  # set random seed for convolution

#  ================================================ parameters setting ================================================

parser = argparse.ArgumentParser(description='Medical Transformer')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int, metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--dataset', default='../../dataset_cardiac/', type=str)
parser.add_argument('--modelname', default='C2FTrans', type=str, help='type of model')
parser.add_argument('--classes', type=int, default=4, help='number of classes')
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str, help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str, help='load a pretrained model')
parser.add_argument('--save', default='default', type=str, help='save the model')
parser.add_argument('--direc', default='./medt', type=str, help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=256)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='yes', type=str)
parser.add_argument('--tensorboard', default='./tensorboard/', type=str)
parser.add_argument('--eval_mode', default='patient', type=str)

#  =============================================== model initialization ===============================================

args = parser.parse_args()
direc = args.direc  # the path of saving model
eval_mode = args.eval_mode

if args.gray == "yes":
    from utils.utils_multi import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 1
else:
    from utils.utils_multi_rgb import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 3
if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(img_size=args.imgsize, crop=crop, p_flip=0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0, p_contr=0.0, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
tf_val = JointTransform2D(img_size=args.imgsize, crop=crop, p_flip=0, p_gama=0, color_jitter_params=None, long_mask=True)  # image reprocessing
train_dataset = ImageToImage2D(args.dataset, 'train', tf_train, args.classes)  # only random horizontal flip, return image, mask, and filename
val_dataset = ImageToImage2D(args.dataset, 'val', tf_val, args.classes)  # no flip, return image, mask, and filename
test_dataset = ImageToImage2D(args.dataset, 'test', tf_val, args.classes)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
device = torch.device("cuda")

if args.modelname == "C2FTrans":
    model = models.MPtrans.C2FTransformer(n_channels=imgchant, n_classes=args.classes, imgsize=args.imgsize)

model.to(device)
criterion = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
celoss = nn.CrossEntropyLoss(ignore_index=-1)  # criterion = LogNLLLoss()
smoothl1 = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=1e-5)
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))
timestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
boardpath = './tensorboard_cardiac/' + args.modelname + '_' + timestr
if not os.path.isdir(boardpath):
    os.makedirs(boardpath)
TensorWriter = SummaryWriter(boardpath)

#  ============================================= begin to train the model =============================================

best_dice = 0.0
for epoch in range(args.epochs):
    #  ---------------------------------- training ----------------------------------

    model.train()
    train_losses = 0
    train_losses1 = 0
    train_losses2 = 0
    train_losses3 = 0
    train_lossesf = 0

    for batch_idx, (X_batch, mask, mask1, mask2, mask3, *rest) in enumerate(dataloader):
        X_batch = Variable(X_batch.to(device='cuda'))
        mask = Variable(mask.to(device='cuda'))
        mask1 = Variable(mask1.to(device='cuda'))
        mask2 = Variable(mask2.to(device='cuda'))
        mask3 = Variable(mask3.to(device='cuda'))

        # ------------------------- forward ------------------------------

        output, output1, output2, output3 = model(X_batch)
        output1 = F.softmax(output1, dim=1)
        output2 = F.softmax(output2, dim=1)
        output3 = F.softmax(output3, dim=1)
        '''
        train_loss1 = 8*smoothl1(output1[:, 1:, :, :], mask1[:, 1:, :, :]) + 2 * smoothl1(output1[:, 0, :, :], mask1[:, 0, :, :])
        train_loss2 = 8*smoothl1(output2[:, 1:, :, :], mask2[:, 1:, :, :]) + 2 * smoothl1(output2[:, 0, :, :], mask2[:, 0, :, :])
        train_loss3 = 8*smoothl1(output3[:, 1:, :, :], mask3[:, 1:, :, :]) + 2 * smoothl1(output3[:, 0, :, :], mask3[:, 0, :, :])
        '''
        mask2_ = torch.argmax(mask2, dim=1)
        mask3_ = torch.argmax(mask3, dim=1)
        train_loss1 = 0.5*smoothl1(output1, mask1) + 0.5*criterion(output1, mask)
        train_loss2 = 0.5*smoothl1(output2, mask2) + 0.5*criterion(output2, mask2_)
        train_loss3 = 0.5*smoothl1(output3, mask3) + 0.5*criterion(output3, mask3_)
        train_lossf = criterion(output, mask)
        train_loss = 0.6 * train_lossf + 0.2 * train_loss1 + 0.1 * train_loss2 + 0.1 * train_loss3
        print(train_loss)

        # ------------------------- backward -----------------------------

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_losses = train_losses + train_loss.item()
        train_losses1 = train_losses1 +train_loss1.item()
        train_losses2 = train_losses2 +train_loss2.item()
        train_losses3 = train_losses3 +train_loss3.item()
        train_lossesf = train_lossesf +train_lossf.item()
        #print(train_loss)

    #  ---------------------------- log the train progress ----------------------------

    print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, args.epochs, train_losses / (batch_idx + 1)))
    TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
    TensorWriter.add_scalar('train_loss1', train_losses1 / (batch_idx + 1), epoch)
    TensorWriter.add_scalar('train_loss2', train_losses2 / (batch_idx + 1), epoch)
    TensorWriter.add_scalar('train_loss3', train_losses3 / (batch_idx + 1), epoch)
    TensorWriter.add_scalar('train_lossf', train_lossesf / (batch_idx + 1), epoch)

    #  ----------------------------------- evaluate -----------------------------------

    model.eval()
    val_losses = 0
    dices = 0
    hds = 0
    smooth = 1e-25
    mdices, mhds = np.zeros(args.classes), np.zeros(args.classes)
    if eval_mode == "slice":
        for batch_idx, (X_batch, mask, mask1, mask2, mask3, *rest) in enumerate(valloader):
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
            X_batch = Variable(X_batch.to(device='cuda'))
            mask = Variable(mask.to(device='cuda'))
            mask1 = Variable(mask1.to(device='cuda'))
            mask2 = Variable(mask2.to(device='cuda'))
            mask3 = Variable(mask3.to(device='cuda'))
            # start = timeit.default_timer()
            with torch.no_grad():
                y_outf, y_out1, y_out2, y_out3 = model(X_batch)
            # stop = timeit.default_timer()
            # print('Time: ', stop - start)
            val_loss1 = celoss(y_out1, mask1)
            val_loss2 = celoss(y_out2, mask2)
            val_loss3 = celoss(y_out3, mask3)
            val_lossf = criterion(y_outf, mask)
            val_losses += 0.6 * val_lossf + 0.2 * val_loss1.item() + 0.1 * val_loss2.item() + 0.1 * val_loss3.item()
            y_out = F.softmax(y_outf, dim=1)
            gt = mask.detach().cpu().numpy()
            pred = y_out.detach().cpu().numpy()  # (b, c,h, w) tep
            seg = np.argmax(pred, axis=1)  # (b, h, w) whether exist same score?

            b, h, w = seg.shape
            for i in range(1, args.classes):
                pred_i = np.zeros((b, h, w))
                pred_i[seg == i] = 255
                gt_i = np.zeros((b, h, w))
                gt_i[gt == i] = 255
                mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
                mhds[i] += hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan")
                del pred_i, gt_i
        mdices = mdices / (batch_idx + 1)
        mhds = mhds / (batch_idx + 1)
        for i in range(1, args.classes):
            dices += mdices[i]
            hds += mhds[i]
        print(dices / (args.classes - 1), hds / (args.classes - 1))
        print('epoch [{}/{}], test loss:{:.4f}'.format(epoch, args.epochs, val_losses / (batch_idx + 1)))
        print('epoch [{}/{}], test dice:{:.4f}'.format(epoch, args.epochs, dices / (args.classes - 1)))
        TensorWriter.add_scalar('val_loss', val_losses / (batch_idx + 1), epoch)
        TensorWriter.add_scalar('dices', dices / (args.classes - 1), epoch)
        TensorWriter.add_scalar('hausdorff', hds / (args.classes - 1), epoch)

        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True
        if dices / (args.classes - 1) > best_dice:
            best_dice = dices / ((args.classes - 1))
            timestr = time.strftime('%m%d%H%M')
            save_path = './checkpoints/' + args.modelname + '_%s' % timestr + '_' + str(best_dice)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
    else:
        flag = np.zeros(200)  # record the patients
        for batch_idx, (X_batch, mask, mask1, mask2, mask3, *rest) in enumerate(valloader):
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)
            X_batch = Variable(X_batch.to(device='cuda'))
            mask = Variable(mask.to(device='cuda'))
            mask1 = Variable(mask1.to(device='cuda'))
            mask2 = Variable(mask2.to(device='cuda'))
            mask3 = Variable(mask3.to(device='cuda'))
            # start = timeit.default_timer()
            with torch.no_grad():
                y_outf, y_out1, y_out2, y_out3 = model(X_batch)
            # stop = timeit.default_timer()
            # print('Time: ', stop - start)
            y_out1 = F.softmax(y_out1, dim=1)
            y_out2 = F.softmax(y_out2, dim=1)
            y_out3 = F.softmax(y_out3, dim=1)
            val_loss1 = smoothl1(y_out1, mask1) #celoss
            val_loss2 = smoothl1(y_out2, mask2)
            val_loss3 = smoothl1(y_out3, mask3)
            val_lossf = criterion(y_outf, mask)
            val_losses += 0.6 * val_lossf + 0.2 * val_loss1.item() + 0.1 * val_loss2.item() + val_loss3.item()

            y_out = F.softmax(y_outf, dim=1)
            gt = mask.detach().cpu().numpy()
            pred = y_out.detach().cpu().numpy()  # (b, c,h, w) tep
            seg = np.argmax(pred, axis=1)  # (b, h, w) whether exist same score?

            patientid = int(image_filename[:3])
            if flag[patientid] == 0:
                if np.sum(flag) > 0:  # compute the former result
                    b, s, h, w = seg_patient.shape
                    for i in range(1, args.classes):
                        pred_i = np.zeros((b, s, h, w))
                        pred_i[seg_patient == i] = 1
                        gt_i = np.zeros((b, s, h, w))
                        gt_i[gt_patient == i] = 1
                        mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
                        del pred_i, gt_i
                seg_patient = seg[:, None, :, :]
                gt_patient = gt[:, None, :, :]
                flag[patientid] = 1
            else:
                seg_patient = np.concatenate((seg_patient, seg[:, None, :, :]), axis=1)
                gt_patient = np.concatenate((gt_patient, gt[:, None, :, :]), axis=1)
            # ---------------the last patient--------------
        b, s, h, w = seg_patient.shape
        for i in range(1, args.classes):
            pred_i = np.zeros((b, s, h, w))
            pred_i[seg_patient == i] = 1
            gt_i = np.zeros((b, s, h, w))
            gt_i[gt_patient == i] = 1
            mdices[i] += metrics.dice_coefficient(pred_i, gt_i)
            del pred_i, gt_i
        patients = np.sum(flag)
        mdices = mdices / patients
        for i in range(1, args.classes):
            dices += mdices[i]
        print('epoch [{}/{}], test loss:{:.4f}'.format(epoch, args.epochs, val_losses / (batch_idx + 1)))
        TensorWriter.add_scalar('val_loss', val_losses / (batch_idx + 1), epoch)
        TensorWriter.add_scalar('dices', dices / (args.classes - 1), epoch)
        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True
        if dices / (args.classes - 1) > best_dice or epoch == args.epochs - 1:
            best_dice = dices / (args.classes - 1)
            timestr = time.strftime('%m%d%H%M')
            save_path = './checkpoints_cardiac/' + args.modelname + '_%s' % timestr + '_' + str(best_dice)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)