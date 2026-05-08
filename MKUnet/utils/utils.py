
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from medpy import metric
from scipy.ndimage import zoom
import seaborn as sns
from PIL import Image 
import matplotlib.pyplot as plt
from segmentation_mask_overlay import overlay_masks
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import SimpleITK as sitk
import pandas as pd

import os
from thop import profile
from thop import clever_format

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
def one_hot_encoder(input_tensor,dataset,n_classes = None):
    tensor_list = []
    if dataset == 'MMWHS':  
        dict = [0,205,420,500,550,600,820,850]
        for i in dict:
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    else:
        for i in range(n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()    

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #print(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.assd(pred, gt)
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0


class WeightedFusion(nn.Module):
    """
    Computes per-pixel attention weights for each prediction from a set of decoder stages,
    then fuses them as a weighted sum. This preserves the original logit values without
    adding extra non-linearities that might distort class scores.
    """
    def __init__(self, num_stages, num_classes):
        """
        Args:
            num_stages (int): Number of predictions (decoder stages) to fuse.
            num_classes (int): Number of channels (classes) in each prediction.
        """
        super(WeightedFusion, self).__init__()
        # For each stage, learn a 1x1 conv that outputs a single-channel weight map.
        self.weight_convs = nn.ModuleList(
            [nn.Conv2d(num_classes, 1, kernel_size=1) for _ in range(num_stages)]
        )
    
    def forward(self, predictions):
        """
        Args:
            predictions (list[Tensor]): List of predictions, each of shape [B, C, H, W].
        Returns:
            fused (Tensor): Fused prediction of shape [B, C, H, W].
        """
        weight_maps = [conv(pred) for pred, conv in zip(predictions, self.weight_convs)]
        # Stack weight maps: shape [B, num_stages, 1, H, W]
        weights = torch.stack(weight_maps, dim=1)
        # Normalize weights per pixel over the decoder stages.
        weights = F.softmax(weights, dim=1)
        # Stack predictions: shape [B, num_stages, C, H, W]
        preds = torch.stack(predictions, dim=1)
        # Fuse by weighted sum over the stages.
        fused = torch.sum(weights * preds, dim=1)
        return fused
    
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if class_names==None:
        mask_labels = np.arange(1,classes)
    else:
        mask_labels = class_names
    #cmaps = mcolors.BASE_COLORS
    #print(test_save_path)
    cmaps = mcolors.CSS4_COLORS
    #print(cmaps)
    my_colors=['red','darkorange','yellow','forestgreen','blue','purple','magenta','cyan','deeppink', 'chocolate', 'olive','deepskyblue','darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}
    #print(cmap)
    if len(image.shape) == 3:
        #print("Here")
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                #p1, p2, p3, p4 = net(input)
                P = net(input)
                if  not isinstance(P, list):
                    P = [P]
                #print(len(P))
                #outputs = p1 + p2 + p3 + p4
                outputs = 0.0
                #for idx in range(len(P)):
                #    outputs += P[idx]
                outputs = P[-1]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
                # saving the final output
                # as a PNG file
                #print(test_save_path + '/'+case + '' +str(ind))
                #Image.fromarray((pred/8 * 255).astype(np.uint8)).save(test_save_path + '/'+case + '_' +str(ind)+'_pred.png')
                Image.fromarray((image[ind, :, :] * 255).astype(np.uint8)).save(test_save_path + '/'+case + '_' +str(ind)+'_img.png')
                #Image.fromarray((label[ind, :, :]/8 * 255).astype(np.uint8)).save(test_save_path + '/'+case + '_' +str(ind)+'_gt.png')
                #cmap = plt.cm.tab20(np.arange(len(mask_labels)))
                
                lbl = label[ind, :, :]
                masks = []
                for i in range(1, classes):
                    masks.append(lbl==i)
                preds_o = []
                for i in range(1, classes):
                    preds_o.append(pred==i)
                
                fig_gt = overlay_masks(image[ind, :, :], masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                fig_pred = overlay_masks(image[ind, :, :], preds_o, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                # Do with that image whatever you want to do.
                fig_gt.savefig(test_save_path + '/'+case + '_' +str(ind)+'_gt.png', bbox_inches="tight", dpi=300)
                fig_pred.savefig(test_save_path + '/'+case + '_' +str(ind)+'_pred.png', bbox_inches="tight", dpi=300)

    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            #p1, p2, p3, p4 = net(input)
            P = net(input)
            if  not isinstance(P, list):
                P = [P]
            #print(len(P))
            #outputs = p1 + p2 + p3 + p4
            outputs = 0.0
            #for idx in range(len(P)):
            #    outputs += P[idx]
            outputs = P[-1]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def fig_to_numpy(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape((h, w, 3))

def make_overlay_rgb(image, gt_mask, pred_mask, class_names, palette, alpha_gt=0.3, alpha_pred=0.6):
    # 1) GT overlay
    pil_gt = overlay_masks(image,
                           [gt_mask == c for c in class_names],
                           labels=class_names,
                           colors={c: palette[i] for i, c in enumerate(class_names)},
                           mask_alpha=alpha_gt)
    arr = np.asarray(pil_gt)  # H×W×3
    # 2) prediction overlay
    pil_pred = overlay_masks(arr,
                             [pred_mask == c for c in class_names],
                             labels=class_names,
                             colors={c: palette[i] for i, c in enumerate(class_names)},
                             mask_alpha=alpha_pred)
    return np.asarray(pil_pred)  # H×W×3

def test_single_volume_save_preds(image, label, net, loss_module,
                                  classes, patch_size, test_save_path, case,
                                  z_spacing=1.0, class_names=None):
    os.makedirs(test_save_path, exist_ok=True)
    img3d = image.squeeze(0).cpu().numpy()   # D×H×W
    lbl3d = label.squeeze(0).cpu().numpy()   # D×H×W
    D, H0, W0 = img3d.shape

    # pick palette
    class_list = class_names if class_names else list(range(1,classes))
    palette = ['red','darkorange','yellow','forestgreen','blue',
               'purple','magenta','cyan','deeppink','chocolate','olive']
    
    net.eval()
    orig_overlays   = [[] for _ in range(4)]
    synth_overlays  = [[] for _ in range(len(loss_module.operations)*(2**len(loss_module.operations)-len(loss_module.operations)-1))]

    with torch.no_grad():
        for z in range(D):
            slice_ = img3d[z]
            H, W = slice_.shape
            # 1) resize to patch
            if (H,W) != tuple(patch_size):
                inp_slice = zoom(slice_, (patch_size[0]/H, patch_size[1]/W), order=3)
            else:
                inp_slice = slice_

            inp = torch.from_numpy(inp_slice[None,None]).float().to(next(net.parameters()).device)
            P = net(inp)
            if not isinstance(P,list): P=[P]

            # 2) originals
            for i, logit in enumerate(P):
                pred = torch.argmax(torch.softmax(logit,1),1)[0].cpu().numpy()
                # back to original size
                if pred.shape != (H0,W0):
                    pred = zoom(pred,(H0/pred.shape[0],W0/pred.shape[1]),order=0)
                # overlay and store
                rgb = make_overlay_rgb(slice_, lbl3d[z], pred, class_list, palette)
                orig_overlays[i].append(rgb)

            # 3) synthetics
            fused = loss_module(P)
            for j, flogit in enumerate(fused):
                pred = torch.argmax(torch.softmax(flogit,1),1)[0].cpu().numpy()
                if pred.shape != (H0,W0):
                    pred = zoom(pred,(H0/pred.shape[0],W0/pred.shape[1]),order=0)
                rgb = make_overlay_rgb(slice_, lbl3d[z], pred, class_list, palette)
                synth_overlays[j].append(rgb)

    # 4) Mosaic each slice horizontally for orig and synth
    for z in range(D):
        # original mosaic: H×(4W)×3
        row = np.concatenate([orig_overlays[i][z] for i in range(4)], axis=1)
        plt.imsave(os.path.join(test_save_path,f"{case}_{z}_orig_mosaic.png"), row)
        # synthetic mosaic: 4 rows × 11 cols => (4H)×(11W)×3
        rows = []
        for r in range(4):
            cols = synth_overlays[r*11:(r+1)*11]
            rows.append(np.concatenate([cols[c][z] for c in range(11)], axis=1))
        big = np.concatenate(rows, axis=0)
        plt.imsave(os.path.join(test_save_path,f"{case}_{z}_synth_mosaic.png"), big)

    # 5) single legend image (one time)
    handles = [mpatches.Patch(color=palette[i], label=str(c)) 
               for i,c in enumerate(class_list)]
    fig = plt.figure(figsize=(len(class_list)*0.5, 1))
    fig.legend(handles=handles, ncol=len(class_list), frameon=False)
    fig.savefig(os.path.join(test_save_path,"legend.png"), bbox_inches='tight')
    plt.close(fig)

    # 6) save volumes
    sitk.WriteImage(sitk.GetImageFromArray(lbl3d.astype(np.uint8)),
                    os.path.join(test_save_path,f"{case}_gt.nii.gz"))
    for i in range(4):
        vol = np.stack(orig_overlays[i], axis=0).argmax(axis=-1).astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(vol),
                        os.path.join(test_save_path,f"{case}_orig{i}.nii.gz"))
    for j in range(len(synth_overlays)):
        vol = np.stack(synth_overlays[j], axis=0).argmax(axis=-1).astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(vol),
                        os.path.join(test_save_path,f"{case}_syn{j}.nii.gz"))

def test_single_volume_fm(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P = net(input)
                ss = [[0],[1],[2],[3],[0,1,2,3], [0,1], [0,2], [0,3], [1,2], [1,3], [2,3], [0,1,2],[0,1,3],[0,2,3],[1,2,3]]
                outputs = 0.0
                for s in ss:
                    iout = 0.0
                    #print(s)
                    for idx in range(len(s)):
                        iout += P[s[idx]]
                    outputs += torch.softmax(iout, dim=1) #F.interpolate(iout, size=(val_label_batch.shape[-2:]), mode='bilinear'), dim=1)
                out = torch.argmax(outputs, dim=1).squeeze(0)
                #p1, p2, p3, p4 = net(input)
                #outputs = p1 + p2 + p3 + p4
                #out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            ss = [[0],[1],[2],[3],[0,1,2,3], [0,1], [0,2], [0,3], [1,2], [1,3], [2,3], [0,1,2],[0,1,3],[0,2,3],[1,2,3]]
            outputs = 0.0
            for s in ss:
                iout = 0.0
                #print(s)
                for idx in range(len(s)):
                    iout += P[s[idx]]
                outputs += torch.softmax(iout, dim=1) #F.interpolate(iout, size=(val_label_batch.shape[-2:]), mode='bilinear'), dim=1)
            out = torch.argmax(outputs, dim=1).squeeze(0)
            #p1, p2, p3, p4 = net(input)
            #outputs = p1 + p2 + p3 + p4
            #out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    #label = F.interpolate(label, size=(patch_size[0], patch_size[1]), mode='bilinear') # not for double_maxvits
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
    
def test_single_volume_snorm(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                p1, p2, p3, p4 = net(input)
                outputs = p1 + p2 + p3 + p4
                outputs = torch.softmax(outputs, dim=1)
                outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-8)
                out = torch.argmax(outputs, dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            p1, p2, p3, p4 = net(input)
            outputs = p1 + p2 + p3 + p4
            outputs = torch.softmax(outputs, dim=1)
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-8)
            out = torch.argmax(outputs, dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def test_single_volume1(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P = net(input)
                outputs = P[-1]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = P[-1]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    '''device = next(net.parameters()).device
    weighted_fusion_module = WeightedFusion(num_stages=4, num_classes=classes).to(device)
    # If this module is part of your overall model, its parameters would be learned during training.
    # Otherwise, you can load the trained state dict here before inference:
    # weighted_fusion_module.load_state_dict(torch.load('weighted_fusion_weights.pt'))

    # Set the module to evaluation mode.
    weighted_fusion_module.eval()'''

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                #p1, p2, p3, p4 = net(input)
                #P = net(input, img_size=patch_size[1])
                #P = net(input, im_size=patch_size[0])
                P = net(input)
                #print(len(P))
                #outputs = p1 + p2 + p3 + p4
                outputs = 0.0
                #for idx in range(len(P)):#range(len(P)):
                #   outputs += P[idx]
                outputs = P[-1]
                # Fuse the four predictions using the learned weights.
                
                #outputs = weighted_fusion_module([P[0], P[1], P[2], P[3]])

                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            #p1, p2, p3, p4 = net(input)
            P = net(input)
            #outputs = p1 + p2 + p3 + p4
            outputs = 0.0
            #for idx in range(len(P)):#range(len(P)):
            #   outputs += P[idx]
            outputs = P[-1]
            # Fuse the four predictions using the learned weights.
            #outputs = weighted_fusion_module([P[0], P[1], P[2], P[3]])
            
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list

def val_single_volume_2out(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                p1, p2 = net(input)
                outputs = p1 + p2
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            p1, p2 = net(input)
            outputs = p1 + p2
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list

def val_single_volume_1out(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                #p1, p2, p3, p4 = net(input)
                #P = net(input, img_size=patch_size[1])
                #P = net(input, im_size=patch_size[0])
                P = net(input)
                #print(len(P))
                #outputs = p1 + p2 + p3 + p4
                outputs = 0.0
                #for idx in range(len(P)):#range(len(P)):
                #   outputs += P[idx]
                outputs = P[0]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            #p1, p2, p3, p4 = net(input)
            P = net(input)
            #outputs = p1 + p2 + p3 + p4
            outputs = 0.0
            #for idx in range(len(P)):#range(len(P)):
            #   outputs += P[idx]
            outputs = P[3]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list        
def val_single_volume2(image, label, net1, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                p1, p2, p3, p4 = net1(input)
                p1_2, p2_2, p3_2, p4_2 = net(input)
                outputs = p1 + p2 + p3 + p4 + p1_2 + p2_2 + p3_2 + p4_2
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            p1, p2, p3, p4 = net1(input)
            p1_2, p2_2, p3_2, p4_2 = net(input)
            outputs = p1 + p2 + p3 + p4 + p1_2 + p2_2 + p3_2 + p4_2
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list

def horizontal_flip(image):
    image = image[:, ::-1, :]
    return image

def vertical_flip(image):
    image = image[::-1, :, :]
    return image

def tta_model(model, image):
    n_image = image
    h_image = horizontal_flip(image)
    v_image = vertical_flip(image)

    n_mask = model.predict(np.expand_dims(n_image, axis=0))[0]
    h_mask = model.predict(np.expand_dims(h_image, axis=0))[0]
    v_mask = model.predict(np.expand_dims(v_image, axis=0))[0]

    n_mask = n_mask
    h_mask = horizontal_flip(h_mask)
    v_mask = vertical_flip(v_mask)

    mean_mask = (n_mask + h_mask + v_mask) / 3.0
    return mean_mask

def predict_sam(predictor_tuned, image, mask, perturb_h_len=30):
    height, width, _ = image.shape
    H, W = mask.shape
                    
    y_indices, x_indices = np.where(mask > 0)
                    
    if(len(x_indices) == 0 or len(y_indices) == 0):
        x_min, x_max = 0, W-1
        y_min, y_max = 0, H-1
    else:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        
    # add perturbation to bounding box coordinates                       
    x_min = max(0, x_min - perturb_h_len)#np.random.randint(perturb_l_len, perturb_h_len)) #20
    x_max = min(W, x_max + perturb_h_len)#np.random.randint(perturb_l_len, perturb_h_len)) #20
    y_min = max(0, y_min - perturb_h_len)#np.random.randint(perturb_l_len, perturb_h_len)) #20
    y_max = min(H, y_max + perturb_h_len)#np.random.randint(perturb_l_len, perturb_h_len)) #20
    input_bbox = np.array([x_min, y_min, x_max, y_max])
    predictor_tuned.set_image(image)

    #input_bbox = np.array([50,50,width-1,height-1])

    pred, _, _ = predictor_tuned.predict(
                point_coords=None,
                box=input_bbox,
                multimask_output=False,
            ) 
    #print(pred.max())
    return pred

def tta_model_sam_1st(predictor_tuned, image, mask, perturb_h_len=30):

    
    #print(image.shape)
    #print(mask.shape)

    n_image = image
    n_mask = mask
    h_image = np.transpose(horizontal_flip(np.transpose(image,(2,0,1))),(1,2,0))
    h_mask = horizontal_flip(np.expand_dims(mask, axis=0))
    v_image = np.transpose(vertical_flip(np.transpose(image,(2,0,1))),(1,2,0))
    v_mask = vertical_flip(np.expand_dims(mask, axis=0))

    n_pred_mask = predict_sam(predictor_tuned, n_image, n_mask, perturb_h_len=perturb_h_len) #model.predict(np.expand_dims(n_image, axis=0))[0]
    h_pred_mask = predict_sam(predictor_tuned, h_image, h_mask[0,:,:], perturb_h_len=perturb_h_len) #model.predict(np.expand_dims(h_image, axis=0))[0]
    v_pred_mask = predict_sam(predictor_tuned, v_image, v_mask[0,:,:], perturb_h_len=perturb_h_len) #model.predict(np.expand_dims(v_image, axis=0))[0]

    n_pred_mask = n_pred_mask.astype(np.float32)
    h_pred_mask = horizontal_flip(np.expand_dims(h_pred_mask, axis=0)).astype(np.float32)
    v_pred_mask = vertical_flip(np.expand_dims(v_pred_mask, axis=0)).astype(np.float32)

    
    #print(n_pred_mask==True)
    '''n_pred_mask[n_pred_mask==True]=1.0
    n_pred_mask[n_pred_mask==False]=0.0

    h_pred_mask[h_pred_mask==True]=1.0
    h_pred_mask[h_pred_mask==False]=0.0

    v_pred_mask[v_pred_mask==True]=1.0
    v_pred_mask[v_pred_mask==False]=0.0'''

    #print(n_pred_mask)

    #print(n_pred_mask.max(), h_pred_mask.max(), v_pred_mask.max())
    mean_mask = (n_pred_mask + h_pred_mask[0,:,:] + v_pred_mask[0,:,:]) / 3.0

    #print(mean_mask.max())

    return mean_mask

def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')