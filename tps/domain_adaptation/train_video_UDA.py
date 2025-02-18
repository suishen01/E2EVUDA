import os
import sys
import random
from pathlib import Path
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as T
from tqdm import tqdm
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask
from tps.utils.resample2d_package.resample2d import Resample2d
from PIL import Image, ImageFilter
from tps.model.accel_deeplabv2 import get_accel_deeplab_v2
import math

from torchvision.transforms import ToPILImage
from PIL import Image

from advent.utils import transformmasks
from advent.utils import transformsgpu

from tps.utils.mix_augs import mix_operation

def gaussian_heatmap(x, prev):
        """
        It produces single gaussian at a random point
        """
        flag = True if np.random.rand() < 10.5 else False
        if flag:
            sig = torch.randint(low=1,high=150,size=(1,)).cuda()[0]
            data_min = torch.min(x)
            data_max = torch.max(x)
            scale = data_max - data_min
            x_norm = (x - torch.min(x))/(torch.max(x)-torch.min(x))
            image_size = x.shape[2:]
            rand_x = torch.randint(image_size[0],(1,))[0].cuda()
            rand_y = torch.randint(image_size[1],(1,))[0].cuda()
            center = (rand_x, rand_y)
            x_axis = torch.linspace(0, image_size[0]-1, image_size[0]).cuda() - center[0]
            y_axis = torch.linspace(0, image_size[1]-1, image_size[1]).cuda() - center[1]
            xx, yy = torch.meshgrid(x_axis, y_axis)
            kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig)).cuda()
            new_img = (x_norm.cuda()*(1-kernel) + 1*kernel)
            new_img = torch.clamp(new_img, 0.0, 1.0)
            x = new_img*scale + data_min

            prev_min = torch.min(x)
            prev_max = torch.max(x)
            scale = prev_max - prev_min
            prev_norm = (prev - torch.min(prev))/(torch.max(prev)-torch.min(prev))
            image_size = prev.shape[2:]
            rand_x = rand_x + 40 #- np.random.randint(40)
            rand_y = rand_y + 40 #- np.random.randint(40)
            rand_x = torch.clamp(rand_x, 0, image_size[0]-1)
            rand_y = torch.clamp(rand_y, 0, image_size[1]-1)

            center = (rand_x, rand_y)
            x_axis = torch.linspace(0, image_size[0]-1, image_size[0]).cuda() - center[0]
            y_axis = torch.linspace(0, image_size[1]-1, image_size[1]).cuda() - center[1]
            xx, yy = torch.meshgrid(x_axis, y_axis)
            kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig)).cuda()
            new_img = (prev_norm.cuda()*(1-kernel) + 1*kernel)
            new_img = torch.clamp(new_img, 0.0, 1.0)
            prev = new_img*scale + prev_min

        return x, prev

def gamma_aug(data, prev):
    flag = True if np.random.rand() < 10.5 else False
    if flag:
        data_min = torch.min(data)
        data_max = torch.max(data)
        scale = data_max - data_min
        data_norm = (data - torch.min(data))/(torch.max(data)-torch.min(data))
        '''data_tmp = data_norm.squeeze().to('cpu', torch.uint8)
        image = ToPILImage()(data_tmp*255)

        # Save the image
        image.save('clean_img.png')'''
        ratio = (np.random.rand()*0.8+0.2)
        val = 1/0.2#ratio
        data_norm = T.functional.adjust_gamma(data_norm,val)
        data = data_norm*scale + data_min

        ratio = ratio + (np.random.rand() - 0.5) * 0.2
        prev_min = torch.min(prev)
        prev_max = torch.max(prev)
        scale = prev_max - prev_min
        prev_norm = (prev - torch.min(prev))/(torch.max(prev)-torch.min(prev))

        val = 1/0.3#ratio
        prev_norm = T.functional.adjust_gamma(prev_norm,val)
        prev = prev_norm*scale + prev_min
    return data, prev

def chromaticity_aug(data, prev):
    flag = True if np.random.rand() < 10.5 else False
    if flag:
        data_min = torch.min(data)
        data_max = torch.max(data)
        scale = data_max - data_min
        data_norm = (data - torch.min(data))/(torch.max(data)-torch.min(data))
        '''data_tmp = data_norm.squeeze().to('cpu', torch.uint8)
        image = ToPILImage()(data_tmp*255)

        # Save the image
        image.save('clean_img.png')'''
        mask, y1, y2, x1, x2 = generate_random_rectangle_mask(data_norm)
        scale_f = 0.3#np.random.rand() * 0.8
        data_norm = data_norm + scale_f * mask
        data_norm = torch.clamp(data_norm, 0.0, 1.0)
        data = data_norm*scale + data_min

        prev_min = torch.min(prev)
        prev_max = torch.max(prev)
        scale = prev_max - prev_min
        prev_norm = (prev - torch.min(prev))/(torch.max(prev)-torch.min(prev))
        '''prev_tmp = prev_norm.squeeze().to('cpu', torch.uint8)
        image = ToPILImage()(prev_tmp*255)

        # Save the image
        image.save('clean_img.png')'''
        scale_f = 0.2#np.random.rand() * 0.8
        prev_norm = prev_norm + scale_f * mask
        prev_norm = torch.clamp(prev_norm, 0.0, 1.0)
        prev = prev_norm*scale + prev_min
    return data, prev

def generate_random_rectangle_mask(tensor):
    mask = torch.zeros_like(tensor)
    height, width = tensor.size()[-2:]

    min_size = min(height, width) // 2
    max_size = min(height, width) // 2

    x1 = random.randint(0, width - max_size)
    y1 = random.randint(0, height - max_size)
    x2 = random.randint(x1 + min_size, min(x1 + max_size, width))
    y2 = random.randint(y1 + min_size, min(y1 + max_size, height))

    mask[..., y1:y2, x1:x2] = 1

    return mask, y1, y2, x1, x2


def update_variance(labels, pred1, pred2, class_weight=None):
    criterion = nn.CrossEntropyLoss(weight = class_weight, ignore_index=255, reduction = 'none')
    kl_distance = nn.KLDivLoss( reduction = 'none')
    loss = criterion(pred1, labels)
    sm = torch.nn.Softmax(dim = 1)
    log_sm = torch.nn.LogSoftmax(dim = 1)

    #n, h, w = labels.shape
    #labels_onehot = torch.zeros(n, self.num_classes, h, w)
    #labels_onehot = labels_onehot.cuda()
    #labels_onehot.scatter_(1, labels.view(n,1,h,w), 1)

    variance = torch.sum(kl_distance(log_sm(pred1),(pred2)), dim=1)
    exp_variance = torch.exp(-variance)
    #variance = torch.log( 1 + (torch.mean((pred1-pred2)**2, dim=1)))
    #torch.mean( kl_distance(self.log_sm(pred1),pred2), dim=1) + 1e-6
    #loss = torch.mean(loss/variance) + torch.mean(variance)
    loss = torch.mean(loss*exp_variance) + torch.mean(variance)
    return loss

def gaussian_noise(mean, std, data=None):
    if not (data is None):
        if data.shape[1] == 3:
            gaussiannoise_flag = True if np.random.rand() < 10.5 else False
            if gaussiannoise_flag:
                data_min = torch.min(data)
                data_max = torch.max(data)
                scale = data_max - data_min
                data_norm = (data - torch.min(data))/(torch.max(data)-torch.min(data))
                '''data_tmp = data_norm.squeeze().to('cpu', torch.uint8)
                image = ToPILImage()(data_tmp*255)

                # Save the image
                image.save('clean_img.png')'''

                noise = torch.from_numpy(np.random.normal(0, 0.1, data.shape)).float()
                data_norm = data_norm.cpu() + noise
                data_norm = torch.clamp(data_norm, 0.0, 1.0)
                data = data_norm*scale.cpu() + data_min.cpu()
    return data.cuda()



def denorm_(img, mean, std, scale):
    img.mul_(std).add_(mean).div_(scale)


def renorm_(img, mean, std, scale):
    img.mul_(scale).sub_(mean).div_(std)

def augmentationTransform(parameters, data=None, target=None, aux_labels=None, probs=None, ignore_label=255):
    """
    Args:
        parameters: dictionary with the augmentation configuration
        data: BxCxWxH input data to augment
        target: BxWxH labels to augment
        probs: BxWxH probability map to augment
        jitter_vale:  jitter augmentation value
        min_sigma: min sigma value for blur
        max_sigma: max sigma value for blur
        ignore_label: value for ignore class
    Returns:
            augmented data, target, probs
    """
    assert ((data is not None) or (target is not None))
    if "Mix" in parameters:
        data, target, aux_labels, probs = transformsgpu.mix(mask=parameters["Mix"], data=data, target=target, aux_labels=aux_labels, probs=probs)

    return data, target, aux_labels, probs


def augment_samples(images, labels, aux_labels, probs, do_classmix, batch_size, ignore_label, weak = False, classes = None):
    """
    Perform data augmentation
    Args:
        images: BxCxWxH images to augment
        labels:  BxWxH labels to augment
        probs:  BxWxH probability maps to augment
        do_classmix: whether to apply classmix augmentation
        batch_size: batch size
        ignore_label: ignore class value
        weak: whether to perform weak or strong augmentation
    Returns:
        augmented data, augmented labels, augmented probs
    """

    # ClassMix: Get mask for image A
    for image_i in range(batch_size):  # for each image
        classes = torch.unique(labels[image_i])  # get unique classes in pseudolabel A
        nclasses = classes.shape[0]

        # remove ignore class
        if ignore_label in classes and len(classes) > 1 and nclasses > 1:
            classes = classes[classes != ignore_label]
            nclasses = nclasses - 1

        # pick half of the classes randomly
        if classes is None:
            classes = (classes[torch.Tensor(
                np.random.choice(nclasses, int(((nclasses - nclasses % 2) / 2) + 1), replace=False)).long()]).cuda()
        else:
            pass

        # acumulate masks
        if image_i == 0:
            MixMask = transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()
        else:
            MixMask = torch.cat(
                (MixMask, transformmasks.generate_class_mask(labels[image_i], classes).unsqueeze(0).cuda()))


    params = {"Mix": MixMask}
    image_aug, labels_aug, aux_labels_aug, probs_aug = augmentationTransform(params,
                                                             data=images, target=labels,
                                                             aux_labels=aux_labels,
                                                             probs=probs, ignore_label=ignore_label)

    return image_aug, labels_aug, aux_labels_aug, probs_aug, params, classes


def create_ema_model(model, net_class, cfg, hasgrad=False):
    """
    Args:
        model: segmentation model to copy parameters from
        net_class: segmentation model class
    Returns: Segmentation model from [net_class] with same parameters than [model]
    """
    ema_model = net_class(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)

    if hasgrad:
        pass
    else:
        for param in ema_model.parameters():
            param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()

    return ema_model

def restore_ema_model(net_class, cfg, path=None):
    """
    Args:
        model: segmentation model to copy parameters from
        net_class: segmentation model class
    Returns: Segmentation model from [net_class] with same parameters than [model]
    """
    ema_model = net_class(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
    saved_state_dict = torch.load(path)
    ema_model.load_state_dict(saved_state_dict)

    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    """
    Args:
        ema_model: model to update
        model: model from which to update parameters
        alpha_teacher: value for weighting the ema_model
        iteration: current iteration
    Returns: ema_model, with parameters updated follwoing the exponential moving average of [model]
    """
    # Use the "true" average until the exponential average is more correct
    #alpha_teacher = min(1 - 1 / (iteration*10 + 1), alpha_teacher)
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if not param.data.shape:  # scalar tensor
            ema_param.data = \
                alpha_teacher * ema_param.data + \
                (1 - alpha_teacher) * param.data
        else:
            ema_param.data[:] = \
                alpha_teacher * ema_param[:].data[:] + \
                (1 - alpha_teacher) * param[:].data[:]
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        if not buffer.data.shape:  # scalar tensor
            ema_buffer.data = \
                alpha_teacher * ema_buffer.data + \
                (1 - alpha_teacher) * buffer.data
        else:
            ema_buffer.data[:] = \
                alpha_teacher * ema_buffer[:].data[:] + \
                (1 - alpha_teacher) * buffer[:].data[:]

    return ema_model

def train_domain_adaptation(model, img_model, source_loader, target_loader, cfg):
    if cfg.TRAIN.DA_METHOD == 'SourceOnly':
        train_TPS_srconly(model, img_model, source_loader, target_loader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'TPS':
        train_TPS(model, img_model, source_loader, target_loader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")

def rand_crop(img, crop_factor, hr=None):
    mask = torch.zeros_like(img)
    height, width = img.size()[-2:]

    width_size = round(width * crop_factor)
    height_size = round(height * crop_factor)

    x1 = random.randint(0, (width - width_size)/8)*8
    y1 = random.randint(0, (height - height_size)/8)*8
    x2 = x1 + width_size
    y2 = y1 + height_size
    mask[..., y1:y2, x1:x2] = 1
    if hr is not None:
        return mask, y1,y2,x1,x2

    return mask

def train_TPS(model, img_model, source_loader, target_loader, cfg):
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # TEACHER
    ema_model = create_ema_model(model, get_accel_deeplab_v2, cfg)
    ema_model.train()
    ema_model.to(device)




    # OPTIMIZERS
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(512, 1024), mode='bilinear',
                                align_corners=True)

    interp_input = nn.Upsample(size=(512, 1024), mode='bilinear',align_corners=True)
    interp_hr = nn.Upsample(size=(1024, 2048), mode='bilinear',align_corners=True)
    # propagate predictions (of previous frames) forward
    warp_bilinear = Resample2d(bilinear=True)
    #
    source_loader_iter = enumerate(source_loader)
    target_loader_iter = enumerate(target_loader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):

        ####  optimizer  ####
        optimizer.zero_grad()

        ####  adjust LR  ####
        adjust_learning_rate(optimizer, i_iter, cfg)

        ####  load data  ####
        _, source_batch = source_loader_iter.__next__()

        src_img_cf, src_label, src_img_kf, src_label_kf, src_img_nf, src_label_nf, _, src_img_name, src_cf, src_kf, src_nf = source_batch

        _, target_batch = target_loader_iter.__next__()
        trg_img_d_ori, trg_img_c_ori, trg_img_b_ori, trg_img_a_ori, d_ori,  _, name, frames = target_batch

        trg_img_d = interp_input(trg_img_d_ori)
        trg_img_c = interp_input(trg_img_c_ori)
        trg_img_b = interp_input(trg_img_b_ori)
        trg_img_a = interp_input(trg_img_a_ori)
        d = interp_input(d_ori)

        if cfg.TARGET != 'MVSS':
            frames = frames.squeeze().tolist()

        ##  match
        src_cf = hist_match(src_cf, d)
        src_kf = hist_match(src_kf, d)
        src_nf = hist_match(src_nf, d)
        ##  normalize
        src_cf = torch.flip(src_cf, [1])
        src_kf = torch.flip(src_kf, [1])
        src_nf = torch.flip(src_nf, [1])
        src_cf -= torch.tensor(cfg.TRAIN.IMG_MEAN).view(1, 3, 1, 1)
        src_kf -= torch.tensor(cfg.TRAIN.IMG_MEAN).view(1, 3, 1, 1)
        src_nf -= torch.tensor(cfg.TRAIN.IMG_MEAN).view(1, 3, 1, 1)
        ##  recover
        src_img_cf = src_cf
        src_img_kf = src_kf
        src_img_nf = src_nf

        ####  supervised | source  ####
        if src_label.dim() == 4:
            src_label = src_label.squeeze(-1)
        file_name = src_img_name[0].split('/')[-1]
        '''if cfg.SOURCE == 'Viper':
            frame = int(file_name.replace('.jpg', '')[-5:])
            frame1 = frame - 1
            flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
        elif cfg.SOURCE == 'SynthiaSeq':
            flow_int16_x10_name = file_name.replace('.png', '_int16_x10')
        flow_int16_x10 = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        src_flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)'''
        #if i_iter < 20000:
        cf_max = torch.max(src_img_cf)
        cf_min = torch.min(src_img_cf)
        norm_src_img_cf = (src_img_cf - cf_min) / (cf_max - cf_min)

        norm_src_img_cf_mask = rand_crop(norm_src_img_cf, 0.5)
        norm_src_img_cf_masked = norm_src_img_cf * norm_src_img_cf_mask


        cropped_src_img_cf = (norm_src_img_cf_masked - torch.min(norm_src_img_cf_masked)) / (torch.max(norm_src_img_cf_masked) - torch.min(norm_src_img_cf_masked)) * (cf_max - cf_min) + cf_min

        src_img_cf_crop = cropped_src_img_cf

        src_pred_cf_aux, src_pred_cf = model(src_img_cf.cuda(device), src_img_kf.cuda(device), src_img_nf.cuda(device), device, imgonly=True)
        src_pred_aux, src_pred, _, _, _, _ = model(src_img_cf_crop.cuda(device), src_img_kf.cuda(device), src_img_nf.cuda(device), device, warp=True)
        src_pred = interp_source(src_pred)
        src_pred_cf = interp_source(src_pred_cf)
        loss_seg_src_main = loss_calc(src_pred, src_label, device)
        loss_seg_src_cf = loss_calc(src_pred_cf, src_label, device)
        if cfg.TRAIN.MULTI_LEVEL:
            src_pred_aux = interp_source(src_pred_aux)
            loss_seg_src_aux = loss_calc(src_pred_aux, src_label, device)
            src_pred_cf_aux = interp_source(src_pred_cf_aux)
            loss_seg_src_cf_aux = loss_calc(src_pred_cf_aux, src_label, device)
        else:
            loss_seg_src_aux = 0
            loss_seg_src_cf_aux = 0

        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * (loss_seg_src_main + loss_seg_src_cf) + cfg.TRAIN.LAMBDA_SEG_AUX * (loss_seg_src_aux + loss_seg_src_cf_aux)
        loss.backward()
        del src_pred_aux, src_pred, src_pred_cf_aux, src_pred_cf#, src_pred_kf_aux, src_pred_kf

        ####  unsupervised | target  ####
        ##  optical flow  ##
        '''
            {d, c} or {b, a}: pair of consecutive frames extracted from the same video
        '''
        file_name = name[0].split('/')[-1]


        ##  augmentation  ##
        # flip {b, a}
        flip = random.random() < -1
        if flip:
            trg_img_b_wk = torch.flip(trg_img_b, [3])
            trg_img_d_wk = torch.flip(trg_img_d, [3])
            trg_img_a_wk = torch.flip(trg_img_a, [3])
            trg_img_c_wk = torch.flip(trg_img_c, [3])
            #trg_flow_b_wk = torch.flip(trg_flow_b, [3])
            #trg_flow_d_wk = torch.flip(trg_flow_d, [3])
        else:
            trg_img_b_wk = trg_img_b
            trg_img_d_wk = trg_img_d
            trg_img_a_wk = trg_img_a
            trg_img_c_wk = trg_img_c
            #trg_flow_b_wk = trg_flow_b
            #trg_flow_d_wk = trg_flow_d


        ##  Temporal Pseudo Supervision  ##
        # Cross Frame Pseudo Label
        if True: # For testing different settings
            with torch.no_grad():
                #trg_pred_aux, trg_pred, _, _, _, _ = model(trg_img_b_wk.cuda(device), trg_img_a_wk.cuda(device), trg_flow_b_wk, device)
                trg_pred_aux, trg_pred, trg_cf_aux, trg_cf, trg_kf_aux, trg_kf = ema_model(trg_img_d_wk.cuda(device), trg_img_c_wk.cuda(device), trg_img_b_wk.cuda(device), device, warp=True)
                #trg_pred_aux_std, trg_pred_std, _, _, _, _ = ema_model(trg_img_d_wk.cuda(device), trg_img_c_wk.cuda(device), trg_img_b_wk.cuda(device), device, warp=True, prov_cf=std_cf, prov_cf_aux=std_cf_aux, prov_kf=std_kf, prov_kf_aux=std_kf_aux)


                unlabeled_weight = torch.sum(trg_pred.ge(0.968).long() == 1).item() / np.size(np.array(trg_pred.cpu()))
                unlabeled_weight_aux = torch.sum(trg_pred_aux.ge(0.968).long() == 1).item() / np.size(np.array(trg_pred_aux.cpu()))

                # softmax
                trg_prob = F.softmax(trg_pred, dim=1)
                trg_prob_aux = F.softmax(trg_pred_aux, dim=1)
                trg_prob_cf = F.softmax(trg_cf, dim=1)
                trg_prob_aux_cf = F.softmax(trg_cf_aux, dim=1)
                # pseudo label
                trg_pl = torch.argmax(trg_prob, 1)
                trg_pl_aux = torch.argmax(trg_prob_aux, 1)
                trg_pl_cf = torch.argmax(trg_prob_cf, 1)
                trg_pl_aux_cf = torch.argmax(trg_prob_aux_cf, 1)

                # softmax
                #trg_pl = torch.argmax(trg_prob_warp, 1)
                #trg_pl_aux = torch.argmax(trg_prob_warp_aux, 1)
                if flip:
                    trg_pl = torch.flip(trg_pl, [2])
                    trg_pl_aux = torch.flip(trg_pl_aux, [2])
                    trg_pl_cf = torch.flip(trg_pl_cf, [2])
                    trg_pl_aux_cf = torch.flip(trg_pl_aux_cf, [2])

                    #trg_pl_img = torch.flip(trg_pl_img, [2])
                    #trg_pl_aux_img = torch.flip(trg_pl_aux_img, [2])
                    trg_cf_aux = torch.flip(trg_cf_aux, [2])
                    trg_cf = torch.flip(trg_cf, [2])

                # rescale param
                trg_interp_sc2ori = nn.Upsample(size=(trg_pred.shape[-2], trg_pred.shape[-1]), mode='bilinear', align_corners=True)


            # concatenate {d, c}
            trg_img_concat = torch.cat((trg_img_d.cpu(), trg_img_c.cpu(), trg_img_b.cpu()), 2)
            # strong augment {d, c}
            aug = T.Compose([
                T.ToPILImage(),
                T.RandomApply([GaussianBlur(radius=random.choice([5, 7, 9]))], p=0.6),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor()
            ])
            trg_img_concat_st = aug(torch.squeeze(trg_img_concat)).unsqueeze(dim=0)
            # seperate {d, c}
            trg_img_d_st = trg_img_concat_st[:, :, 0:512, :]
            trg_img_c_st = trg_img_concat_st[:, :, 512:1024, :]
            trg_img_b_st = trg_img_concat_st[:, :, 1024:, :]
            # rescale {d, c}
            scale_ratio = np.random.randint(100.0 * cfg.TRAIN.SCALING_RATIO[0], 100.0 * cfg.TRAIN.SCALING_RATIO[1]) / 100.0
            trg_scaled_size = (round(512 * scale_ratio / 8) * 8, round(1024 * scale_ratio / 8) * 8)
            trg_interp_sc = nn.Upsample(size=trg_scaled_size, mode='bilinear', align_corners=True)
            trg_img_d_st = trg_interp_sc(trg_img_d_st)
            trg_img_c_st = trg_interp_sc(trg_img_c_st)
            trg_img_b_st = trg_interp_sc(trg_img_b_st)

            # forward prop
            trg_pred_aux, trg_pred, _, _, _, _ = model(trg_img_d_st.cuda(device), trg_img_c_st.cuda(device), trg_img_b_st.cuda(device), device, warp=True)
            # rescale
            trg_pred = trg_interp_sc2ori(trg_pred)
            trg_pred_aux = trg_interp_sc2ori(trg_pred_aux)

            # unsupervised loss
            loss_trg = loss_calc(trg_pred, trg_pl, device)

            if cfg.TRAIN.MULTI_LEVEL:
                loss_trg_aux = loss_calc(trg_pred_aux, trg_pl_aux, device)
            else:
                loss_trg_aux = 0

            # Standard Teacher-Student part
            loss = cfg.TRAIN.LAMBDA_T * (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_trg * unlabeled_weight + cfg.TRAIN.LAMBDA_SEG_AUX * loss_trg_aux * unlabeled_weight_aux)

            del trg_img_d_st, trg_img_c_st, trg_img_b_st

            # randomly erase information from image d


            # Temporal Teacher-Student Starts Here
            d_max = torch.max(trg_img_d)
            d_min = torch.min(trg_img_d)
            norm_trg_img_d = (trg_img_d - d_min) / (d_max - d_min)

            norm_trg_img_d_mask,y1,y2,x1,x2 = rand_crop(norm_trg_img_d, 0.5, hr=True)
            trg_img_d_hr = trg_img_d_ori[:,:,y1*2:y2*2,x1*2:x2*2]
            hr_mask = torch.zeros_like(trg_img_d_ori)
            hr_mask[:,:,y1*2:y2*2,x1*2:x2*2] = 1
            hr_mask = hr_mask[:,1,:,:]

            norm_trg_img_d_masked = norm_trg_img_d * norm_trg_img_d_mask

            c_max = torch.max(trg_img_c)
            c_min = torch.min(trg_img_c)
            norm_trg_img_c = (trg_img_c - c_min) / (c_max - c_min)
            norm_trg_img_c_mask = rand_crop(norm_trg_img_c, 0.5)
            norm_trg_img_c_masked = norm_trg_img_c * (1 - norm_trg_img_c_mask)

            b_max = torch.max(trg_img_b)
            b_min = torch.min(trg_img_b)
            norm_trg_img_b = (trg_img_b - b_min) / (b_max - b_min)
            norm_trg_img_b_mask = rand_crop(norm_trg_img_b, 0.5)
            norm_trg_img_b_masked = norm_trg_img_b * (1 - norm_trg_img_b_mask)


            cropped_trg_img_d = (norm_trg_img_d_masked - torch.min(norm_trg_img_d_masked)) / (torch.max(norm_trg_img_d_masked) - torch.min(norm_trg_img_d_masked)) * (d_max - d_min) + d_min
            trg_img_c_crop = (norm_trg_img_c_masked - torch.min(norm_trg_img_c_masked)) / (torch.max(norm_trg_img_c_masked) - torch.min(norm_trg_img_c_masked)) * (c_max - c_min) + c_min
            trg_img_b_crop = (norm_trg_img_b_masked - torch.min(norm_trg_img_b_masked)) / (torch.max(norm_trg_img_b_masked) - torch.min(norm_trg_img_b_masked)) * (b_max - b_min) + b_min

            trg_img_d_crop = cropped_trg_img_d

            trg_img_d_crop = trg_interp_sc(trg_img_d_crop)
            trg_img_c_interp = trg_interp_sc(trg_img_c)
            trg_img_b_interp = trg_interp_sc(trg_img_b)

            with torch.no_grad():
                trg_prob_cf = F.softmax(trg_cf, dim=1)
                trg_prob_cf_aux = F.softmax(trg_cf_aux, dim=1)
                trg_pl_cf = torch.argmax(trg_prob_cf, 1)
                trg_pl_cf_aux = torch.argmax(trg_prob_cf_aux, 1)
                crop_interp = nn.Upsample(size=(trg_pl.shape[-2], trg_pl.shape[-1]), mode='bilinear', align_corners=True)
                crop_hr_interp = nn.Upsample(size=(32,64), mode='bilinear', align_corners=True)
                norm_trg_img_d_mask = crop_interp(norm_trg_img_d_mask)
                norm_trg_img_d_mask = norm_trg_img_d_mask[:,0,:,:].cuda()
                norm_trg_img_d_mask = (1 - norm_trg_img_d_mask).int()*255

                trg_pl_cropped = torch.max(norm_trg_img_d_mask, trg_pl_cf)

                trg_pl_aux_cropped = torch.max(norm_trg_img_d_mask, trg_pl_cf_aux)

                # rescale param
                trg_interp_sc2ori = nn.Upsample(size=(trg_pred.shape[-2], trg_pred.shape[-1]), mode='bilinear', align_corners=True)


            # forward prop
            trg_pred_aux, trg_pred, cf_aux, cf, _, _ = model(trg_img_d_crop.cuda(device), trg_img_c_interp.cuda(device), trg_img_b_interp.cuda(device), device, warp=True)



            # Spatial Teacher-Student Starts Here
            with torch.no_grad():
                cf_aux_hr, cf_hr = ema_model(trg_img_d_hr.cuda(device), trg_img_c_wk.cuda(device), trg_img_b_wk.cuda(device), device, imgonly=True)
            # rescale
            trg_pred = trg_interp_sc2ori(trg_pred)
            trg_pred_aux = trg_interp_sc2ori(trg_pred_aux)
            cf = trg_interp_sc2ori(cf)
            cf_aux = trg_interp_sc2ori(cf_aux)

            cf_prob_hr = F.softmax(cf_hr, dim=1)
            cf_prob_aux_hr = F.softmax(cf_aux_hr, dim=1)
            # pseudo label
            cf_pl_hr = torch.argmax(cf_prob_hr, 1)
            cf_pl_aux_hr = torch.argmax(cf_prob_aux_hr, 1)

            cf_hr = crop_hr_interp(cf_pl_hr.unsqueeze(0).float())
            cf_hr = cf_hr.squeeze(0)
            cf_aux_hr = crop_hr_interp(cf_pl_aux_hr.unsqueeze(0).float())
            cf_aux_hr = cf_aux_hr.squeeze(0)

            cf_hr_full = torch.zeros_like(cf_pl_hr)
            cf_hr_aux_full = torch.zeros_like(cf_pl_aux_hr)

            hr_y1 = int(y1/8)
            hr_y2 = hr_y1+32

            hr_x1 = int(x1/8)
            hr_x2 = hr_x1+64

            cf_hr_full[:, hr_y1:hr_y2, hr_x1:hr_x2] = cf_hr
            cf_hr_aux_full[:,hr_y1:hr_y2,hr_x1:hr_x2] = cf_aux_hr

            cf_hr_full = crop_interp(cf_hr_full.unsqueeze(0).float())
            cf_hr_full = cf_hr_full.squeeze(0)
            cf_hr_pl = torch.max(norm_trg_img_d_mask, cf_hr_full)

            cf_hr_aux_full = crop_interp(cf_hr_aux_full.unsqueeze(0).float())
            cf_hr_aux_full = cf_hr_aux_full.squeeze(0)
            cf_hr_aux_pl = torch.max(norm_trg_img_d_mask, cf_hr_aux_full)

            #cf = trg_interp_sc2ori(cf)
            #cf_aux = trg_interp_sc2ori(cf_aux)
            # unsupervised loss
            loss_crop_trg = loss_calc(trg_pred, trg_pl, device)
            #loss_crop_trg = loss_calc(trg_pred, trg_pl_img, device)
            loss_crop_trg_cst = loss_calc(cf, cf_hr_pl, device)
            #loss_cst = loss_calc(trg_pred, trg_pl_cst, device)
            if cfg.TRAIN.MULTI_LEVEL:
                loss_trg_crop_aux = loss_calc(trg_pred_aux, trg_pl_aux, device)
                #loss_trg_crop_aux = loss_calc(trg_pred_aux, trg_pl_aux_img, device)
                loss_trg_crop_aux_cst = loss_calc(cf_aux, cf_hr_aux_pl, device)
                #loss_aux_cst = loss_calc(trg_pred_aux_cst, trg_pl_aux_cst, device)
            else:
                loss_trg_crop_aux = 0
                loss_trg_crop_aux_cst = 0

            loss += cfg.TRAIN.LAMBDA_T * (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_crop_trg * unlabeled_weight + cfg.TRAIN.LAMBDA_SEG_AUX * loss_trg_crop_aux * unlabeled_weight_aux)
            loss += cfg.TRAIN.LAMBDA_T * (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_crop_trg_cst * unlabeled_weight + cfg.TRAIN.LAMBDA_SEG_AUX * loss_trg_crop_aux_cst * unlabeled_weight_aux)
            #loss += cfg.TRAIN.LAMBDA_T * (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_cst * unlabeled_weight_cst + cfg.TRAIN.LAMBDA_SEG_AUX * loss_aux_cst * unlabeled_weight_aux_cst)
            loss.backward()
        else:
            loss_seg_src_cf = 0
            loss_seg_src_cf_aux = 0
            loss_trg = 0
            loss_trg_aux = 0
            loss_crop_trg = 0
            loss_trg_crop_aux = 0
            loss_crop_trg_cst = 0
            loss_trg_crop_aux_cst = 0
        ####  step  ####
        optimizer.step()
        #optimizer_img.step()

        #m = 1 - (1 - 0.995) * (math.cos(math.pi * i_iter / cfg.TRAIN.EARLY_STOP) + 1) / 2

        m = 0.995
        if i_iter > 4000:
            ema_model = update_ema_variables(ema_model=ema_model, model=model, alpha_teacher=m, iteration=i_iter)
        #img_ema_model = update_ema_variables(ema_model=img_ema_model, model=img_model, alpha_teacher=m, iteration=i_iter)

        ####  logging  ####
        if cfg.TRAIN.MULTI_LEVEL:
            current_losses = {'loss_src': loss_seg_src_main,
                              'loss_src_aux': loss_seg_src_aux,
                              'loss_src_cf': loss_seg_src_cf,
                              'loss_src_cf_aux': loss_seg_src_cf_aux,
                              'loss_trg': loss_trg,
                              'loss_trg_aux': loss_trg_aux,
                              #'loss_trg_mix': loss_trg_mix,
                              #'loss_trg_aux_mix': loss_trg_aux_mix,
                              #'loss_src_img': loss_seg_src_main_img,
                              #'loss_src_aux_img': loss_seg_src_aux_img,
                              #'loss_trg_img': loss_trg_img,
                              #'loss_trg_aux_img': loss_trg_aux_img,
                              #'loss_img_model': loss_img_model,
                              #'loss_aux_img_model': loss_aux_img_model,
                              #'loss_cst': loss_cst,
                              #'loss_aux_cst': loss_aux_cst,
                              'loss_crop_trg': loss_crop_trg,
                              'loss_trg_crop_aux': loss_trg_crop_aux,
                              'loss_crop_trg_cst': loss_crop_trg_cst,
                              'loss_trg_crop_aux_cst': loss_trg_crop_aux_cst
                             }
        else:
            current_losses = {'loss_src': loss_seg_src_main,
                              'loss_trg': loss_trg,
                              'loss_crop_trg': loss_crop_trg,
                              'loss_crop_trg_cst': loss_crop_trg_cst
                             }
        print_losses(current_losses, i_iter)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            torch.save(ema_model.state_dict(), snapshot_dir / f'ema_model_{i_iter}.pth')
            #torch.save(img_model.state_dict(), snapshot_dir / f'img_model_{i_iter}.pth')
            #torch.save(img_ema_model.state_dict(), snapshot_dir / f'img_ema_model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

        torch.cuda.empty_cache()

## utils
def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def hist_match(img_src, img_trg):
    import skimage
    from skimage import exposure
    img_src = np.asarray(img_src.squeeze(0).transpose(0, 1).transpose(1, 2), np.float32)
    img_trg = np.asarray(img_trg.squeeze(0).transpose(0, 1).transpose(1, 2), np.float32)
    images_aug = exposure.match_histograms(img_src, img_trg, multichannel=True)
    return torch.from_numpy(images_aug).transpose(1, 2).transpose(0, 1).unsqueeze(0)

class GaussianBlur(object):

    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class EMA(object):

    def __init__(self, model, alpha=0.999):
        """ Model exponential moving average. """
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        # NOTE: Buffer values are for things that are not parameters,
        # such as batch norm statistics
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = self.alpha
        state = self.model.state_dict()  # current params
        for name in self.param_keys:
            self.shadow[name].copy_(
                    decay * self.shadow[name] + (1 - decay) * state[name])
        self.step += 1

    def update_buffer(self):
        # No EMA for buffer values (for now)
        state = self.model.state_dict()
        for name in self.buffer_keys:
            self.shadow[name].copy_(state[name])

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }
