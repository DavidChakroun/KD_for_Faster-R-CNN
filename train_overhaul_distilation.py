# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import pdb
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #Permet de choisir la carte graphique qu'on veut utiliser

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list

from model.faster_rcnn.resnet_distil import resnet

from model.utils.net_utils import adjust_learning_rate, save_checkpoint, sampler
from model.utils.parser_func import distiller_option, set_dataset_args


def distillation_loss(source, target, margin):


  target = torch.max(target, margin)
  loss = torch.nn.functional.mse_loss(source, target, reduction="none")
  loss = loss * ((source > target) | (target > 0)).float()
  loss=loss.sum()
  return loss



if __name__ == '__main__':

  args = distiller_option()

  print('Called with args:')
  print(args)
  args = set_dataset_args(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  output_dir = "models/" + args.dataset + "/" + args.teacher + '/distillation'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.teacher == 'res101':
      fasterRCNN_prof = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
      numLayerProf=101
  elif args.teacher == 'res50':
      fasterRCNN_prof = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
      numLayerProf = 50
  elif args.teacher == 'res34':
      fasterRCNN_prof = resnet(imdb.classes, 34, pretrained=True, class_agnostic=args.class_agnostic)
      numLayerProf = 34
  elif args.teacher == 'res18':
      fasterRCNN_prof = resnet(imdb.classes, 18, pretrained=True, class_agnostic=args.class_agnostic)
      numLayerProf = 18

  fasterRCNN_prof.create_architecture()
  fasterRCNN_prof.eval()
  for key, value in dict(fasterRCNN_prof.named_parameters()).items():
    print(key)

  for param in fasterRCNN_prof.parameters():
      param.requires_grad = False

  for param in fasterRCNN_prof.RCNN_base.parameters():
      param.requires_grad = False

  for param in fasterRCNN_prof.RCNN_top.parameters():
      param.requires_grad = False


  if args.student == 'res101':
    fasterRCNN_etudiant = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic, etudiant=True, numLayerTeacher=numLayerProf)
  elif args.student == 'res50':
    fasterRCNN_etudiant = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, etudiant=True, numLayerTeacher=numLayerProf)
  elif args.student == 'res34':
    fasterRCNN_etudiant = resnet(imdb.classes, 34, pretrained=True, class_agnostic=args.class_agnostic, etudiant=True, numLayerTeacher=numLayerProf)
  elif args.student == 'res18':
    fasterRCNN_etudiant = resnet(imdb.classes, 18, pretrained=True, class_agnostic=args.class_agnostic, etudiant=True, numLayerTeacher=numLayerProf)


  fasterRCNN_etudiant.create_architecture()
  print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in fasterRCNN_prof.parameters()])))
  print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in fasterRCNN_etudiant.parameters()])))


  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr

  params = []
  for key, value in dict(fasterRCNN_etudiant.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.cuda:
    fasterRCNN_etudiant.cuda()
    fasterRCNN_prof.cuda()
      
  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)


  # load_prof = "models/kitti_car/res101/faster/kitti_car_faster_rcnn_12_5_7477.pth"
  load_prof ="models/pascal_voc/res101/faster/pascal_voc_faster_rcnn_12_6_16435.pth"
  print("loading checkpoint %s" % (load_prof))
  checkpoint = torch.load(load_prof)
  fasterRCNN_prof.load_state_dict(checkpoint['model'])

  # pretrained_dict = torch.load(load_prof)
  #
  # model_dict = fasterRCNN_prof.state_dict()
  # # 1. filter out unnecessary keys
  # pretrained_dict['model'] = {k: v for k, v in pretrained_dict['model'].items() if k in model_dict}
  # # 2. overwrite entries in the existing state dict
  # model_dict.update(pretrained_dict['model'])
  # # 3. load the new state dict
  # fasterRCNN_prof.load_state_dict(model_dict)

  # checkpoint = torch.load(load_prof)
  # fasterRCNN_prof.load_state_dict(checkpoint['model'])

  if args.resume:
    load_name = "models/pascal_voc/res101/distillation/pascal_voc_faster_rcnn_3_3_16435.pth"
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN_etudiant.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']

    print("loaded checkpoint %s" % (load_name))


  iters_per_epoch = int(train_size / args.batch_size)


  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN_etudiant.train()
    loss_temp = 0
    start = time.time()

    if epoch - 1 in  args.lr_decay_step:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch ):
      try:
        data = next(data_iter)
      except:
        data_iter = iter(dataloader)
        data = next(data_iter)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])

              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])

      fasterRCNN_etudiant.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, etudiant_caracteristiques_transformees, mask_batch = fasterRCNN_etudiant(im_data, im_info, gt_boxes, num_boxes)

      #professeur
      with torch.no_grad():
        rois_prof, cls_prob_prof, bbox_pred_prof, rpn_loss_cls_prof, rpn_loss_box_prof, \
        RCNN_loss_cls_prof, RCNN_loss_bbox_prof, \
        rois_label_prof, prof_caracteristique,mask_batch_prof = fasterRCNN_prof(im_data, im_info, gt_boxes, num_boxes)
      loss_distill=0
      for i in range(4):
        loss_distill += distillation_loss(etudiant_caracteristiques_transformees[i], prof_caracteristique[i].detach(), getattr(fasterRCNN_prof, 'margin%d' % (i+1))) \
                        / 2 ** (4 - i - 1)
      loss_distil=loss_distill/100000
      if epoch ==1 and step<400:
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
      else:
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
             + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() +loss_distil


      loss_temp += loss.item()
      if loss != loss:
          raise Exception('')
      # backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, distil %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_distil))



    save_name = os.path.join(output_dir,
                             '{}_{}_distillation_s_{}_ep_{}_st_{}_alpha_{}.pth'.format(args.student, args.dataset,
                                                                                       args.session, epoch, step,
                                                                                       args.alpha))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN_etudiant.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))
