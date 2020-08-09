from __future__ import division

import models_bit
from models import *
#from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
   
class Bit_yolov3(torch.nn.Module):
    def __init__(self, darknet_model, Bit_model, freeze_some=False):
        super(Bit_yolov3, self).__init__()

        self.bit_model = Bit_model
        
        if freeze_some:
            for parameter in self.bit_model.root.parameters():
                parameter.requires_grad = False
            for parameter in self.bit_model.body.block1.parameters():
                parameter.requires_grad = False

        self.yolo_layers    = darknet_model.yolo_layers

        self.mapping_layer1 = nn.Sequential(
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=512, out_channels=18, kernel_size=(1,1)))   # 512 to 18

        self.mapping_layer2 = nn.Sequential(
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=1024, out_channels=18, kernel_size=(1,1))) # 1024 to 18

        self.mapping_layer3 = nn.Sequential(
                                nn.BatchNorm2d(2048),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=2048, out_channels=18, kernel_size=(1,1))) # 2048 to 18
        self.seen = 0
        
        
    def forward(self, x, targets=None):
       
        loss = 0 
        yolo_outputs = []
        img_dim = x.shape[2]
                 
        x = self.bit_model.root(x)
        x1 = self.bit_model.body.block1(x)
        x2 = self.bit_model.body.block2(x1)
        x3 = self.bit_model.body.block3(x2)
        x4 = self.bit_model.body.block4(x3)

        x2 = self.mapping_layer1(x2)
        out1, layer_loss = self.yolo_layers[0](x2, targets, img_dim)
        loss += layer_loss
        yolo_outputs.append(out1)
        x3 = self.mapping_layer2(x3)
        out2, layer_loss = self.yolo_layers[1](x3, targets, img_dim)
        loss += layer_loss
        yolo_outputs.append(out2)
        x4 = self.mapping_layer3(x4)
        out3, layer_loss = self.yolo_layers[2](x4, targets, img_dim)
        loss += layer_loss
        yolo_outputs.append(out3)

        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

def adjust_learning_rate(lr, optimizer, steps, epoch, gamma):
    if epoch in steps:
        lr = lr*gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return lr

best_map = -1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--freeze_some", default=False, help="if True, freezes conv1 bn1 and layer1")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--bit_model_type", type=str)
    opt = parser.parse_args()
    print(opt)

    #logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs(opt.checkpoint_path, exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])


    bit_model = models_bit.KNOWN_MODELS[opt.bit_model_type](head_size=2, zero_head=True)
    bit_model.load_from(np.load(f"{opt.bit_model_type}.npz"))

    bit_model = bit_model.to(device)
    # Initiate model
    darknet_model = Darknet(opt.model_def).to(device)
    #model.apply(weights_init_normal)
    model = Bit_yolov3(darknet_model, bit_model, freeze_some=opt.freeze_some).to(device)
    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = opt.learning_rate)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    lr = opt.learning_rate
    lr_steps = [30,60,90]
    

    losses = AverageMeter()
    logger = open(os.path.join(opt.checkpoint_path,'log.txt'), "w+")
    for epoch in range(opt.epochs):
        lr = adjust_learning_rate(lr, optimizer, lr_steps, epoch, gamma=0.1)
        model.train()

        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            losses.update(loss.item(), imgs.size(0))
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                #logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if 1:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.3,
                conf_thres=0.3,
                nms_thres=0.3,
                img_size=opt.img_size,
                batch_size=32,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            #logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        log_info = str({ "lr" : lr,
                        "map": AP.mean(),
                        "loss" : losses.avg})
        logger.write(log_info + "\n")
        #if epoch % opt.checkpoint_interval == 0:
        best = AP.mean() > best_map
        if best:
            best_map = AP.mean()
        if best:             
            torch.save({"state_dict" : model.state_dict(),
                        'best_map'   : AP.mean(),
                        'training_loss' : loss.item(),
                         'epoch' : epoch}, os.path.join(opt.checkpoint_path,"yolov3_ckpt.pth"))
