import os
import datetime

import torch

import transforms
from model.fasterrcnn.network.rpn import AnchorsGenerator

from model import FasterRCNN, FastRCNNPredictor, EnhancedFasterRCNN
from model import resnet50_backbone, resnet50_fpn_backbone

from dataset.my_dataset import AITODv2Dataset
from utils import GroupedBatchSampler, create_aspect_ratio_groups
from utils import train_eval_utils as utils


def collate_fn(batch):
    return tuple(zip(*batch))

def create_model(num_classes, load_pretrain_weights=True):
    backbone = resnet50_fpn_backbone(
        pretrain_path="model/fasterrcnn/backbone/resnet50-0676ba61.pth",
        norm_layer=torch.nn.BatchNorm2d,
        trainable_layers=5
    )

    anchor_generator = AnchorsGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # model = EnhancedFasterRCNN(backbone=backbone,
    #                            num_classes=num_classes,
    #                            rpn_anchor_generator=anchor_generator
    #                            )

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator
                       )

    model.rpn.pre_nms_top_n_train = 3000
    model.rpn.post_nms_top_n_train = 3000
    model.rpn.pre_nms_top_n_test = 3000
    model.rpn.post_nms_top_n_test = 3000

    model.rpn.nms_thresh = 0.7

    model.rpn.fg_iou_thresh = 0.7
    model.rpn.bg_iou_thresh = 0.3
    model.rpn.batch_size_per_image = 256
    model.rpn.positive_fraction = 0.5

    model.roi_heads.batch_size_per_image = 512
    model.roi_heads.positive_fraction = 0.25

    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 3000

    if load_pretrain_weights:
        weights_dict = torch.load(
            "model/fasterrcnn/backbone/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
            map_location="cpu"
        )

        for k in list(weights_dict.keys()):
            if "roi_heads.box_predictor" in k:
                del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(
            weights_dict, strict=False
        )

        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    root = args.data_path

    if os.path.exists(os.path.join(root, "aitodv2")) is False:
        raise FileNotFoundError("aitodv2 dose not in path:'{}'.".format(root))

    img_root = os.path.join(root, "aitodv2")
    ann_root = os.path.join(img_root, "annotations_v2")

    train_img_root = os.path.join(img_root, "trainval", "images")
    val_img_root = os.path.join(img_root, "val", "images")

    train_ann_root = os.path.join(ann_root, "aitodv2_trainval.json")
    val_ann_root = os.path.join(ann_root, "aitodv2_val.json")

    train_dataset = AITODv2Dataset(train_img_root, train_ann_root, data_transform["train"])
    train_sampler = None

    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 6])  # number of workers
    print('Using %g dataloader workers' % nw)
    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=collate_fn)

    val_dataset = AITODv2Dataset(val_img_root, val_ann_root, data_transform["val"])
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=collate_fn)

    model = create_model(num_classes=args.num_classes + 1)
    model.to(device)

    print("\nChecking ...")
    print("PFIM:", hasattr(model, 'pfim'))
    print("PGDP:", hasattr(model, 'pgdp'))
    print("CBAM1:", hasattr(model, 'cbam1'))
    print("CBAM2:", hasattr(model, 'cbam2'))
    print("Total Parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        lr_scheduler.step()

        coco_info = utils.evaluate(model, val_data_set_loader, device=device)

        with open(results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[0])

        # save
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "weights/resNetFpn-model-{}.pth".format(epoch))

    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data-path', default='/share/home/u18042/code/ly/ORFENet/data')
    parser.add_argument('--num-classes', default=8, type=int)
    parser.add_argument('--output-dir', default='weights')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)

    parser.add_argument('--epochs', default=36, type=int, metavar='N')
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N')
    parser.add_argument('--aspect-ratio-group-factor', default=-1, type=int)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
