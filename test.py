import os
import json
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import transforms
from model.fasterrcnn.network.rpn import AnchorsGenerator
from model import FasterRCNN, FastRCNNPredictor, EnhancedFasterRCNN
from model import resnet50_fpn_backbone
from dataset.my_dataset import AITODv2Dataset
from utils import train_eval_utils as utils


def collate_fn(batch):
    return tuple(zip(*batch))


def create_model(num_classes):
    backbone = resnet50_fpn_backbone(
        pretrain_path="model/fasterrcnn/backbone/resnet50-0676ba61.pth",
        norm_layer=torch.nn.BatchNorm2d,
        trainable_layers=5
    )

    anchor_generator = AnchorsGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator
                       )

    model.rpn.pre_nms_top_n_test = 3000
    model.rpn.post_nms_top_n_test = 3000
    model.rpn.nms_thresh = 0.7

    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 3000

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )

    return model


@torch.no_grad()
def visualize_and_save(model, dataset, data_loader, device, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    coco = dataset.coco
    img_root = dataset.img_root

    for images, targets in tqdm(data_loader, desc="Visualizing"):
        images = list(img.to(device) for img in images)
        outputs = model(images)

        target = targets[0]
        image_id = target["image_id"].item()

        img_info = coco.loadImgs(image_id)[0]
        img_path = os.path.join(img_root, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        pred = outputs[0]
        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        for box, score, label in zip(boxes, scores, labels):
            if score < 0.05:
                continue

            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text(
                (x1, y1),
                f"{label.item()}",
                fill="green"
            )

        save_name = img_info["file_name"]
        image.save(os.path.join(save_dir, save_name))


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    root = args.data_path
    img_root = os.path.join(root, "aitodv2")
    ann_root = os.path.join(img_root, "annotations_v2")

    val_img_root = os.path.join(img_root, "val", "images")
    val_ann_root = os.path.join(ann_root, "aitodv2_val.json")

    val_dataset = AITODv2Dataset(
        val_img_root, val_ann_root, data_transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    model = create_model(num_classes=args.num_classes + 1)
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    coco_info = utils.evaluate(model, val_loader, device=device)

    ap_dict = {
        "AP": coco_info[0],
        "AP50": coco_info[1],
        "AP75": coco_info[2],
        "AP_small": coco_info[3],
        "AP_medium": coco_info[4],
        "AP_large": coco_info[5],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "ap.json"), "w") as f:
        json.dump(ap_dict, f, indent=4)

    vis_dir = os.path.join(args.output_dir, "vis")
    visualize_and_save(model, val_dataset, val_loader, device, vis_dir)

    print("Visualization saved to:", vis_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Faster R-CNN Test + Visualization")

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data-path', default='dataset/data')
    parser.add_argument('--num-classes', default=8, type=int)
    parser.add_argument('--weights', default='weights/resNetFpn-model-11.pth', type=str)
    parser.add_argument('--output-dir', default='test_results')

    args = parser.parse_args()
    main(args)
