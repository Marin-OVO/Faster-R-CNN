import os
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw

import transforms
from dataset.my_dataset import AITODv2Dataset


def collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def visualize_gt_and_save(dataset, data_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    coco = dataset.coco
    img_root = dataset.img_root

    for images, targets in tqdm(data_loader, desc="Visualizing GT"):
        target = targets[0]
        image_id = target["image_id"].item()

        img_info = coco.loadImgs(image_id)[0]
        img_path = os.path.join(img_root, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle(
                [x1, y1, x2, y2],
                outline="green",   # 与你预测框一致
                width=2
            )
            draw.text(
                (x1, y1),
                f"{label.item()}",
                fill="green"
            )

        image.save(os.path.join(save_dir, img_info["file_name"]))


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
        val_img_root,
        val_ann_root,
        data_transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    save_dir = os.path.join(args.output_dir, "vis_gt")
    visualize_gt_and_save(val_dataset, val_loader, save_dir)

    print("GT visualization saved to:", save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("GT Bounding Box Visualization")

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data-path', default='dataset/data')
    parser.add_argument('--output-dir', default='gt__results')

    args = parser.parse_args()
    main(args)
