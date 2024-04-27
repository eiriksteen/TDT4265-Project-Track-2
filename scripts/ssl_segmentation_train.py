import torch
import wandb
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from pprint import pprint
from scipy.spatial.distance import dice
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from medical_ssl.data import BrainTumorDataset
from transformers import SegformerForSemanticSegmentation
from medical_ssl.anomaly_detection import VAE
from medical_ssl.utils import concatanate_images
from medical_ssl.settings import DEVICE, LATENT_SIZE


def train_segmentation(
        model: SegformerForSemanticSegmentation,
        train_data: Dataset,
        validation_data: Dataset,
        out_dir: Path,
        num_epochs: int = 10,
        lr: float = 1e-03,
        batch_size: int = 16,
        wandb_run_name: str = "self_supervised_segmentation",
):

    config = {
        "model": "linear",
        "num_epochs": num_epochs,
        "lr": lr,
        "batch_size": batch_size
    }
    wandb.init(name=wandb_run_name, config=config)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False)

    model = model.to(DEVICE)
    # loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    max_dice = float("-inf")

    for epoch in range(num_epochs):

        print(f"STARTING EPOCH {epoch+1}")
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            images, masks = batch["image"].to(
                DEVICE), batch["mask"].to(DEVICE)

            optimizer.zero_grad()

            loss = model(pixel_values=images, labels=masks).loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        total_images, total_masks, total_predicted_masks = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader):
                images, masks = batch["image"].to(
                    DEVICE), batch["mask"].to(DEVICE)

                outputs = model(pixel_values=images, labels=masks)
                loss, logits = outputs.loss, outputs.logits
                val_loss += loss.item()

                upsampled_logits = nn.functional.interpolate(logits,
                                                             size=masks.size(
                                                             )[-2:],
                                                             mode='bilinear',
                                                             align_corners=False)

                predicted_masks = upsampled_logits.argmax(dim=1)

                images_list = images.detach().cpu().tolist()
                masks_list = masks[:, None, :, :].detach().cpu().tolist()
                predicted_masks_list = predicted_masks[:, None, :, :].detach(
                ).cpu().tolist()

                total_images += images_list
                total_masks += masks_list
                total_predicted_masks += predicted_masks_list

        images, masks, predicted_masks = np.asarray(total_images), np.asarray(
            total_masks), np.asarray(total_predicted_masks)

        wandb_images = []

        for _ in range(5):

            random_idx = np.random.randint(0, len(total_images))
            image, mask, predicted_mask = images[random_idx], masks[random_idx], predicted_masks[random_idx]

            res_image = concatanate_images([image, mask, predicted_mask])

            image = wandb.Image(
                res_image,
                caption="image, masked image, reconstruction"
            )

            wandb_images.append(image)

        wandb.log({"images": wandb_images})

        masks_fi = masks.flatten().astype(np.uint8)
        predicted_masks_fi = predicted_masks.flatten().astype(np.uint8)

        dice_score = 1 - dice(masks_fi, predicted_masks_fi)

        if dice_score > max_dice:
            max_dice = dice_score
            print("New top dice, saving model")
            model.save_pretrained(out_dir / "model")

        epoch_metrics = {
            "dice": dice_score,
            "validation_loss": val_loss,
            "train_loss": train_loss
        }

        pprint(epoch_metrics)

        wandb.log(epoch_metrics)


if __name__ == "__main__":

    train_data = BrainTumorDataset(
        "train", normalize=True, to_torch=True, with_rotation=True, encoding="RGB")
    validation_data = BrainTumorDataset(
        "validation", normalize=True, to_torch=True, with_rotation=True, encoding="RGB")

    print(f"TRAIN SIZE: {len(train_data)}")
    print(f"VALIDATION SIZE: {len(validation_data)}")

    out_dir = Path(
        f"ssl_segmentation_results")
    out_dir.mkdir(exist_ok=True)
    model_dir = out_dir / "model"

    if model_dir.exists():
        print("Tuning from saved checkpoint")

    model = SegformerForSemanticSegmentation.from_pretrained(model_dir if model_dir.exists() else "nvidia/mit-b0",
                                                             num_labels=2,
                                                             id2label={0: "no_anomaly", 1: "anomaly"}, label2id={"no_anomaly": 0, "anomaly": 1})

    train_segmentation(
        model,
        train_data,
        validation_data,
        out_dir,
        num_epochs=100
    )
