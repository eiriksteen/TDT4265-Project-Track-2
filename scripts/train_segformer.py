import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from mis import BratsDataset
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import huggingface_hub
from pprint import pprint
from mis.datasets import ASOCADataset
from mis.settings import DEVICE, ASOCA_PATH
from mis.loss import dice_loss

def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    n = len(y_true.flatten())
    return (1/n) * (torch.sum(1 - (2*y_true*y_pred + 1)/(y_true+y_pred + 1)))


def train_segformer(
        segformer,
        train_data,
        validation_data,
        num_epochs,
        lr,
        batch_size,
        out_dir = Path("segformer_training_results")
    ):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # loss_fn = dice_loss       # Call it manually
    optimizer = torch.optim.AdamW(segformer.parameters(), lr=lr)
    min_val_loss = float("inf")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):

        print(f"STARTING EPOCH {epoch+1}")
        segformer.train()
        train_loss = 0

        pbar = tqdm(train_loader)
        for i, batch in enumerate(pbar):

            # Prepare data
            image = batch["image"].to(DEVICE)
            seg = batch["mask"].to(DEVICE)
            image = image.repeat(1, 3, 1, 1)                        # Repeat image 3 times to match number of channels required by Segformer
            seg = seg.squeeze(dim=1).type(torch.LongTensor)

            # Forward pass
            outputs = segformer(pixel_values=image, labels=seg)
            
            # Upsample logits
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(logits,
                                                             size=seg.size()[-2:],
                                                             mode='bilinear',
                                                             align_corners=False)

            # Predict masks
            predicted_masks = upsampled_logits.argmax(dim=1)
            masks_fi = seg.flatten().type(torch.uint8)
            predicted_masks_fi = predicted_masks.flatten().type(torch.uint8)
            
            # Calculate loss
            loss = dice_loss(masks_fi, predicted_masks_fi)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
            if i % 2 == 0:
                pbar.set_description(f"Training loss at step {i} = {train_loss / (i+1)}")

        segformer.eval()
        val_loss = 0
        # total_labels, total_preds = [], []
        with torch.no_grad():
            for batch in tqdm(validation_loader):

                # Prepare data
                image = batch["image"]
                seg = batch["mask"]
                image = image.repeat(1, 3, 1, 1)                        # Repeat image 3 times to match number of channels required by Segformer
                seg = seg.squeeze(dim=1).type(torch.LongTensor)

                # Forward pass
                outputs = segformer(pixel_values=batch, labels=seg)

                # Upsample logits
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(logits,
                                                             size=seg.size()[-2:],
                                                             mode='bilinear',
                                                             align_corners=False)

                # Predict masks
                predicted_masks = upsampled_logits.argmax(dim=1)
                masks_fi = seg.flatten().astype(np.uint8)
                predicted_masks_fi = predicted_masks.flatten().astype(np.uint8)

                # Calculate loss
                loss = dice_loss(masks_fi, predicted_masks_fi)
                val_loss += loss.item()

        metrics = {
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(validation_loader)
        }

        if metrics["val_loss"] < min_val_loss:
            min_val_loss = metrics["val_loss"]
            torch.save(segformer.state_dict(), out_dir / "segformer")

        pprint(metrics)


if __name__ == "__main__":

    # train_data = BratsDataset(
    #     "train",
    #     size=128,
    #     normalize=True,
    #     to_torch=True,
    #     compression_params=[80, 8]
    # )

    # validation_data = BratsDataset(
    #     "train",
    #     size=128,
    #     normalize=True,
    #     to_torch=True,
    #     compression_params=[80, 8]
    # )
    
    # train_data = torch.rand((100, 3, 128, 128))
    # validation_data = torch.rand((100, 3, 128, 128))
    
    # train_data = torch.randint(size=(100, 3, 128, 128), low=0, high=1)
    # validation_data = torch.randint(size=(100, 3, 128, 128), low=0, high=1)

    config = SegformerConfig()
    segformer = SegformerForSemanticSegmentation(config=config)
    # segformer = SegformerModel(configuration)                           # Alternative
    
    
    data = ASOCADataset(
        size=256,
        two_dim=True,
        to_torch=True,
        norm=True,
        data_dir=ASOCA_PATH
    )
    train_data, validation_data = torch.utils.data.random_split(data, [0.8, 0.2])

    print(train_data[0]["mask"].shape)
    # x = train_data[0]["mask"].repeat(3, 1, 1)
    # print(x.shape)

    # Update class labels
    id2label = {0: "background", 1: "artery"}
    label2id = {"background": 0, "artery": 1}
    segformer.config.id2label = id2label
    segformer.config.label2id = label2id
    
    # Update image size
    image_size = 256
    segformer.config.image_size = image_size
    
    # Update number of channels
    num_channels = 1
    segformer.config.num_channels = num_channels

    train_segformer(
        segformer, 
        train_data, 
        validation_data,
        25,
        1e-03,
        8)
