import json
import torch
import time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from mis import BratsDataset
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import huggingface_hub
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from pprint import pprint
from mis.datasets import ASOCADataset
from mis.settings import DEVICE, ASOCA_PATH
from mis.loss import dice_loss, gdlv_loss, focal_loss
import argparse
import seaborn as sns

sns.set_style("darkgrid")
plt.rc("figure", figsize=(16, 8))
plt.rc("font", size=13)

torch.manual_seed(0)

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
    loss_fn = focal_loss
    optimizer = torch.optim.AdamW(segformer.parameters(), lr=lr)
    min_val_loss = float("inf")
    out_dir.mkdir(exist_ok=True)
    metrics = []
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):

        print(f"STARTING EPOCH {epoch+1}")
        segformer.train()
        train_loss = 0

        pbar = tqdm(train_loader)
        for i, batch in enumerate(pbar):

            if i < 2:

                # Prepare data
                image = batch["image"]
                seg = batch["mask"]
                seg = seg.squeeze(dim=1).type(torch.LongTensor)

                # Forward pass
                outputs = segformer(pixel_values=image, labels=seg)
                
                # Upsample logits
                logits = outputs.logits
                upsampled_logits = nn.functional.interpolate(logits,
                                                                 size=seg.size()[-2:],
                                                                 mode='bilinear',
                                                                 align_corners=False)

                # Predict masks - Focal loss
                predicted_masks = upsampled_logits.argmax(dim=1)
                masks_fi = seg.flatten().type(torch.FloatTensor)
                predicted_masks_fi = predicted_masks.flatten().type(torch.FloatTensor)
                # masks_fi = seg.flatten().type(np.uint8)                               # For dice loss
                # predicted_masks_fi = predicted_masks.flatten().type(np.uint8)         # For dice loss
                
                # # Calculate loss - Focal loss
                loss = loss_fn(masks_fi, predicted_masks_fi)
                loss = Variable(loss, requires_grad=True)

                # loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
                if i % 2 == 0:
                    pbar.set_description(f"Training loss at step {i} = {train_loss / (i+1)}")
                    
            else:
                break

        segformer.eval()
        val_loss = 0
        total_imgs, total_preds, total_segs = [], [], []
        with torch.no_grad():
            pbar = tqdm(validation_loader)
            for i, batch in enumerate(pbar):
                
                if i < 2:
                    
                    # Prepare data
                    image = batch["image"]
                    seg = batch["mask"]
                    seg = seg.squeeze(dim=1).type(torch.LongTensor)

                    # Forward pass
                    outputs = segformer(pixel_values=image, labels=seg)

                    # Upsample logits
                    logits = outputs.logits
                    upsampled_logits = nn.functional.interpolate(logits,
                                                                 size=seg.size()[-2:],
                                                                 mode='bilinear',
                                                                 align_corners=False)

                    # Predict masks - Focal loss
                    predicted_masks = upsampled_logits.argmax(dim=1)
                    masks_fi = seg.flatten().type(torch.FloatTensor)
                    predicted_masks_fi = predicted_masks.flatten().type(torch.FloatTensor)
                    # masks_fi = seg.flatten().type(np.uint8)                               # For dice loss
                    # predicted_masks_fi = predicted_masks.flatten().type(np.uint8)         # For dice loss

                    # # Calculate loss - Focal loss
                    loss = loss_fn(masks_fi, predicted_masks_fi)
                    loss = Variable(loss, requires_grad=True)
                    val_loss += loss.item()
                    
                    # Store images, predictions and masks
                    # preds = torch.where(logits >= 0.5, 1.0, 0.0)
                    total_imgs += image.detach().cpu().tolist()
                    total_preds += predicted_masks[:, None, :, :].detach().cpu().tolist()
                    total_segs += seg[:, None, :, :].detach().cpu().tolist()
                
                    if i % 2 == 0:
                        pbar.set_description(f"Validation loss at step {i} = {train_loss / (i+1)}")
                
                else:
                    break

        total_segs_np = np.asarray(total_segs)
        total_preds_np = np.asarray(total_preds)
        a = accuracy_score(total_segs_np.flatten(), total_preds_np.flatten())
        p, r, f, s = precision_recall_fscore_support(total_segs_np.flatten(), total_preds_np.flatten())

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(validation_loader)

        metrics.append({
            "val_loss": val_loss,
            "train_loss": train_loss,
            "accuracy": a,
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f.tolist(),
            "support": s.tolist()
        })
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < min_val_loss:
            print("New min loss, saving model...")
            torch.save(segformer.state_dict(), out_dir / "segformer")
            min_val_loss = val_loss
            
            epoch_dir = out_dir / f"epoch{epoch+1}"
            epoch_dir.mkdir(exist_ok=True)
  
            with open(epoch_dir / "metrics.json", "w") as f:
                json.dump(metrics[-1], f, indent=4)

            total_imgs_np = np.asarray(total_imgs)
            rand_idx = np.random.choice(len(total_preds), 50, replace=False)

            for idx in rand_idx:
                _, ax = plt.subplots(ncols=3)
                img = total_imgs_np[idx]
                seg = total_segs_np[idx]
                pred = total_preds_np[idx]

                ax[0].imshow(img.squeeze())
                ax[0].set_title("Image")
                ax[1].imshow(seg.squeeze())
                ax[1].set_title("Ground Truth")
                ax[2].imshow(pred.squeeze())
                ax[2].set_title("Prediction")

                plt.savefig(epoch_dir / f"pred{idx}")
                plt.close()

            pprint(metrics[-1])
    
    plt.title("Loss Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Focal Loss")
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Train Loss", "Validation Loss"])
    plt.savefig(out_dir / "loss.png")
    plt.close()


if __name__ == "__main__":

    # Update config
    semantic_loss_ignore_index = 0      # Background class
    num_channels = 1

    config = SegformerConfig()
    config.id2label = {0: "background", 1: "artery"}
    config.label2id = {"background": 0, "artery": 1}
    config.semantic_loss_ignore_index = semantic_loss_ignore_index
    config.num_channels = num_channels
    
    # Get model
    segformer = SegformerForSemanticSegmentation(config=config)
    # segformer = SegformerModel(configuration)                           # Alternative
    
    # Load data
    data = ASOCADataset(
        size=256,
        two_dim=True,
        to_torch=True,
        norm=True,
        data_dir=ASOCA_PATH
    )
    train_data, validation_data = torch.utils.data.random_split(data, [0.8, 0.2])
    
    # Ensure image size is correct
    image_size = 256
    segformer.config.image_size = image_size
    
    
    # Run model
    print(f"RUNNING WITH {len(train_data)} TRAIN SAMPLES AND {len(validation_data)} VALID SAMPLES")

    loss = "focal"
    dataset = "asoca"
    num_epochs = 2
    lr = 1e-03
    batch_size = 8

    out_dir = Path(f"segformer_training_results_{loss}_{dataset}")
    out_dir.mkdir(exist_ok=True)

    try:
        segformer.load_state_dict(torch.load(out_dir / "segformer"))
        print(f"Found model at {out_dir / 'segformer'}, resuming training")
    except FileNotFoundError:
        print(f"No model found at {out_dir / 'segformer'}, training from scratch")

    start = time.time()

    train_segformer(
        segformer, 
        train_data, 
        validation_data,
        num_epochs,
        lr,
        batch_size,
        out_dir=out_dir
        )

    end = time.time()
    total_seconds_elapsed = end - start
    print(f"Training took {total_seconds_elapsed} seconds")
    
    # Save time elapsed
    hours_elapsed = total_seconds_elapsed // 3600
    minutes_elapsed = (total_seconds_elapsed % 3600) // 60
    seconds_elapsed = total_seconds_elapsed % 60
    with open(out_dir / "time_elapsed.txt", "w") as f:
        f.write(f'Training took {total_seconds_elapsed} seconds\n')
        f.write(f'Hours: {hours_elapsed}, Minutes: {minutes_elapsed}, Seconds: {seconds_elapsed}\n')
    