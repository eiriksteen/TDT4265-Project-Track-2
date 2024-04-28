import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
import seaborn as sns
import torch.utils
import torch.utils.data
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from mis.models import Unet2d, Unet2dNonLocal
from mis.datasets import ASOCADataset
from mis.settings import DEVICE, ASOCA_PATH
from mis.loss import dice_loss

sns.set_style("darkgrid")
plt.rc("figure", figsize=(16, 8))
plt.rc("font", size=13)

torch.manual_seed(0)

def train(
        model: nn.Module,
        train_data: Dataset,
        validation_data: Dataset,
        args,
        out_dir: Path = Path("./training_results")
    ):

    print(f"lr={args.lr}\nnum_epochs={args.num_epochs}\nbatch_size={args.batch_size}")

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.99)
    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_dl = DataLoader(validation_data, batch_size=args.batch_size)
    metrics = []
    min_loss = float("inf")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(870.0))
    train_losses, val_losses = [], []

    for epoch in range(args.num_epochs):

        print(f"STARTING EPOCH {epoch+1}")
        train_loss = 0
        model.train()

        pbar = tqdm(train_dl)
        for i, batch in enumerate(pbar):

            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            logits, _ = model(images)
            # loss = dice_loss(masks, probs)
            loss = loss_fn(masks, logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if i % 50 == 0:
                pbar.set_description(f"Loss at step {i} = {train_loss / (i+1)}")

        print("RUNNING VALIDATION")
        model.eval()
        validation_loss = 0
        total_imgs, total_preds, total_masks = [], [], []
        with torch.no_grad():
            
            for batch in tqdm(validation_dl):

                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                logits, probs = model(images)
                #loss = dice_loss(masks, probs)
                loss = loss_fn(masks, logits)
                validation_loss += loss.item()

                preds = torch.where(probs >= 0.5, 1.0, 0.0)
                total_imgs += images.detach().cpu().tolist()
                total_preds += preds.detach().cpu().tolist()
                total_masks += masks.detach().cpu().tolist()

        total_masks_np = np.asarray(total_masks)
        total_preds_np = np.asarray(total_preds)
        a = accuracy_score(total_masks_np.flatten(), total_preds_np.flatten())
        p, r, f, s = precision_recall_fscore_support(total_masks_np.flatten(), total_preds_np.flatten())

        train_loss = train_loss / len(train_dl)
        validation_loss = validation_loss / len(validation_dl)

        metrics.append({
            "dice_val": validation_loss,
            "dice_train": train_loss,
            "accuracy": a,
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f.tolist(),
            "support": s.tolist()
        })

        train_losses.append(train_loss)
        val_losses.append(validation_loss)

        if validation_loss < min_loss:
            print("New min loss, saving model...")
            torch.save(model.state_dict(), out_dir / "model")
            min_loss = validation_loss

            epoch_dir = out_dir / f"epoch{epoch+1}"
            epoch_dir.mkdir(exist_ok=True)
  
            with open(epoch_dir / "metrics.json", "w") as f:
                json.dump(metrics[-1], f)

            total_imgs_np = np.asarray(total_imgs)
            rand_idx = np.random.choice(len(total_preds), 50, replace=False)

            for idx in rand_idx:
                _, ax = plt.subplots(ncols=3)
                img = total_imgs_np[idx]
                mask = total_masks_np[idx]
                pred = total_preds_np[idx]

                ax[0].imshow(img.squeeze())
                ax[0].set_title("Image")
                ax[1].imshow(mask.squeeze())
                ax[1].set_title("Ground Truth")
                ax[2].imshow(pred.squeeze())
                ax[2].set_title("Prediction")

                plt.savefig(epoch_dir / f"pred{idx}")
                plt.close()

        pprint(metrics[-1])

    plt.title("Loss Per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Train Loss", "Validation Loss"])
    plt.savefig(out_dir / "loss.png")
    plt.close()

    return model, metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=1e-03, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_epochs", default=25, type=int)
    parser.add_argument("--non_local", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    data = ASOCADataset(
        size=256,
        two_dim=True,
        to_torch=True,
        norm=True,
        data_dir=ASOCA_PATH
    )

    train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])

    print(f"RUNNING WITH {len(train_data)} TRAIN SAMPLES AND {len(val_data)} VALID SAMPLES")

    out_dir = Path(f"unet2d{'_nonlocal' if args.non_local else ''}_training_results")
    out_dir.mkdir(exist_ok=True)

    model = Unet2dNonLocal(1, 1) if args.non_local else Unet2d(1, 1)

    try:
        model.load_state_dict(torch.load(out_dir / "model"))
        print(f"Found model at {out_dir / 'model'}, resuming training")
    except FileNotFoundError:
        print(f"No model found at {out_dir / 'model'}, training from scratch")

    train(
        model,
        train_data,
        val_data,
        args,
        out_dir=out_dir
    )
