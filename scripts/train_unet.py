import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from mis import UNet, BratsDataset
from pprint import pprint


def train_unet(
        unet,
        train_data,
        validation_data,
        num_epochs,
        lr,
        batch_size,
        out_dir = Path("unet_training_results")
    ):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    min_val_loss = float("inf")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):

        print(f"STARTING EPOCH {epoch+1}")
        unet.train()
        train_loss = 0

        for batch in tqdm(train_loader):

            image = batch["image"]
            seg = batch["seg"]
            t1 = batch["t1"]
            t1ce = batch["t1ce"]
            t2 = batch["t2"]

            inp = torch.concat((image, t1, t1ce, t2), dim=1)

            outputs = unet(inp)

            loss = loss_fn(outputs, seg)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
        

        unet.eval()
        val_loss = 0
        # total_labels, total_preds = [], []
        with torch.no_grad():
            for batch in tqdm(validation_loader):

                image = batch["image"]
                seg = batch["seg"]
                t1 = batch["t1"]
                t1ce = batch["t1ce"]
                t2 = batch["t2"]

                inp = torch.concat((image, t1, t1ce, t2), dim=1)

                outputs = unet(inp)

                loss = loss_fn(outputs, seg)

                val_loss += loss.item()

        metrics = {
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(validation_loader)
        }

        if metrics["val_loss"] < min_val_loss:
            min_val_loss = metrics["val_loss"]
            torch.save(unet.state_dict(), out_dir / "unet")

        pprint(metrics)


if __name__ == "__main__":

    train_data = BratsDataset(
        "train",
        size=128,
        normalize=True,
        to_torch=True,
        compression_params=[80, 8]
    )

    validation_data = BratsDataset(
        "train",
        size=128,
        normalize=True,
        to_torch=True,
        compression_params=[80, 8]
    ) 

    unet = UNet(in_channels=4, out_channels=1)

    train_unet(
        unet, 
        train_data, 
        validation_data,
        25,
        1e-03,
        8)
