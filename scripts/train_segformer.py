import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from mis import BratsDataset
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import huggingface_hub
from pprint import pprint

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

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    # loss_fn = dice_loss       # Call it manually
    optimizer = torch.optim.AdamW(segformer.parameters(), lr=lr)
    min_val_loss = float("inf")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):

        print(f"STARTING EPOCH {epoch+1}")
        segformer.train()
        train_loss = 0

        for batch in tqdm(train_loader):

            # Define test labels
            seg = torch.rand((batch_size, 3, 128, 128))
            # seg = torch.randint(size=(batch_size, 3, 128, 128), low=0, high=1)
            # print(batch.shape)
            # print(seg.shape)

            outputs = segformer(batch)
            # outputs = segformer(pixel_values=batch, labels=seg)
            # print(outputs)
            # loss = outputs.loss
            # print(loss)
            # logits = outputs.logits
            # print(logits.shape)

            # loss = loss_fn(outputs, seg)
            # print(outputs.logits.shape, seg.shape)
            loss = dice_loss(outputs.logits, seg)
            # loss = segformer(batch, seg).loss

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
        

        segformer.eval()
        val_loss = 0
        # total_labels, total_preds = [], []
        with torch.no_grad():
            for batch in tqdm(validation_loader):

                # Define test labels
                seg = torch.rand((batch_size, 3, 128, 128))

                outputs = segformer(batch)

                # loss = loss_fn(outputs, seg)
                loss = dice_loss(outputs, seg)

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
    
    train_data = torch.rand((100, 3, 128, 128))
    validation_data = torch.rand((100, 3, 128, 128))
    
    # train_data = torch.randint(size=(100, 3, 128, 128), low=0, high=1)
    # validation_data = torch.randint(size=(100, 3, 128, 128), low=0, high=1)

    config = SegformerConfig()
    segformer = SegformerForSemanticSegmentation(config=config)
    # segformer = SegformerModel(configuration)                           # Alternative
    

    # Update class labels
    id2label = {0: "background", 1: "artery"}
    label2id = {"background": 0, "artery": 1}
    segformer.config.id2label = id2label
    segformer.config.label2id = label2id
    
    # Update image size
    image_size = 128
    segformer.config.image_size = image_size
    
    # Update number of channels
    num_channels = 3
    segformer.config.num_channels = num_channels

    train_segformer(
        segformer, 
        train_data, 
        validation_data,
        25,
        1e-03,
        8)
