import os
import torch
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from torch.optim import AdamW
from tqdm import tqdm

# Set paths
MODEL_DIR = "./trocr-finetuned-captcha"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")

def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"[Rank {rank}] Setting up DDP...")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Process initialized.")

def load_captcha_dataset(image_dir, labels_file):
    data = {"image_path": [], "text": []}
    if not os.path.exists(labels_file):
        print(f"[Warning] Labels file '{labels_file}' not found. Returning empty dataset.")
        return Dataset.from_dict(data)
    with open(labels_file, "r") as f:
        for line in f:
            img_name, label = line.strip().split(" ", 1)
            data["image_path"].append(os.path.join(image_dir, img_name))
            data["text"].append(label)
    return Dataset.from_dict(data)

def preprocess_data(batch, processor):
    image = Image.open(batch["image_path"]).convert("RGB")
    batch["pixel_values"] = processor(images=image, return_tensors="pt").pixel_values[0]
    batch["labels"] = processor.tokenizer(batch["text"], padding="max_length", truncation=True).input_ids
    return batch

class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "pixel_values": torch.tensor(item["pixel_values"]),
            "labels": torch.tensor(item["labels"]),
        }

def save_checkpoint(model, optimizer, scaler, epoch, processor):
    """Save model, optimizer, scaler state for checkpointing."""
    if dist.get_rank() == 0:
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save the full model (including encoder, decoder, and config)
        model.module.save_pretrained(MODEL_DIR)
        processor.save_pretrained(MODEL_DIR)

        # Save optimizer and scaler states separately
        checkpoint = {
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

        print(f"[Rank {dist.get_rank()}] Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, scaler, device):
    """Load model, optimizer, and scaler state from checkpoint, handling DDP wrapping."""
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # Load model state before wrapping with DDP
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"[Rank {dist.get_rank()}] Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    return 0

def evaluate(model, val_loader, device):
    """Evaluate model on validation set and return average loss across all ranks."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            batch_size = pixel_values.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    # Aggregate loss and sample count from all processes
    total_loss_tensor = torch.tensor(total_loss, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    avg_loss = total_loss_tensor.item() / total_samples_tensor.item() if total_samples_tensor.item() != 0 else float('inf')
    return avg_loss

def train(rank, world_size, num_epochs=1, batch_size=8, grad_accumulation_steps=4):
    print(f"[Rank {rank}] Setting up DDP...")
    setup_ddp(rank, world_size)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{rank}")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    optimizer = AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    # Load from checkpoint before wrapping with DDP
    start_epoch = load_checkpoint(model, optimizer, scaler, device) + 1

    # Wrap model with DDP after loading checkpoint
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Enable static graph optimization
    model._set_static_graph()

    train_dir = "dataset/train"
    val_dir = "dataset/val"
    train_labels = "dataset/train/labels.txt"
    val_labels = "dataset/val/labels.txt"

    train_dataset = load_captcha_dataset(train_dir, train_labels)
    val_dataset = load_captcha_dataset(val_dir, val_labels)

    train_dataset = train_dataset.map(lambda batch: preprocess_data(batch, processor), remove_columns=["image_path", "text"])
    val_dataset = val_dataset.map(lambda batch: preprocess_data(batch, processor), remove_columns=["image_path", "text"])

    train_dataset = CaptchaDataset(train_dataset)
    val_dataset = CaptchaDataset(val_dataset)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 3
    no_improve_counter = 0
    min_delta = 3e-4

    # For the very first epoch, calculate initial validation loss baseline BEFORE training
    initial_val_loss = evaluate(model, val_loader, device)
    best_val_loss = initial_val_loss
    if rank == 0:
        print(f"[Rank {rank}] Baseline Validation Loss at start of first epoch: {initial_val_loss:.4f}")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training phase for this epoch
        train_sampler.set_epoch(epoch)
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch} (Training)", position=rank)
        optimizer.zero_grad()
        model.train()  # Ensure model is in training mode
        for step, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()

            if (step + 1) % grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})
        print(f"[Rank {rank}] Epoch {epoch} completed. Avg Training Loss: {total_loss / len(train_loader):.4f}")

        # Validation phase after training the epoch (including the first epoch)
        avg_val_loss = evaluate(model, val_loader, device)
        improved = False  # flag to indicate if this epoch improved

        if rank == 0:
            print(f"[Rank {rank}] Validation Loss at epoch {epoch} after training: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            if best_val_loss - avg_val_loss >= min_delta:
                improved = True
                best_val_loss = avg_val_loss
                no_improve_counter = 0
                if rank == 0:
                    print(f"[Rank {rank}] Improvement detected! (improved: {improved})")
            else:
                best_val_loss = avg_val_loss
                no_improve_counter += 1
                if rank == 0:
                    print(f"[Rank {rank}] Minor improvement, not counted (improved: {improved}). Early stopping counter: {no_improve_counter}/{patience}")
        else:
            no_improve_counter += 1
            if rank == 0:
                print(f"[Rank {rank}] No improvement (improved: {improved}). Early stopping counter: {no_improve_counter}/{patience}")

        if no_improve_counter >= patience:
            if rank == 0:
                print(f"[Rank {rank}] Early stopping triggered. Exiting training.")
            break

        # Ensure all ranks have finished the epoch before saving checkpoint
        dist.barrier()

        if rank == 0 and True:
            save_checkpoint(model, optimizer, scaler, epoch, processor)
            print(f"[Rank 0] Saved checkpoint for epoch {epoch}")

    print(f"[Rank {rank}] Training complete. Cleaning up DDP...")
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training.")

    try:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"[Error] Multi-GPU training failed: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
