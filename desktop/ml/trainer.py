import threading
import time

import torch
import torch.nn as nn
import torch.optim as optim
from ml.loss import DiceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        device="cpu",
        learning_rate=0.001,
        on_log=None,
        on_epoch_end=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        print(f"[DEBUG] Model moved to: {self.device}")
        if self.device.type == "cuda":
            print(f"[DEBUG] Model is on GPU: {next(self.model.parameters()).device}")

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.on_log = on_log
        self.on_epoch_end = on_epoch_end

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)

        self.scaler = torch.amp.GradScaler() if device == "cuda" else None

        self.is_training = False
        self.stop_training = False
        self.training_thread = None

    def train(self, epochs):
        self.is_training = True
        self.stop_training = False
        self.training_thread = threading.Thread(target=self._train_loop, args=(epochs,))
        self.training_thread.start()

    def _train_loop(self, epochs):
        self.log("[INFO] Training started")

        if self.device.type == "cuda":
            self.log(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
            self.log(f"[INFO] Initial GPU memory: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")

        epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Progress", position=0)

        for epoch in epoch_pbar:

            if self.stop_training:
                break

            self.model.train()
            epoch_loss = 0
            batch_times = []

            batch_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")

            for batch_idx, (images, masks) in enumerate(batch_pbar):
                batch_start = time.time()

                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                if batch_idx == 0 and epoch == 1:
                    print(f"[DEBUG] Images device: {images.device}")
                    print(f"[DEBUG] Masks device: {masks.device}")
                    if self.device.type == "cuda":
                        print(f"[DEBUG] GPU memory after loading: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")

                self.optimizer.zero_grad()

                if self.device.type == "cuda" and self.scaler:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(images)
                        ce = self.ce_loss(outputs, masks)
                        dice = self.dice_loss(outputs, masks)
                        loss = ce + dice

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    ce = self.ce_loss(outputs, masks)
                    dice = self.dice_loss(outputs, masks)
                    loss = ce + dice

                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                if self.device.type == "cuda" and batch_idx % 10 == 0:
                    allocated = torch.cuda.memory_allocated(0) / 1024**2
                    reserved = torch.cuda.memory_reserved(0) / 1024**2
                    batch_pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "GPU_mem": f"{allocated:.0f}MB",
                            "cache": f"{reserved:.0f}MB",
                            "time": f"{batch_time:.2f}s",
                        }
                    )
                else:
                    batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / len(self.train_loader)

            val_loss = None
            if self.val_loader:
                val_loss = self.validate()

            self.scheduler.step(val_loss if val_loss else avg_loss)

            log_msg = f"[Epoch {epoch}/{epochs}] train_loss={avg_loss:.4f}"
            if val_loss:
                log_msg += f" val_loss={val_loss:.4f}"

            self.log(log_msg)

            epoch_pbar.set_postfix(
                {"train_loss": f"{avg_loss:.4f}", "val_loss": f"{val_loss:.4f}" if val_loss else "N/A"}
            )

            if self.on_epoch_end:
                self.on_epoch_end(epoch, avg_loss)

        self.is_training = False
        self.log("[INFO] Training completed")

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                if self.device.type == "cuda" and self.scaler:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(images)
                        ce = self.ce_loss(outputs, masks)
                        dice = self.dice_loss(outputs, masks)
                        loss = ce + dice
                else:
                    outputs = self.model(images)
                    ce = self.ce_loss(outputs, masks)
                    dice = self.dice_loss(outputs, masks)
                    loss = ce + dice

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def stop(self):
        self.stop_training = True

    def log(self, message):
        if self.on_log:
            self.on_log(message)
        else:
            print(message)
