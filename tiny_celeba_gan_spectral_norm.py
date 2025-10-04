#!/usr/bin/env python3
"""
Tiny CelebA GAN with Spectral Norm (Lightning) + explicit generator/critic checkpointing.

Defaults:
  generator weights -> drive/MyDrive/model/generator.pth
  critic (discriminator) weights -> drive/MyDrive/model/critic.pth

Features:
  - Loads generator/discriminator weights at startup if files exist.
  - Optionally loads optimizer states (separate files).
  - Saves weights (and optionally optimizer states) at:
      * end of every epoch
      * end of training
      * on SIGINT/SIGTERM (Ctrl+C)
"""

import os
import argparse
import signal
from argparse import Namespace
from typing import Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, utils

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.nn.utils import spectral_norm

# ---------------------------
# Default hyperparameters
# ---------------------------
DEFAULTS = Namespace(
    root='./data',
    dataset_path='celeba_hq',
    image_size=64,
    batch_size=256,
    latent_dim=64,
    base_ch=32,
    lr=2e-4,
    betas=(0.5, 0.999),
    max_epochs=30,
    num_workers=2,
    gpus=0,
    log_every_n_steps=50,
    save_dir='./outputs',
    tb_logdir='logs',
    subset_size=0,
    seed=42,
    sample_grid_size=12,
    sample_every_n_steps=200,
    # checkpoint defaults (paths you requested)
    gen_checkpoint='drive/MyDrive/model/generator.pth',
    crit_checkpoint='drive/MyDrive/model/critic.pth',
    gen_opt_checkpoint='drive/MyDrive/model/generator_opt.pth',
    crit_opt_checkpoint='drive/MyDrive/model/critic_opt.pth',
    load_optimizers=False,
    save_optimizers=False,
)

os.makedirs(DEFAULTS.save_dir, exist_ok=True)
# also ensure drive dir exists by default when saving
os.makedirs(os.path.dirname(DEFAULTS.gen_checkpoint) or ".", exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except Exception:
            pass
    elif classname.find('BatchNorm') != -1:
        try:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
        except Exception:
            pass

# ---------------------------
# Generator (tiny DCGAN-style)
# ---------------------------
class TinyGenerator(nn.Module):
    def __init__(self, latent_dim=64, base_ch=32, out_channels=3, image_size=64):
        super().__init__()
        assert image_size in (32, 64, 128), "image_size supported: 32/64/128"
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch, out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)

# ---------------------------
# Discriminator with spectral norm
# ---------------------------
class TinyDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()
        def SNConv(*args, **kwargs):
            return spectral_norm(nn.Conv2d(*args, **kwargs))

        self.net = nn.Sequential(
            SNConv(in_channels, base_ch, 4, 2, 1, bias=False),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            SNConv(base_ch, base_ch * 2, 4, 2, 1, bias=False),   # 16x16
            nn.LeakyReLU(0.2, inplace=True),

            SNConv(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False), # 8x8
            nn.LeakyReLU(0.2, inplace=True),

            SNConv(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False), # 4x4
            nn.LeakyReLU(0.2, inplace=True),

            SNConv(base_ch * 8, 1, 4, 1, 0, bias=False),  # 1x1
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1)

# ---------------------------
# DataModule
# ---------------------------
class CelebADataModule(pl.LightningDataModule):
    def __init__(self, root, image_size=64, batch_size=128, num_workers=4, subset_size=0):
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset_size = subset_size

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = datasets.ImageFolder(root=DEFAULTS.dataset_path, transform=self.transform)
        if self.subset_size and self.subset_size > 0:
            dataset = Subset(dataset, list(range(min(self.subset_size, len(dataset)))))
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

# ---------------------------
# LightningModule: implements hinge loss + spectral norm discriminator
# ---------------------------
class TinyGAN(pl.LightningModule):
    def __init__(self, cfg: Namespace):
        super().__init__()
        self.automatic_optimization = False

        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.generator = TinyGenerator(latent_dim=cfg.latent_dim, base_ch=cfg.base_ch, image_size=cfg.image_size)
        self.discriminator = TinyDiscriminator(base_ch=cfg.base_ch)

        # fixed z for visual samples
        self.fixed_z = torch.randn(cfg.sample_grid_size, cfg.latent_dim)

        # init weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

    # ------------------
    # Hinge loss functions
    # ------------------
    def d_hinge_loss(self, real_logits, fake_logits):
        loss_real = torch.mean(F.relu(1.0 - real_logits))
        loss_fake = torch.mean(F.relu(1.0 + fake_logits))
        return loss_real + loss_fake

    def g_hinge_loss(self, fake_logits):
        return -torch.mean(fake_logits)

    # ------------------
    # Lightning hooks
    # ------------------
    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        d_opt, g_opt = self.optimizers()

        # Train Discriminator
        z = torch.randn(batch_size, self.cfg.latent_dim, device=self.device)
        fake_imgs = self.generator(z).detach()

        real_logits = self.discriminator(real_imgs)
        fake_logits = self.discriminator(fake_imgs)

        d_loss = self.d_hinge_loss(real_logits, fake_logits)

        if hasattr(self.logger, 'experiment'):
            try:
                self.logger.experiment.add_scalar('loss/d_loss', d_loss.detach().cpu().item(), self.global_step)
            except Exception:
                pass
        
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()

        # Train Generator
        z = torch.randn(batch_size, self.cfg.latent_dim, device=self.device)
        fake_imgs = self.generator(z)
        fake_logits = self.discriminator(fake_imgs)
        g_loss = self.g_hinge_loss(fake_logits)

        if hasattr(self.logger, 'experiment'):
            try:
                self.logger.experiment.add_scalar('loss/g_loss', g_loss.detach().cpu().item(), self.global_step)
            except Exception:
                pass

        if (self.global_step % self.cfg.sample_every_n_steps == 0) or (batch_idx == 0 and self.current_epoch % 1 == 0):
            self.log_image_grid()

        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True, on_step=True, on_epoch=True)


    @rank_zero_only
    def log_image_grid(self):
        self.generator.eval()
        with torch.no_grad():
            grid = utils.make_grid(self.generator(self.fixed_z.to(self.device)).detach().cpu(), nrow=6, normalize=True)
        self.generator.train()

        if hasattr(self.logger, 'experiment'):
            try:
                self.logger.experiment.add_image('generated_images', grid, global_step=self.global_step)
            except Exception:
                pass

        out_path = os.path.join(self.cfg.save_dir, f'sample_epoch{self.current_epoch}_step{self.global_step}.png')
        utils.save_image(grid, out_path)

    def configure_optimizers(self):
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.lr, betas=self.cfg.betas)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.cfg.lr, betas=self.cfg.betas)
        return [opt_d, opt_g], []

# ---------------------------
# Checkpoint helpers (generator + critic)
# ---------------------------

@rank_zero_only
def save_model_weights(gen_path: str, crit_path: str, trainer: Optional[pl.Trainer], pl_module: pl.LightningModule,
                       save_optimizers: bool = False, gen_opt_path: Optional[str] = None, crit_opt_path: Optional[str] = None):
    """
    Save generator and discriminator weights (and optionally optimizer states).
    - gen_path/crit_path: where to save state_dict() of generator/discriminator
    - save_optimizers: if True, will attempt to save trainer.optimizers states to gen_opt_path/crit_opt_path
      (requires trainer to have created optimizers).
    """
    os.makedirs(os.path.dirname(gen_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(crit_path) or ".", exist_ok=True)
    print(f"=> Saving generator -> {gen_path}")
    torch.save(pl_module.generator.state_dict(), gen_path)
    print(f"=> Saving critic -> {crit_path}")
    torch.save(pl_module.discriminator.state_dict(), crit_path)

    if save_optimizers and trainer is not None and getattr(trainer, "optimizers", None):
        try:
            optimizers = trainer.optimizers
            # Lightning returns list [opt_d, opt_g] in our configure_optimizers order
            opt_d_state = optimizers[0].state_dict() if len(optimizers) >= 1 else None
            opt_g_state = optimizers[1].state_dict() if len(optimizers) >= 2 else None
            if gen_opt_path and opt_g_state is not None:
                torch.save(opt_g_state, gen_opt_path)
                print(f"=> Saved generator optimizer -> {gen_opt_path}")
            if crit_opt_path and opt_d_state is not None:
                torch.save(opt_d_state, crit_opt_path)
                print(f"=> Saved critic optimizer -> {crit_opt_path}")
        except Exception as e:
            print(f"Warning: failed to save optimizer states: {e}")

def load_model_weights(gen_path: str, crit_path: str, pl_module: pl.LightningModule, map_location: str = 'cpu'):
    """
    Load generator and discriminator weights into pl_module if files exist.
    (Does NOT attempt to load optimizer states here — see callback below for that.)
    """
    loaded = {"generator": False, "critic": False}
    if gen_path and os.path.isfile(gen_path):
        try:
            state = torch.load(gen_path, map_location=map_location)
            pl_module.generator.load_state_dict(state)
            loaded["generator"] = True
            print(f"=> Loaded generator weights from {gen_path}")
        except Exception as e:
            print(f"Warning: failed to load generator weights from {gen_path}: {e}")
    else:
        print(f"=> Generator checkpoint not found at {gen_path} (skipping)")
        raise KeyboardInterrupt

    if crit_path and os.path.isfile(crit_path):
        try:
            state = torch.load(crit_path, map_location=map_location)
            pl_module.discriminator.load_state_dict(state)
            loaded["critic"] = True
            print(f"=> Loaded critic weights from {crit_path}")
        except Exception as e:
            print(f"Warning: failed to load critic weights from {crit_path}: {e}")
    else:
        print(f"=> Critic checkpoint not found at {crit_path} (skipping)")

    return loaded

class OptimizerStateLoaderCallback(pl.Callback):
    """
    Loads optimizer state dicts saved as separate files at the earliest safe moment:
    on_train_start. Expects two files mapping to (discriminator optimizer, generator optimizer).
    """
    def __init__(self, gen_opt_path: str, crit_opt_path: str, override_lr: Optional[float] = None):
        super().__init__()
        self.gen_opt_path = gen_opt_path
        self.crit_opt_path = crit_opt_path
        self.override_lr = override_lr

    def on_train_start(self, trainer, pl_module):
        # trainer.optimizers should exist now
        if not getattr(trainer, "optimizers", None):
            # nothing to restore
            return

        try:
            optimizers = trainer.optimizers
            # opt_d = optimizers[0], opt_g = optimizers[1] (same order as configure_optimizers)
            if self.crit_opt_path and os.path.isfile(self.crit_opt_path) and len(optimizers) >= 1:
                try:
                    state = torch.load(self.crit_opt_path, map_location=trainer.strategy.root_device)
                    optimizers[0].load_state_dict(state)
                    if self.override_lr is not None:
                        for pg in optimizers[0].param_groups:
                            pg['lr'] = self.override_lr
                    print(f"=> Restored critic optimizer state from {self.crit_opt_path}")
                except Exception as e:
                    print(f"Warning: failed to restore critic optimizer state: {e}")

            if self.gen_opt_path and os.path.isfile(self.gen_opt_path) and len(optimizers) >= 2:
                try:
                    state = torch.load(self.gen_opt_path, map_location=trainer.strategy.root_device)
                    optimizers[1].load_state_dict(state)
                    if self.override_lr is not None:
                        for pg in optimizers[1].param_groups:
                            pg['lr'] = self.override_lr
                    print(f"=> Restored generator optimizer state from {self.gen_opt_path}")
                except Exception as e:
                    print(f"Warning: failed to restore generator optimizer state: {e}")
        except Exception as e:
            print(f"Warning: error during optimizer state restore: {e}")

class SaveEveryEpochCallback(pl.Callback):
    """
    Saves weights at end of every epoch (and at training end).
    """
    def __init__(self, gen_path: str, crit_path: str, gen_opt_path: Optional[str] = None,
                 crit_opt_path: Optional[str] = None, save_optimizers: bool = False):
        super().__init__()
        self.gen_path = gen_path
        self.crit_path = crit_path
        self.gen_opt_path = gen_opt_path
        self.crit_opt_path = crit_opt_path
        self.save_optimizers = save_optimizers

    def on_train_epoch_end(self, trainer, pl_module, *args):
        save_model_weights(self.gen_path, self.crit_path, trainer, pl_module,
                           save_optimizers=self.save_optimizers,
                           gen_opt_path=self.gen_opt_path, crit_opt_path=self.crit_opt_path)

    def on_train_end(self, trainer, pl_module):
        # final save
        save_model_weights(self.gen_path, self.crit_path, trainer, pl_module,
                           save_optimizers=self.save_optimizers,
                           gen_opt_path=self.gen_opt_path, crit_opt_path=self.crit_opt_path)

# ---------------------------
# CLI / main
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Tiny CelebA GAN with Spectral Norm (Lightning) + checkpoints')
    parser.add_argument('--root', type=str, default=DEFAULTS.root)
    parser.add_argument('--image_size', type=int, default=DEFAULTS.image_size)
    parser.add_argument('--batch_size', type=int, default=DEFAULTS.batch_size)
    parser.add_argument('--latent_dim', type=int, default=DEFAULTS.latent_dim)
    parser.add_argument('--base_ch', type=int, default=DEFAULTS.base_ch)
    parser.add_argument('--lr', type=float, default=DEFAULTS.lr)
    parser.add_argument('--betas', nargs=2, type=float, default=list(DEFAULTS.betas))
    parser.add_argument('--max_epochs', type=int, default=DEFAULTS.max_epochs)
    parser.add_argument('--num_workers', type=int, default=DEFAULTS.num_workers)
    parser.add_argument('--gpus', type=int, default=DEFAULTS.gpus)
    parser.add_argument('--save_dir', type=str, default=DEFAULTS.save_dir)
    parser.add_argument('--tb_logdir', type=str, default=DEFAULTS.tb_logdir)
    parser.add_argument('--subset_size', type=int, default=DEFAULTS.subset_size)
    parser.add_argument('--seed', type=int, default=DEFAULTS.seed)
    parser.add_argument('--sample_grid_size', type=int, default=DEFAULTS.sample_grid_size)
    parser.add_argument('--sample_every_n_steps', type=int, default=DEFAULTS.sample_every_n_steps)

    # checkpointing options
    parser.add_argument('--gen_checkpoint', type=str, default=DEFAULTS.gen_checkpoint,
                        help='path to generator weight file (state_dict)')
    parser.add_argument('--crit_checkpoint', type=str, default=DEFAULTS.crit_checkpoint,
                        help='path to critic (discriminator) weight file (state_dict)')
    parser.add_argument('--gen_opt_checkpoint', type=str, default=DEFAULTS.gen_opt_checkpoint,
                        help='path to generator optimizer state (optional)')
    parser.add_argument('--crit_opt_checkpoint', type=str, default=DEFAULTS.crit_opt_checkpoint,
                        help='path to critic optimizer state (optional)')
    parser.add_argument('--load_optimizers', action='store_true', help='attempt to load optimizer states at start')
    parser.add_argument('--save_optimizers', action='store_true', help='save optimizer states alongside weights')
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Namespace(**vars(args))
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.gen_checkpoint) or ".", exist_ok=True)

    pl.seed_everything(cfg.seed)

    # DataModule and Model
    dm = CelebADataModule(root=cfg.root, image_size=cfg.image_size, batch_size=cfg.batch_size,
                          num_workers=cfg.num_workers, subset_size=cfg.subset_size)
    model = TinyGAN(cfg)

    # Attempt to load model weights at startup (only model weights here)
    # map_location set to cpu so it can be loaded on CPU and then model moved by Lightning later
    load_model_weights(cfg.gen_checkpoint, cfg.crit_checkpoint, model, map_location='cpu')

    # Logger and Trainer
    tb_logger = pl.loggers.TensorBoardLogger(cfg.tb_logdir, name='tiny_gan_sn')

    callbacks = []
    # Save callback: saves weights every epoch and at training end
    save_cb = SaveEveryEpochCallback(cfg.gen_checkpoint, cfg.crit_checkpoint,
                                     gen_opt_path=cfg.gen_opt_checkpoint, crit_opt_path=cfg.crit_opt_checkpoint,
                                     save_optimizers=cfg.save_optimizers)
    callbacks.append(save_cb)

    # Optimizer loader callback (optional)
    if cfg.load_optimizers:
        callbacks.append(OptimizerStateLoaderCallback(cfg.gen_opt_checkpoint, cfg.crit_opt_checkpoint, override_lr=cfg.lr))

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        precision=16 if torch.cuda.is_available() else 32,
        logger=tb_logger,
        log_every_n_steps=DEFAULTS.log_every_n_steps,
        default_root_dir=cfg.save_dir,
        callbacks=callbacks,
    )

    # Register signal handlers for clean save on Ctrl+C / termination
    def _signal_handler(sig, frame):
        print(f"\nReceived signal {sig}. Saving weights to disk before exit...")
        try:
            # trainer.fit may not have been called yet; in any case, save model weights (no optimizers)
            # If trainer has been created and model is on GPU/TPU, trainer.strategy.root_device is used inside save function via trainer
            save_model_weights(cfg.gen_checkpoint, cfg.crit_checkpoint, trainer if 'trainer' in locals() else None, model,
                               save_optimizers=cfg.save_optimizers,
                               gen_opt_path=cfg.gen_opt_checkpoint, crit_opt_path=cfg.crit_opt_checkpoint)
        except Exception as e:
            print(f"Warning: failed to save on signal: {e}")
        # re-raise default behavior (exit)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Fit
    trainer.fit(model, dm)

if __name__ == '__main__':
    main()
