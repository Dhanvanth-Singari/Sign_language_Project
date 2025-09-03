# backend/track2_model/src/diffusion_model.py

from diffusers import UNet2DModel, DDPMScheduler

def build_diffusion_models():
    unet = UNet2DModel(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    return unet, scheduler
