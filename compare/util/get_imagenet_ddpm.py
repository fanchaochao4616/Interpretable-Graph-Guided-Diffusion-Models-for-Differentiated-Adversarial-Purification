import torch
from guided_diffusion.script_util import create_model,create_gaussian_diffusion

def get_imagenet_ddpm(conf):
    model = create_model(image_size=conf.net.image_size,
                         num_channels=conf.net.num_channels,
                         num_res_blocks=conf.net.num_res_blocks,
                         channel_mult=conf.net.channel_mult,
                         learn_sigma=conf.net.learn_sigma,
                         class_cond=conf.net.class_cond,
                         use_checkpoint=conf.net.use_checkpoint,
                         attention_resolutions=conf.net.attention_resolutions,
                         num_heads=conf.net.num_heads,
                         num_head_channels=conf.net.num_head_channels,
                         num_heads_upsample=conf.net.num_heads_upsample,
                         use_scale_shift_norm=conf.net.use_scale_shift_norm,
                         dropout=conf.net.dropout,
                         resblock_updown=conf.net.resblock_updown,
                         use_fp16=conf.net.use_fp16,
                         use_new_attention_order=conf.net.use_new_attention_order, )
    diffusion = create_gaussian_diffusion(
        steps=conf.net.diffusion_steps,
        learn_sigma=conf.net.learn_sigma,
        noise_schedule=conf.net.noise_schedule,
        use_kl=conf.net.use_kl,
        predict_xstart=conf.net.predict_xstart,
        rescale_timesteps=conf.net.rescale_timesteps,
        rescale_learned_sigmas=conf.net.rescale_learned_sigmas,
        timestep_respacing=conf.net.timestep_respacing,
    )
    model.load_state_dict(
        torch.load(conf.net.path, map_location="cpu")
    )
    model.to("cuda")
    if conf.net.use_fp16:
        model.convert_to_fp16()
    _ = model.eval()
    return model, diffusion
