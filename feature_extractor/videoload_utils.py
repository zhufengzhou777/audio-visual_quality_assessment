import skvideo.io


def load_video(video_path):
    video = skvideo.io.vread(video_path, 1920, 1080, inputdict={'-pix_fmt': 'yuvj420p'})
    # video = rearrange(video, 'f h w c ->f c h w')
    return video
