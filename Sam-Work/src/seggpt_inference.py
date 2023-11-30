import os
import argparse

import torch
import numpy as np

from seggpt_engine import inference_image, inference_video
import models_seggpt


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)

    parser.add_argument('--input_image', type=str, help='path to input image to be tested',
                        default=None)
    parser.add_argument('--input_video', type=str, help='path to input video to be tested',
                        default=None)
    parser.add_argument('--num_frames', type=int, help='number of prompt frames in video',
                        default=0)
    parser.add_argument('--prompt_image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt_target', type=str, nargs='+', help='path to prompt target',
                        default=None)

    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cpu')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


def run_inference(device, output_dir, input_image=None, input_video=None, prompt_image=None, prompt_target=None):
    device = torch.device(device)
    model = prepare_model('/Users/samkoshythomas/Desktop/UoA-Masters/Research-Project-DS/AIML/WeedDetection/Sam-Work/SegGPT_inference/seggpt_vit_large.pth',
                          'seggpt_vit_large_patch16_input896x448', 'instance').to(device)
    print('Model loaded.')

    assert input_image or input_video and not (input_image and input_video)
    if input_image is not None:
        assert prompt_image is not None and prompt_target is not None

        img_name = os.path.basename(input_image)
        out_path = os.path.join(
            output_dir, '.'.join(img_name.split('.')[:-1]) + '.png')
        mask_out_path = os.path.join(
            output_dir, "mask_" + '.'.join(img_name.split('.')[:-1]) + '.png')
        inference_image(model, device, input_image, prompt_image,
                        prompt_target, out_path, mask_out_path)

    if input_video is not None:
        assert prompt_target is not None and len(args.prompt_target) == 1
        vid_name = os.path.basename(input_video)
        out_path = os.path.join(
            args.output_dir, "output_" + '.'.join(vid_name.split('.')[:-1]) + '.mp4')

        inference_video(model, device, input_video, num_frames,
                        prompt_image, prompt_target, out_path)

    print('Finished.')


if __name__ == '__main__':
    args = get_args_parser()
    # run_inference(device, output_dir, input_image=None, input_video=None, prompt_image=None, prompt_target=None)
