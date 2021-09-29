import os
import tempfile
from pathlib import Path

import cog

from evaluate import *
from transform_video import *


class Predictor(cog.Predictor):
    def setup(self):
        """nothing to pre-load"""
        # no setup here as we need to
        # dynamically change which checkpoint to load
        # loading is very quick though!

    @cog.input(
        "input",
        type=Path,
        help="Input file: can be image (jpg, png) or video (mp4). Video processing takes ~100 milliseconds per frame",
    )
    @cog.input(
        "style",
        type=str,
        options=["la_muse", "rain_princess", "scream", "udnie", "wave", "wreck"],
        help="Pre-trained style to apply to input image",
        default="udnie",
    )
    def predict(self, input, style):
        """Compute prediction"""
        output_path_jpg = Path(tempfile.mkdtemp()) / "output.jpg"
        output_path_video = Path(tempfile.mkdtemp()) / "output.mp4"

        checkpoints_dir = "pretrained_models"
        checkpoint_path = os.path.join(checkpoints_dir, style + ".ckpt")
        device = "/gpu:0"
        batch_size = 4

        img_extensions = [".jpg", ".png"]
        video_extensions = [".mp4"]

        if input.suffix in img_extensions:
            ffwd_to_img(str(input), output_path_jpg, checkpoint_path, device=device)
            return output_path_jpg

        elif input.suffix in video_extensions:
            ffwd_video(
                str(input), str(output_path_video), checkpoint_path, device, batch_size
            )
            return output_path_video

        else:
            raise NotImplementedError("Input file extension not supported")
