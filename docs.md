## style.py 

`style.py` trains networks that can transfer styles from artwork into images.

**Flags**
- `--checkpoint-dir`: Directory to save checkpoint in. Required.
- `--style`: Path to style image. Required.
- `--train-path`: Path to training images folder. Default: `data/train2014`.
- `--test`: Path to content image to test network on at at every checkpoint iteration. Default: no image.
- `--test-dir`: Path to directory to save test images in. Required if `--test` is passed a value.
- `--epochs`: Epochs to train for. Default: `2`.
- `--batch-size`: Batch size for training. Default: `4`.
- `--checkpoint-iterations`: Number of iterations to go for between checkpoints. Default: `2000`.
- `--vgg-path`: Path to VGG19 network (default). Can pass VGG16 if you want to try out other loss functions. Default: `data/imagenet-vgg-verydeep-19.mat`.
- `--content-weight`: Weight of content in loss function. Default: `7.5e0`.
- `--style-weight`: Weight of style in loss function. Default: `1e2`.
- `--tv-weight`: Weight of total variation term in loss function. Default: `2e2`.
- `--learning-rate`: Learning rate for optimizer. Default: `1e-3`.
- `--slow`: For debugging loss function. Direct optimization on pixels using Gatys' approach. Uses `test` image as content value, `test_dir` for saving fully optimized images.


## evaluate.py
`evaluate.py` evaluates trained networks given a checkpoint directory. If evaluating images from a directory, every image in the directory must have the same dimensions.

**Flags**
- `--checkpoint`: Directory or `ckpt` file to load checkpoint from. Required.
- `--in-path`: Path of image or directory of images to transform. Required.
- `--out-path`: Out path of transformed image or out directory to put transformed images from in directory (if `in_path` is a directory). Required.
- `--device`: Device used to transform image. Default: `/cpu:0`.
- `--batch-size`: Batch size used to evaluate images. In particular meant for directory transformations. Default: `4`.
- `--allow-different-dimensions`: Allow different image dimensions. Default: not enabled

## transform_video.py
`transform_video.py` transforms videos into stylized videos given a style transfer net.

**Flags**
- `--checkpoint-dir`: Directory or `ckpt` file to load checkpoint from. Required.
- `--in-path`: Path to video to transfer style to. Required.
- `--out-path`: Path to out video. Required.
- `--tmp-dir`: Directory to put temporary processing files in. Will generate a dir if you do not pass it a path. Will delete tmpdir afterwards. Default: randomly generates invisible dir, then deletes it after execution completion.
- `--device`: Device to evaluate frames with. Default: `/gpu:0`.
- `--batch-size`: Batch size for evaluating images. Default: `4`.
