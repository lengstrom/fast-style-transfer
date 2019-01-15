## style.py 

`style.py` 训练画风迁移网络

**参数**
- `--checkpoint-dir`: 模型文件的路径，必选（注：在训练 TensorFlow 模型时，每迭代若干轮需要保存一次权值到磁盘，称为“checkpoint”）
- `--style`: 画风文件路径，必选
- `--train-path`: 训练集数据文件路径，默认: `data/train2014`（注：train2014 为微软2014年发布的图像识别、分割和图像语义数据集）
- `--test`: 每若干轮迭代结束后，用于测试神经网络渲染效果的图片路径，默认值为空
- `--test-dir`: 神经网络测试结果（渲染后的图片）的存储目录，如果 `--test` 已赋值，则该项必选
- `--epochs`: 神经网络训练周期，默认值：`2`
- `--batch_size`: 迭代的批量大小，默认值：`4`
- `--checkpoint-iterations`: 每两次checkpoint间的迭代次数，默认值：`2000`
- `--vgg-path`: VGG19网络（默认）的matlab数据文件路径，（如想尝试其他损失网络，如：VGG16，则可传入VGG16对应的matlab数据文件路径），默认值：`data/imagenet-vgg-verydeep-19.mat`
- `--content-weight`: 损失网络中的内容权重，默认值：`7.5e0`
- `--style-weight`: 损失网络中的风格权重，默认值：`1e2`
- `--tv-weight`: 损失网络中总变异项的权重，默认值：`2e2`
- `--learning-rate`: 优化器的学习率，默认值：`1e-3`
- `--slow`: 该参数为debug损失网络而设置，采用Gatys论文中的方法，直接在像素层面渲染，`test`参数对应的文件作为debug内容，`test_dir`对应的目录用来保存debug结果


## evaluate.py
`evaluate.py` 评估画风迁移网络（需指定模型文件）， 如用多张图片评估，则图片尺寸必须相同

**参数**
- `--checkpoint`: 模型文件的路径（一般为一个目录），或`ckpt`文件路径，必选
- `--in-path`: 待渲染的单张图片路径，或待渲染的多张图片所在的目录，必选
- `--out-path`: 单张图片渲染后的保存路径，多张图片渲染后的保存目录，必选
- `--device`: 画风渲染的设备（cpu或gpu），默认值：`/cpu:0`
- `--batch-size`: 批量渲染图片时的批量大小，仅当带渲染对象是目录下的一批图片时，该参数有效，默认值：`4`
- `--allow-different-dimensions`: 批量渲染时，允许不同的图片尺寸，默认值：不允许

## transform_video.py
`transform_video.py` 渲染视频为指定画风

**参数**
- `--checkpoint`: 模型文件路径（一般为一个目录），或`ckpt`文件路径，必选
- `--in-path`: 待渲染视频文件路径，必选
- `--out-path`: 渲染后视频文件存储路径，必选
- `--tmp-dir`: 渲染过程中产生的临时文件存储目录，如未设置该值，则渲染程序自动生成目录，并在程序结束前自动删除。默认值：随机生成隐藏目录，并自动删除
- `--device`: 用于画风渲染的设备（cpu或gpu），默认值：`/gpu:0`
- `--batch-size`: 批量渲染时的批量大小，默认值：`4`
