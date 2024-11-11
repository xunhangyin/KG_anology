## 多模态知识图谱类比推理项目

文件夹描述：

CLIP:用于存放huggingface的CLIP初始权重

Code:用于存放代码。老师目前我们只需要考虑里面的CLIP_Code文件夹里的代码。其他的目前没用。

dataset:存放数据集的地方

其他文件夹暂时没用，不考虑。

运行的时候首先把CLIP_large权重从huggingface上下载放到CLIP文件夹下，

[openai/clip-vit-large-patch14 at main](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)

之后将数据放入dataset，dataset里就按照原论文里给的数据，两个文件夹放入即可，分别为MARS和MarKG

环境的话torch2.0版本以及transformers 4.40.0以及PIL

emmm，上传的时候发现git没有把空的文件夹传上来，老师跑的话要自己创建dataset和CLIP文件夹。就在项目根目录里即可,和Code文件夹一样.
