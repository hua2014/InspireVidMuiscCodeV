# InspireVidMuiscCodeV

> InspireVidMuisc项目的 源码管理版本（仅涉及InspireMusic和VideMuse的源码，不以子模块的方式进行管理，不涉及二级子模块）； 以下信息copy自InspireVidMuisc；

# InspireVidMuisc
Jion VidMuse-LSTV-M into InspireMuisc to surport Video Mode 

## InspireMuisc + VidMuse

- fork InspireMuisc: https://github.com/hua2014/FunMusic
- fork VidMuse: https://github.com/hua2014/VidMuse

```
git submodule add https://github.com/hua2014/FunMusic.git InspireMusic
git submodule add https://github.com/hua2014/VidMuse.git VidMuse-main
```

添加或更新子模块后，需在父项目中提交变更；子模块如有更新（一般在各自单独的项目中维护），需要父项目中同步更新至最新提交；

# [子模块常用操作](https://blog.csdn.net/lianghudream/article/details/148654036)

## 添加
添加子模块：`git submodule add <仓库URL> <本地路径>`
- 示例：`git submodule add https://github.com/user/lib.git libs/lib`

提交变更：
```py
git add .gitmodules libs/lib
git commit -m "添加子模块 lib"
```

## 克隆

克隆含子模块的项目
```py
# 方法 1（推荐递归克隆）：
git clone --recursive <主仓库URL>

# 方法 2（分步初始化）：
git clone <主仓库URL>
cd <主项目目录>
git submodule init      # 初始化配置
git submodule update    # 拉取子模块代码
```

## 更新
更新子模块:

| 场景	| 命令 | 
| ----	| --- | 
| 更新所有子模块到最新提交	| `git submodule update --remote --recursive` | 
| 更新指定子模块	|  `git submodule update --remote <子模块路径>` | 
| 切换到子模块特定版本	| `bash cd <子模块路径> git checkout <分支/标签/Commit>` | 


提交主仓库变更：子模块更新后需提交新 Commit ID：
```py
git add <子模块路径>
git commit -m "更新子模块版本"
```


## 子模块的日常维护
拉取子模块最新代码：
```py
git submodule foreach git pull origin master  # 所有子模块拉取 master 分支
```

批量操作所有子模块
```py
git submodule foreach --recursive 'git checkout main'  # 所有子模块切到 main 分支
```




# 20250926_InspireMusicVidMuse修改点汇总

方案：从VidMuse中拆出LSTV-M模块的推理，做为InspireMusic的前置工具进行调用，提取视频的特征，扩充InspireMusic的输入模态，支持输入该特征的训练和推理，完成为视频生成配乐的任务；

环境：使用Echoink-r1的基础镜像进行环境初始化（主要是该镜像中成功安装了flash_attn），然后参照InspireMusic的readme初始化所需环境，对于VidMuse的推理所需环境，参考其requirements中的版本进行必要库的安装；

配置：单张H20 96Gx1 pytorch2.6.0+cu124（<=12.7）Python==3.11.13


## VidMuse

方案：沿用VidMuse原始的推理路径，通过新增推理方法以支持对video_emb的提取；

### 前置修改

下载模型权重文件

下载openai/clip-vit-base-patch32，修改`/audiocraft/models/lm.py` LMModel初始化方法中载入clip为本地模型路径：

```py
def __init__...
    ...
    if self.visual_encoder == 'clip':
        self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained("/root/autodl-tmp/VidMuse-main/pretrained_model/openai_clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("/root/autodl-tmp/VidMuse-main/pretrained_model/openai_clip-vit-base-patch32")
```


由于‘compression_state_dict.bin’这个模型[权重文件](https://huggingface.co/HKUSTAudio/VidMuse/tree/main)，没能对应到符合格式的‘facebook/encodec_32kh’，因此推理中涉及该模型的代码全部予以注释，该模型载入方法返回设置为None（注意这样做后，推理无法再生成音频）；

```py
# /audiocraft/models/loaders.py
def load_compression_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    # pkg = load_compression_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    # if 'pretrained' in pkg:
    #     return CompressionModel.get_pretrained(pkg['pretrained'], device=device)
    # cfg = OmegaConf.create(pkg['xp.cfg'])
    # cfg.device = str(device)
    # model = builders.get_compression_model(cfg)
    # model.load_state_dict(pkg['best_state'])
    # model.eval()
    # return model

    return None
```


```py
# /audiocraft/models/vidmuse.py  VidMuse类
def __init__...
    ...
    # self.compression_model.eval()


@property
def frame_rate(self) -> float:
    """Roughly the number of AR steps per seconds."""
    return 50# self.compression_model.frame_rate

@property
def sample_rate(self) -> int:
    """Sample rate of the generated audio."""
    return 32000# self.compression_model.sample_rate

@property
def audio_channels(self) -> int:
    """Audio channels of the generated audio."""
    return 1# self.compression_model.channels
```


### `audiocraft/models/lm.py`

为LMModel新增方法 generate_video_emb
- 注意去掉cfg的部分


```py
@torch.no_grad()
def generate_video_emb(self,
              prompt: tp.Optional[torch.Tensor] = None,
              conditions_list: tp.List = [],
              num_samples: tp.Optional[int] = None):
    """Generate tokens sampling from the model given a prompt or unconditionally. 
    """
    assert not self.training, "generation shouldn't be used in training mode."
    first_param = next(iter(self.parameters()))
    device = first_param.device
    assert isinstance(conditions_list, list)
    
    assert len(conditions_list) == 2
    local_conditions = conditions_list[0]
    global_conditions = conditions_list[1]
    # Checking all input shapes are consistent.
    possible_num_samples = []
    if num_samples is not None:
        possible_num_samples.append(num_samples)
    elif prompt is not None:
        possible_num_samples.append(prompt.shape[0])
    elif local_conditions is not None:
        possible_num_samples.append(len(local_conditions))
    else:
        possible_num_samples.append(1)
        
    assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
    num_samples = possible_num_samples[0]

    # local_cfg_conditions: CFGConditions
    # global_cfg_conditions: CFGConditions
    # local_null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(local_conditions)
    # local_cfg_conditions = torch.cat((local_conditions, local_null_conditions), dim=0)
    # global_null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(global_conditions)
    # global_cfg_conditions = torch.cat((global_conditions, global_null_conditions), dim=0)
    # print("\t local_cfg_conditions size = ", local_cfg_conditions.size(), local_conditions.size())
    # print("\t global_cfg_conditions size = ", global_cfg_conditions.size(), global_conditions.size())
    # video_hidden, video_emb = self.compute_video_emb([local_cfg_conditions, global_cfg_conditions], device=device)
    video_hidden, video_emb = self.compute_video_emb([local_conditions, global_conditions], device=device)

    # print("lm.py video_hidden size = ", video_hidden.size())
    return video_hidden, video_emb
```

### `audiocraft/models/vidmuse.py`

为VidMuse类增加_generate_video_embs方法；并同步修改调用条件，为generate方法新增控制参数return_video_emb以执行 _generate_video_embs方法：

```py
def generate(self, descriptions_list: tp.List, progress: bool = False, return_tokens: bool = False, return_video_emb:bool=False)...
    ...
    assert isinstance(descriptions_list,list)
    assert len(descriptions_list)<=2

    assert len(descriptions_list)==2
    local_descriptions=[descriptions_list[0]]
    global_descriptions=[descriptions_list[1]]

    local_attributes = torch.stack(local_descriptions)
    global_attributes = torch.stack(global_descriptions)

    prompt_tokens = None
    assert prompt_tokens is None
    

    assert len(descriptions_list)==2
    if return_video_emb:
        video_emb = self._generate_video_embs([local_attributes, global_attributes])
        return video_emb
    tokens = self._generate_tokens([local_attributes, global_attributes], prompt_tokens, progress)
    return tokens

```

```py
def _generate_video_embs(self, attributes: tp.List) -> torch.Tensor:
    """Generate discrete audio tokens given audio prompt and/or conditions.
    """
    # 输入数据不分片 
    with self.autocast:
        video_hidden, video_emb = self.lm.generate_video_emb(
            None, attributes)

    return video_hidden

    # 输入数据分片
    # self.max_duration = 30
    
    # if self.duration <= self.max_duration:
    #     # generate by sampling from LM, simple case.
    #     with self.autocast:
    #         video_hidden, video_emb = self.lm.generate_video_emb(
    #             None, attributes)
    #         print("vidmuse <30.py video_hidden size = ", video_hidden.size())
    # else:
    #     self.fps = 2
    #     video_frame_rate = 50
    #     # 要生成的视频特征数量
    #     total_gen_len = int(self.duration * self.fps * video_frame_rate)
    #     # 已生成的视频特征数量
    #     current_gen_offset: int = 0
        
    #     # now this gets a bit messier, we need to handle prompts,
    #     # melody conditioning etc.
    #     all_tokens = []
                            
    #     # 如果还有要生成的特征
    #     while current_gen_offset < total_gen_len:
    #         # 已生成特征数 换算到 秒数
    #         time_offset = current_gen_offset / video_frame_rate / self.fps
    #         # 确定当前要处理的视频时间块
    #         chunk_duration = min(self.duration - time_offset, self.max_duration) 

    #         with self.autocast:
    #             assert len(attributes)==2
    #             video_hidden, video_emb = self.lm.generate_video_emb(
    #                 None, [attributes[0][:,:,:int(chunk_duration*self.fps),:,:], attributes[1]])

            
    #         all_tokens.append(video_hidden)
            
    #         if attributes[0].shape[2]-int(chunk_duration*self.fps) < self.max_duration*self.fps:
    #             # # 如果当前全部局部视频帧数 减去 本次处理时间块的帧数 小于 最大可处理时间块：表示下次迭代将是最后一次处理
    #             # 直接取 最后 小于 max_duration的全部
    #             attributes[0]=attributes[0][:,:,-self.max_duration*self.fps:,:,:]
    #         else:
    #             # 如果当前全部视频帧数 减去 本次处理时间块的帧数 大于等于 最大可处理时间块：：表示下次迭代仍以max_duration进行
    #             # 重新定位 全部局部视频帧 的起始位置
    #             attributes[0]=attributes[0][:,:,int(chunk_duration*self.fps):,:,:]
            
    #         # 我们本次计算得到的视频特征数 与 参数计算的时间块之间的换算关系
    #         assert video_hidden.shape[1] == int(chunk_duration*self.fps * video_frame_rate)
    #         # 更新 已生成的视频特征数量
    #         current_gen_offset += video_hidden.shape[1]

    #     video_hidden = torch.cat(all_tokens, dim=1)
    # # 1 ，T x fps x FrameRate，768
    # # 1， T x fps x FrameRate x N，768
    return video_hidden
```        

### `demos/VidMuse_app.py`

新增‘_do_predictions_for_get_video_emb’方法：

```py
def _do_predictions_for_get_video_emb(video, duration, progress=False, gradio_progress=None, **gen_kwargs):
    
    fps = 2
    progress=True
    USE_DIFFUSION=False

    video_path = video[0]
    duration = int(get_video_duration(video_path))
    MODEL.set_generation_params(duration=duration, **gen_kwargs)

    local_video_tensor, global_video_tensor = video_read_global(video_path, seek_time=0., duration=duration, target_fps=fps)

    try:
        outputs = MODEL.generate([local_video_tensor, global_video_tensor], progress=progress, return_tokens=USE_DIFFUSION, return_video_emb=True)
    except RuntimeError as e:
        raise 
    
    outputs = outputs.detach().cpu().float()
    # print("video emb outputs.shape = ", outputs.shape)
    return outputs
```    

### 新建一个测试推理的脚本

参照‘infer.py’新建‘infer_for_inspiremusic.py’，调用_do_predictions_for_get_video_emb方法测试推理；

```py
import argparse
from pathlib import Path
from demos.VidMuse_app import load_model, _do_predictions_for_get_video_emb

def infer(video_dir, output_dir):
    # Get all mp4 files in the directory
    video_files = list(Path(video_dir).glob('*.mp4'))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file_path = Path(output_dir) / f"utt2videoemb_output.pt"
    if output_file_path.exists():
        print(f"Audio file already exists, skipping: {output_file_path}")
        return         
    for video_path in video_files:   
        video_embs = _do_predictions_for_get_video_emb(
            [str(video_path)], duration=30
        )
        print(f"Generated video file: {video_embs.shape}")

        # 需要参考inspiremusic的实现 将特征 写入 output_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VidMuse inference script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    parser.add_argument('--video_dir', type=str, required=True, help='Path to the input video directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    
    args = parser.parse_args()
    
    load_model(args.model_path)
    infer(args.video_dir, args.output_dir)
```    

修改infer.sh所使用的推理脚本，测试video_emb是否正确提取：

```sh
#!/bin/bash

model_path=./model
video_dir=./dataset/example/infer
output_dir=./result/

# python3 infer.py \
python3 infer_for_inspiremusic.py \
    --model_path $model_path \
    --video_dir $video_dir \
    --output_dir $output_dir
```

### 推理过程张量变化

```
一段5s视频在VidMuse原有推理过程的张量size变化：
local_conditions size = torch.Size([1, 3, 10, 224, 224])
global_conditions size = torch.Size([1, 3, 32, 224, 224])
        local_image size =  torch.Size([2, 3, 10, 224, 224])
        global_image size =  torch.Size([2, 3, 32, 224, 224])
        local_image size size=  torch.Size([20, 3, 224, 224])
        global_image size =  torch.Size([64, 3, 224, 224])
        local_pixel_values size=  torch.Size([20, 3, 224, 224])
        global_pixel_values size =  torch.Size([64, 3, 224, 224])
        local_video_hidden size =  torch.Size([2, 500, 768])
        global_video_hidden size =  torch.Size([2, 1600, 768])
        video_hidden size =  torch.Size([2, 500, 768])
        video_emb size =  torch.Size([2, 500, 1536])
video_emb size =  torch.Size([2, 500, 1536])
outputs.shape =  torch.Size([1, 4, 250])
```

`video_hidden size =  torch.Size([2, 500, 768])` 
- 这里的2 是因为加了cfg 如果去掉 则为1
- 我们预期的输出 video_hidden 的shape为 `[1, dur x 2 x 50, 768]`

## InspireMusic

方案：围绕原InspireMusic的训练和推理进行修改，输入数据前置的特征提取增加video_emb，修改涉及组织数据的流程；


### 前置准备

下载模型 1.5b-long，参照readme执行相应的推理测试（推理速度很慢，30s音频生成耗时约4~5min）；

构建测试用数据集dataset_MAP3000，包含video-audio-text，按照示例格式进行数据组织（train-dev=>9-1），音视频固定30s时长；该数据集既可以测试原InspireMusic，也可以测试修改后的程序；
- 原始数据 分别以 mp4 wav 和 txt格式放到dataset_MAP3000的video audio 和text文件夹中，并附加“dataset_MAP3000/datalist_all.xlsx”进行数据说明；
- 通过脚本“`dataset_MAP3000/dataset_create_like_sample.py`”生成 dataset_MAP3000_dev和 dataset_MAP3000_train，供后续测试；
- 注：参照样例数据，可以方便组织数据，这里不展示具体脚本；


修改`run.sh`

```
model_name=InspireMusic-1.5B-Long
dataset_name=dataset_MAP3000

使用的tools各个脚本前加python，否则会提示执行权限问题
由于测试数据有限，我们将make_parquet的参数做调整：num_utts_per_parquet 100
make_parquet_list.py中为避免torch.load()加载.pt文件时出现的weights_only安全性限制问题，将 weights_only=False
对于推理阶段，我们直接使用infer_1.5b_long.sh 可正常推理，run.sh中的推理不做测试
对于训练部分，flow相关的会被注释掉，我们只关注llm train

expr_name="InspireMusic-1.5b-long-datasetMAP3000-ftxxxxxx-x"
CUDA_VISIBLE_DEVICES="0"
num_workers=8

config conf/inspiremusic_1.5b_long.yaml
checkpoint ../../pretrained_models/InspireMusic-1.5B-Long/llm.pt

```




`inspiremusic_1.5b_long.yaml`配置文件修改：
- dtype: 'fp32'
- max_frames_in_batch: 12000 
- warmup_steps: 500
- max_epoch: 10

```
llm: !new:inspiremusic.transformer.qwen_encoder.QwenEmbeddingEncoder
    input_size: !ref <text_encoder_input_size>
    pretrain_path: !ref <basemodel_path>
    dtype: 'fp32'


batch: !name:inspiremusic.dataset.processor.batch
    batch_type: 'dynamic'
    max_frames_in_batch: 12000 # llm 12000


# train conf
train_conf:
    optim: adam
    optim_conf:
        lr: 0.0001 # change to 0.001 if you want to train flow from scratch
    scheduler: warmuplr
    scheduler_conf:
        warmup_steps: 500
    max_epoch: 10
    grad_clip: 5
    accum_grad: 2
    log_interval: 100
    save_per_step: 500
```


测试原InspireMusic的`run.sh`：

- stage0：准备数据集
- stage1：acoustiv_token，这个特征对组织数据不可或缺，虽然实际不会用到
- stage2：semantic_token
- stage3：make_parquet
- stage5：llm train

原InspireMusic可以正常训练，可通过`tensorboard --port 6006 --logdir ../xxx/`或`tensorboard --port 6007 --logdir ../xxx/`可视化训练过程；


### 数据基础特征提取+video_emb

> 相关脚本实现 见同名附件；

- `tools/extract_video_emb.py`
  - 由于提取的是多维向量，比较大，因此将各个视频特征存储在单独的pt文件中；
- `tools/make_parquet_list.py`
  - 组织parquet 会按照num_utts_per_parquet=100组织，对应parquet约1.3G；

在run.sh中 新增stage6 用于启动提取视频特征的脚本；新的测试顺序为：stage0 stage1 stage2 stage6 stage3 stage5；

### 数据预处理+video_emb

- `inspiremusic/dataset/processor.py`
  - padding：仅在pipline的最后一步padding中，读取sample的video_emb的key，组织成batch


### 模型的训练和推理+video_emb


- `inspiremusic/llm/llm.py`
  - LLM的初始化方法
  - LLM的forward方法及关联方法
  - LLM的generate方法（注意 batch_inference方法未做修改）
  - 具体拓展了一个新的任务id task_text_video_to_music
  - 调整了llm_input和llm_target的数据组织顺序：sos embeddings text_tokens sos video_tokens ours_task_id，详见pad_unpad_sequence方法；

这里附一张杨老师的讲解图：

![杨老师做的讲解说明](./static/杨老师做的讲解说明.png)


后续修改沿着对generate的调用进行；

### 推理相关+video_emb

- `inspiremusic/bin/inference.py`
  - 新增任务类型‘bd-task1’，并支持带video_emb参数字典的组织；
- `inspiremusic/bin/train.py`
  - 由于我们改变了模型结构中的llm_embedding，并新增了visual_feature_proj，还需要修改载入原InspireMusic Checkpoint模型权重的代码；
- `inspiremusic/cli/model.py`
  - inference推理增加对video_emb的支持；

### 其他

软连接失效重新创建：
- `ln -s /root/autodl-tmp/InspireMusic/inspiremusic /root/autodl-tmp/InspireMusic/examples/music_generation/inspiremusic`
- `ln -s /root/autodl-tmp/InspireMusic/tools /root/autodl-tmp/InspireMusic/examples/music_generation/tools`

30s视频特征长度将达到3000，会提示显存溢出，为了做测试，我们在`inspiremusic/dataset/processor.py`中对video_emb做了切片，只取了300：
- `video_emb = torch.tensor(np.stack(video_emb)[:,:300,:] , dtype=torch.float32)`
- 注意有 train和inference 两处修改；


llm中张量变化情况如下：

```
exp1：
text_token  torch.Size([3, 800])  text_token_len  tensor([800, 390, 382], device='cuda:0', dtype=torch.int32)
audio_token  torch.Size([3, 2250])  audio_token_len  tensor([2250, 2250, 2250], device='cuda:0', dtype=torch.int32)

video_emb  torch.Size([3, 300, 768])  video_emb_len  tensor([300, 300, 300], device='cuda:0', dtype=torch.int32)
text_embedding text_token torch.Size([3, 800, 1536])  text_token_len  tensor([800,   0, 382], device='cuda:0', dtype=torch.int32)
speech_embedding audio_token torch.Size([3, 2250, 1536])  audio_token_len  tensor([2250, 2250, 2250], device='cuda:0', dtype=torch.int32)

visual_feature_proj video_token torch.Size([3, 300, 1536])  video_emb_len  tensor([300, 300, 300], device='cuda:0', dtype=torch.int32)
pad_unpad_sequence lm_input torch.Size([3, 3356, 1536])  lm_input_len  tensor([3356, 2556, 2938], dtype=torch.int32)

llm lm_output torch.Size([3, 3356, 1536])


exp2：
text_token  torch.Size([3, 1505])  text_token_len  tensor([1505,  366,  387], device='cuda:0', dtype=torch.int32)
audio_token  torch.Size([3, 2250])  audio_token_len  tensor([2250, 2250, 2250], device='cuda:0', dtype=torch.int32)

video_emb  torch.Size([3, 300, 768])  video_emb_len  tensor([300, 300, 300], device='cuda:0', dtype=torch.int32)
text_embedding text_token torch.Size([3, 1505, 1536])  text_token_len  tensor([1505,  366,    0], device='cuda:0', dtype=torch.int32)
speech_embedding audio_token torch.Size([3, 2250, 1536])  audio_token_len  tensor([2250, 2250, 2250], device='cuda:0', dtype=torch.int32)

visual_feature_proj video_token torch.Size([3, 300, 1536])  video_emb_len  tensor([300, 300, 300], device='cuda:0', dtype=torch.int32)
pad_unpad_sequence lm_input torch.Size([3, 4061, 1536])  lm_input_len  tensor([4061, 2922, 2556], dtype=torch.int32)

llm lm_output torch.Size([3, 4061, 1536])


```

训练过程可视化：

![训练测试日志可视化](./static/训练测试日志可视化.png)


### 待测事宜

- review代码；
- 测试训练所得模型支持Video_emb的实际推理过程；
