---
license: cc-by-4.0
library_name: audiocraft
pipeline_tag: video-to-audio
---

# VidMuse 

## VidMuse: A Simple Video-to-Music Generation Framework with Long-Short-Term Modeling

[TL;DR]: VidMuse is a framework for generating high-fidelity music aligned with video content, utilizing Long-Short-Term modeling, and has been accepted to CVPR 2025.

### Links
- **[Paper](https://arxiv.org/pdf/2406.04321)**: Explore the research behind VidMuse.
- **[Project](https://vidmuse.github.io/)**: Visit the official project page for more information and updates.
- **[Dataset](https://huggingface.co/datasets/HKUSTAudio/VidMuse-Dataset)**: Download the dataset used in the paper.

## Clone the repository
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/HKUSTAudio/VidMuse
cd VidMuse
```

## Usage

1. First install the [`VidMuse` library](https://github.com/ZeyueT/VidMuse)
```
conda create -n VidMuse python=3.9
conda activate VidMuse
pip install git+https://github.com/ZeyueT/VidMuse.git
```

2. Install ffmpeg:
Install ffmpeg:
```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```


3. Run the following Python code:


```py
from video_processor import VideoProcessor, merge_video_audio
from audiocraft.models import VidMuse
import scipy

# Path to the video
video_path = 'sample.mp4'
# Initialize the video processor
processor = VideoProcessor()
# Process the video to obtain tensors and duration
local_video_tensor, global_video_tensor, duration = processor.process(video_path)

progress = True
USE_DIFFUSION = False

# Load the pre-trained VidMuse model
MODEL = VidMuse.get_pretrained('HKUSTAudio/VidMuse')
# Set generation parameters for the model based on video duration
MODEL.set_generation_params(duration=duration)

try:
    # Generate outputs using the model
    outputs = MODEL.generate([local_video_tensor, global_video_tensor], progress=progress, return_tokens=USE_DIFFUSION)
except RuntimeError as e:
    print(e)

# Detach outputs from the computation graph and convert to CPU float tensor
outputs = outputs.detach().cpu().float()


sampling_rate = 32000
output_wav_path = "vidmuse_sample.wav"
# Write the output audio data to a WAV file
scipy.io.wavfile.write(output_wav_path, rate=sampling_rate, data=outputs[0, 0].numpy())

output_video_path = "vidmuse_sample.mp4"
# Merge the original video with the generated music
merge_video_audio(video_path, output_wav_path, output_video_path)
```


## Citation
If you find our work useful, please consider citing:

```
@article{tian2024vidmuse,
  title={Vidmuse: A simple video-to-music generation framework with long-short-term modeling},
  author={Tian, Zeyue and Liu, Zhaoyang and Yuan, Ruibin and Pan, Jiahao and Liu, Qifeng and Tan, Xu and Chen, Qifeng and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2406.04321},
  year={2024}
}
```