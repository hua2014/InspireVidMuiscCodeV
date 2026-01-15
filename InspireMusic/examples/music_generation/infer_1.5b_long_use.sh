#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
. ./path.sh || exit 1;

export TOKENIZERS_PARALLELISM=False

model_name="InspireMusic-1.5B-Long"
pretrained_model_dir=../../pretrained_models/${model_name}
# pretrained_LLM_model=../../pretrained_models/InspireMusic-1.5b-long-BD-ESV-260106_normal_caption-s2-ft260107-2/epoch_0_whole/pytorch_model_epoch_0_whole.bin
# pretrained_LLM_model=/root/autodl-tmp/InspireVidMuisc/InspireMusic/examples/music_generation/exp/InspireMusic-1.5b-long-BD-ESV-251222_default_caption-s1-ft251230-3/llm/deepspeed/epoch_4_step_21000/mp_rank_00_model_states.pt
# pretrained_LLM_model=/root/autodl-tmp/InspireVidMuisc/InspireMusic/examples/music_generation/exp/InspireMusic-1.5b-long-BD-ESV-251222_default_caption-s2-ft260104-1/llm/deepspeed/epoch_4_step_34000/mp_rank_00_model_states.pt
# pretrained_LLM_model=/root/autodl-tmp/InspireVidMuisc/InspireMusic/examples/music_generation/exp/InspireMusic-1.5b-long-BD-ESV-260106_normal_caption-s2-ft260107-2/llm/deepspeed/epoch_0_step_5000/mp_rank_00_model_states.pt
pretrained_LLM_model=/root/autodl-tmp/InspireVidMuisc/InspireMusic/examples/music_generation/exp/InspireMusic-1.5b-long-BD-ESV-260106_normal_caption-s2-ft260113-1/llm/deepspeed/epoch_0_step_5000/mp_rank_00_model_states.pt

# pretrained_LLM_model=${pretrained_model_dir}/llm.pt

dataset_name=BD-ESV-260106_normal_caption_mini_dev

expr_name="inspiremusic_${dataset_name}"

echo "Run inference."
# Use Unix-style paths
# inference normal mode
for task in 'bd-task1'; do
  python inspiremusic/bin/inference.py --task $task \
      --gpu 0 \
      --config conf/inspiremusic_1.5b_long.yaml \
      --prompt_data data/${dataset_name}/parquet/data.list \
      --flow_model $pretrained_model_dir/flow.pt \
      --llm_model $pretrained_LLM_model \
      --music_tokenizer $pretrained_model_dir/music_tokenizer \
      --wavtokenizer $pretrained_model_dir/wavtokenizer \
      --chorus default \
      --output_sample_rate 48000 \
      --min_generate_audio_seconds 5.0 \
      --max_generate_audio_seconds 180.0 \
      --fast \
      --result_dir `pwd`/exp/${model_name}/${task}_${expr_name}
#   if use InspireMusic-xxxx-24kHz model, please set output sample rate to 24kHz
#      --output_sample_rate 24000 \
#   use fast inference mode
#      --fast # fast mode without flow matching
  echo `pwd`/exp/${model_name}/${task}_${expr_name}
done
