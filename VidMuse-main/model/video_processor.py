from moviepy.editor import VideoFileClip, AudioFileClip
import torch
from decord import VideoReader, cpu
import math
import einops
import torchvision.transforms as transforms

class VideoProcessor:
    def __init__(self):
        self.resize_transform = transforms.Resize((224, 224))

    def get_video_duration(self, video_path):
        try:
            clip = VideoFileClip(video_path)
            duration_sec = clip.duration
            clip.close()
            return duration_sec
        except Exception as e:
            print(f"Error: {e}")
            return None

    def adjust_video_duration(self, video_tensor, duration, target_fps):
        current_duration = video_tensor.shape[1]
        target_duration = duration * target_fps

        if current_duration > target_duration:
            video_tensor = video_tensor[:, :int(target_duration)]
        elif current_duration < target_duration:
            last_frame = video_tensor[:, -1:]
            repeat_times = int(target_duration - current_duration)
            video_tensor = torch.cat((video_tensor, last_frame.repeat(1, repeat_times, 1, 1)), dim=1)
        return video_tensor

    def video_read_global(self, filepath, seek_time=0., duration=-1, target_fps=2, global_mode='average', global_num_frames=32):
        vr = VideoReader(filepath, ctx=cpu(0))
        fps = vr.get_avg_fps()
        frame_count = len(vr)

        if duration > 0:
            total_frames_to_read = target_fps * duration
            frame_interval = int(math.ceil(fps / target_fps))
            start_frame = int(seek_time * fps)
            end_frame = int(start_frame + frame_interval * total_frames_to_read)
            frame_ids = list(range(start_frame, min(end_frame, frame_count), frame_interval))
        else:
            frame_ids = list(range(0, frame_count, int(math.ceil(fps / target_fps))))

        local_frames = vr.get_batch(frame_ids)
        local_frames = torch.from_numpy(local_frames.asnumpy()).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        
        local_frames = [self.resize_transform(frame) for frame in local_frames]
        local_video_tensor = torch.stack(local_frames)
        local_video_tensor = einops.rearrange(local_video_tensor, 't c h w -> c t h w') # [T, C, H, W] -> [C, T, H, W]
        local_video_tensor = self.adjust_video_duration(local_video_tensor, duration, target_fps)

        if global_mode=='average':
            global_frame_ids = torch.linspace(0, frame_count - 1, global_num_frames).long()

            global_frames = vr.get_batch(global_frame_ids)
            global_frames = torch.from_numpy(global_frames.asnumpy()).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
            
            global_frames = [self.resize_transform(frame) for frame in global_frames]
            global_video_tensor = torch.stack(global_frames)
            global_video_tensor = einops.rearrange(global_video_tensor, 't c h w -> c t h w') # [T, C, H, W] -> [C, T, H, W]

        assert global_video_tensor.shape[1] == global_num_frames, f"the shape of global_video_tensor is {global_video_tensor.shape}"
        return local_video_tensor, global_video_tensor

    def process(self, video_path, target_fps=2, global_mode='average', global_num_frames=32):
        duration = self.get_video_duration(video_path)
        if duration is None:
            raise ValueError("Invalid video path or video file.")
        local_video_tensor, global_video_tensor = self.video_read_global(video_path, duration=duration, target_fps=target_fps, global_mode=global_mode, global_num_frames=global_num_frames)
        return local_video_tensor, global_video_tensor, duration


def merge_video_audio(video_path, audio_path, output_path):
    video = VideoFileClip(video_path).without_audio()
    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')