from typing import Tuple

import my_torchvision
from torch import Tensor


class KineticsWithVideoId(my_torchvision.datasets.Kinetics):
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, video_idx
