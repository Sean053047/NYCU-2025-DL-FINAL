import json
import cv2
import numpy as np
import csv
import decord
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        #     for line in f:
        #         self.data.append(json.loads(line))
        DATA_ROOT = '/eva_data5/kuoyuhuan/DLP_final/data'
        with open(f"{DATA_ROOT}/metadata.csv", 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for _ in range(81):
                    self.data.append({
                        'cam_video': f"{DATA_ROOT}/{row['cam_video']}",
                        'lidar_video': f"{DATA_ROOT}/{row['lidar_video']}",
                        'prompt': row['prompt'],
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        cam_video_filename = item['cam_video']
        lidar_video_filename = item['lidar_video']
        prompt = item['prompt']

        # source = cv2.imread('./training/fill50k/' + source_filename)
        # target = cv2.imread('./training/fill50k/' + target_filename)
        source_reader = decord.VideoReader(cam_video_filename)
        target_reader = decord.VideoReader(lidar_video_filename)

        # Read random frame from the video.
        # frame_index = np.random.randint(0, len(source_reader))
        frame_index = idx % 81
        source = source_reader[frame_index].asnumpy()
        target = target_reader[frame_index].asnumpy()
        # crop to 448x832
        source = source[0:448, 0:832, :]
        target = target[0:448, 0:832, :]

        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1
        del source_reader, target_reader
        return dict(jpg=source, txt=prompt, hint=target)

if __name__ == '__main__':
    dataset = MyDataset()
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        item = dataset[i]
        print(f"Item {i}:")
        print(f"  Prompt: {item['txt']}")
        print(f"  Hint shape: {item['hint'].shape}")
        print(f"  JPG shape: {item['jpg'].shape}")
        # Uncomment to save images
        # cv2.imwrite(f'source_{i}.jpg', (item['hint'] * 255).astype(np.uint8))
        # cv2.imwrite(f'target_{i}.jpg', ((item['jpg'] + 1) * 127.5).astype(np.uint8))