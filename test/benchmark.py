from pathlib import Path
import requests
from time import perf_counter


class Benchmark:
    def __init__(self, images_dir: str, url='http://localhost:18081', repeat=100):
        '''
        images_dir - str path to dir test images
        '''
        self.images = list(Path(images_dir).iterdir())
        self.images_count = len(self.images)
        self.url = f"{url}/multipart/draw_detections"
        self.repeat = repeat

    def start(self):
        total_time = []
        for step in range(self.repeat):
            step_time = 0
            print(f"Step {step} of {self.repeat} started")
            for idx, image in enumerate(self.images):
                files = {"file": open(image, "rb")}
                start = perf_counter()
                requests.post(self.url, files=files)
                request_time = perf_counter() - start
                step_time += request_time
                print(f"Image {idx + 1} of {self.images_count}", end="\r")
            total_time.append(step_time)
        mean_time = sum(total_time) / len(total_time)
        return f"RPS: {self.images_count / mean_time}"


# bm = Benchmark()
# print(bm.start())
