from pathlib import Path
import requests
from time import perf_counter
import glob
import os
import sys; sys.path.append('..')
from demo_client import IFRClient, file2base64, to_chunks

START_SCRIPT_PATH = '/home/ichernoglazov/InsightFace-REST'

class Benchmark:
    def __init__(self, images_dir: str, host='http://localhost', port=18081, repeat=100):
        '''
        images_dir - str path to dir test images
        '''
        self.images = list(Path(images_dir).iterdir())
        self.images_dir = images_dir
        self.images_count = len(self.images)
        self.host = host
        self.port = port
        self.draw_url = f"{host}:{port}/multipart/draw_detections"
        self.repeat = repeat

    def start(self):
        total_time = []
        for step in range(self.repeat):
            step_time = 0
            print(f"Step {step} of {self.repeat} started")
            for idx, image in enumerate(self.images):
                files = {"file": open(image, "rb")}
                start = perf_counter()
                requests.post(self.draw_url, files=files)
                request_time = perf_counter() - start
                step_time += request_time
                print(f"Image {idx + 1} of {self.images_count}", end="\r")
            total_time.append(step_time)
        mean_time = sum(total_time) / len(total_time)
        return f"RPS: {self.images_count / mean_time}"
    
    def start_det(self):
        # start test for detection
        # request api for /extract
        client = IFRClient(self.host, self.port)
        files = glob.glob(os.path.join(self.images_dir, '*.*'))
        print('Images will be sent in base64 encoding')
        files = [file2base64(file) for file in files]
        batches = to_chunks(files, size=1)
        batches = [list(chunk) for chunk in batches]
        
        total_time = []

        for step in range(self.repeat):
            step_time = 0
            print(f"Step {step} of {self.repeat} started")
            for idx, batch in enumerate(batches):
                start = perf_counter()
                client.extract(data=batch, 
                            mode='data',
                                extract_embedding=False)
                request_time = perf_counter() - start
                step_time += request_time
                print(f"Image {idx + 1} of {self.images_count}", end="\r")
            total_time.append(step_time)
        mean_time = sum(total_time) / len(total_time)
        return f"RPS: {self.images_count / mean_time}"

# bm = Benchmark()
# print(bm.start())
