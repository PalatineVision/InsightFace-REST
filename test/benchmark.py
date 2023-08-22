from pathlib import Path
from time import perf_counter
import glob
import os
import sys; sys.path.append('..')
from demo_client import IFRClient, file2base64, to_chunks
from itertools import islice, cycle

START_SCRIPT_PATH = '/home/ichernoglazov/InsightFace-REST'

class Benchmark:
    def __init__(self, images_dir: str, host='http://localhost', port=18081, repeat=100000):
        '''
        images_dir - str path to dir test images
        '''
        self.images_dir = images_dir
        self.host = host
        self.port = port
        self.draw_url = f"{host}:{port}/multipart/draw_detections"
        self.repeat = repeat

    def start(self, det_rec_mode, batch_size):
        # start test for detection
        # request api for /extract
        kwargs = dict(mode='data')
        if det_rec_mode == 'det':
            kwargs['extract_embedding'] = False
        elif det_rec_mode == 'rec':
            kwargs['embed_only'] = False

        client = IFRClient(self.host, self.port)
        files = glob.glob(os.path.join(self.images_dir, '*.*'))
        print('Images will be sent in base64 encoding')
        files = [file2base64(file) for file in files]
        files = islice(cycle(files), self.repeat)
        batches = to_chunks(files, size=batch_size)
        batches = [list(chunk) for chunk in batches]
        
        total_time = []
        for idx, batch in enumerate(batches):
            print(f"Batch {idx + 1} of {len(batches)} length of batch {len(batch)}", end="\r")
            start = perf_counter()
            client.extract(data=batch, **kwargs)
            request_time = perf_counter() - start
            total_time.append(request_time)
        return f"RPS: {self.repeat / sum(total_time)}"

# bm = Benchmark()
# print(bm.start())
