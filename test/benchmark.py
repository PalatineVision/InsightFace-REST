from time import perf_counter
import glob
import os
import sys
sys.path.append('..')
import traceback
from itertools import islice, cycle
import json

from demo_client import IFRClient, file2base64, to_chunks
from alert import alert

class Benchmark:
    def __init__(self, images_dir: str, host='http://localhost', port=18081, cpu=False, repeat=100000):
        '''
        images_dir - str path to dir test images
        '''
        self.images_dir = images_dir
        self.host = host
        self.port = port
        self.repeat = repeat
        self.is_cpu = cpu

    def start(self, det_rec_mode, batch_size):
        # start test for detection
        # request api for /extract
        kwargs = dict(mode='data')
        if det_rec_mode == 'det':
            kwargs['extract_embedding'] = False
        elif det_rec_mode == 'rec':
            kwargs['embed_only'] = True

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
            while True:
                try:
                    start = perf_counter()
                    # throw if sess.post connection abort or ... if message in correct
                    client.extract(data=batch, **kwargs)
                    request_time = perf_counter() - start
                    break
                except Exception as e:
                    # try again
                    alert(test=f'{self.host}:{self.port}')
                    alert(test=f'{det_rec_mode} b={batch_size} {" cpu" if self.is_cpu else ""}')
                    alert(test=f'{json.dumps(client.server_info(show=False), indent=4)}')
                    alert(test=''.join(repr(traceback.format_exception(e))))
                    
            total_time.append(request_time)
        
        rps = f"RPS: {self.repeat / sum(total_time)}"
        print(rps)
        info = client.server_info()
        alert(test=f'Finished\n{self.host}:{self.port}')
        alert(test=f'{info["models"]["det_name"]} {info["models"]["rec_name"]} type: {self.images_dir} b={batch_size}{" cpu" if self.is_cpu else ""}')
        alert(test=rps)
        return rps
    