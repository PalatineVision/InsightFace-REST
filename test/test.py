import argparse
import json
import sys
import time
import requests
from benchmark import Benchmark
import subprocess, shlex
'''
Result saved in file benchmarks.json if key not exist
'''

TYPE_TEST = ['one', 'billion', 'dir']
DET_MODEL = [
'scrfd_2.5g_gnkps',
'yolov5l-face',
'yolov5m-face',
'yolov5s-face',
'yolov5n-face',
'yolov5n-0.5',
'scrfd_500m_gnkps',
'scrfd_10g_gnkps',
'scrfd_500m_bnkps',
'scrfd_2.5g_bnkps',
'scrfd_10g_bnkps',
'centerface',
]

REC_MODEL = [
'w600k_r50',
'w600k_mbf',
'glintr100',
]
START_SCRIPT_PATH = '/home/ichernoglazov/InsightFace-REST'
DEBUG_LAG=120
PORT=18082

def start_container(det, rec, url, cpu=True, gpu=None):
    if cpu:
        subprocess.run(args=shlex.split(f'bash ./deploy_cpu.sh {det} {rec}'), cwd=START_SCRIPT_PATH)
    else:
        if gpu is None:
            subprocess.run(args=shlex.split(f'bash ./deploy_trt.sh {det} {rec}'), cwd=START_SCRIPT_PATH)
        else:
            subprocess.run(args=shlex.split(f'bash ./deploy_trt.sh {det} {rec} {gpu}'), cwd=START_SCRIPT_PATH)
    
    for i in range(10):
        try:
            r = requests.get(f'{url}/info')
            r.raise_for_status()
            print(r.json())
            print('Connected')
            return True
        except Exception as e:
            print(e)
            print(f'Try connect {i + 1} time. Sleep 5')
            time.sleep(5)
    return False

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, choices=[-1, 0, 1], default=-1)
parser.add_argument('--save-file', type=str, required=True)
parser.add_argument('--test-types', type=str, nargs='+', choices=TYPE_TEST, default=TYPE_TEST)
parser.add_argument('--det-models', type=str, nargs='+', choices=DET_MODEL, default=DET_MODEL)
parser.add_argument('--rec-models', type=str, nargs='+', choices=REC_MODEL, default=REC_MODEL)
parser.add_argument('--port', type=int, default=PORT)
args = parser.parse_args()
INSIGHT_FACE = f'http://localhost:{args.port}'
if args.gpu == -1:
    cpu = True
    gpu = None
    key_template = '{} {} {} cpu'
else:
    cpu = False
    gpu = args.gpu
    key_template = '{} {} {}'
test_types = set(args.test_types)
det_models = set(args.det_models)
rec_models = set(args.rec_models)
try:
    result = json.load(open(args.save_file, 'rb'))
except:
    result = {}

for det in det_models:
    for rec in rec_models:
        
        if all([ key_template.format(det, rec, t) in result for t in test_types]) and \
                all([result[key_template.format(det, rec, t)] != "Fail to start container" 
                        for t in test_types]):
            continue

        if not start_container(det, rec, INSIGHT_FACE, cpu, gpu):
            for t in test_types:
                key = key_template.format(det, rec, t)
                if not key in result:
                    result[key] = 'Fail to start container'
            json.dump(result, open(args.save_file, 'w'), indent=4)
            continue

        for t in test_types:
            key = key_template.format(det, rec, t)
            if key in result: 
                r = result[key]
                if r != 'Fail to start container':
                    continue
                print(f'{key} result: {r}')
            print(f"Benchmark {key}")
            result[key] = Benchmark(images_dir=t, url=INSIGHT_FACE).start()
            json.dump(result, open(args.save_file, 'w'), indent=4)
        
        # time.sleep(DEBUG_LAG)


print(json.dumps(result, indent=4))
