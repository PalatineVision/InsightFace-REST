import json
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
INSIGHT_FACE = 'http://localhost:18082'
DEBUG_LAG=120

def start_container(det, rec):
    subprocess.run(args=shlex.split(f'bash ./deploy_trt.sh {det} {rec}'), cwd=START_SCRIPT_PATH)
    for i in range(10):
        try:
            r = requests.get(f'{INSIGHT_FACE}/info')
            r.raise_for_status()
            print(r.json())
            print('Connected')
            return True
        except Exception as e:
            print(e)
            print(f'Try connect {i + 1} time. Sleep 5')
            time.sleep(5)
    return False

try:
    result = json.load(open('benchmarks.json', 'rb'))
except:
    result = {}

for det in DET_MODEL:
    for rec in REC_MODEL:
        
        if all([f'{det} {rec} {t}' in result for t in TYPE_TEST]) and \
                all([result[f'{det} {rec} {t}'] != "Fail to start container" 
                        for t in TYPE_TEST]):
            continue

        if not start_container(det, rec):
            for t in TYPE_TEST:
                if not f'{det} {rec} {t}' in result:
                    result[f'{det} {rec} {t}'] = 'Fail to start container'
            json.dump(result, open('benchmarks.json', 'w'), indent=4)
            continue

        for t in TYPE_TEST:
            if f'{det} {rec} {t}' in result: 
                r = result[f'{det} {rec} {t}']
                if r != 'Fail to start container':
                    continue
                print(f'{det} {rec} {t} result: {r}')
            print(f"Benchmark {det} {rec} {t}")
            result[f'{det} {rec} {t}'] = Benchmark(images_dir=t).start()
            json.dump(result, open('benchmarks.json', 'w'), indent=4)
        
        # time.sleep(DEBUG_LAG)


print(json.dumps(result, indent=4))
