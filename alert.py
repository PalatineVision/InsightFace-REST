import time
import nvsmi
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)

import urllib
import requests

ALERT_TIME = 180
POLL = 10

def alert(gid=None, test='Test'):
    CHAT = '-1001535956891'
    TOKEN = '6243613144:AAF9TIeA2eyuj_9ri8mpSSpfKYLNNALGS7A'
    # send alert
    url = 'https://api.telegram.org/bot%s/sendMessage?chat_id=%s&text=%s' % (
    TOKEN, CHAT, urllib.parse.quote_plus(f'gpu {gid} not work!' if gid else test))
    try:
        _ = requests.get(url, timeout=60)
    except:
        logging.error('Failed to alert!')

def main():
    thresh = ALERT_TIME // POLL
    util_zero = {}
    for g in nvsmi.get_gpus():
        util_zero[g.id] = 0

    i = 0
    while True:
        for g in nvsmi.get_gpus():
            if g.gpu_util < 1e-2:
                util_zero[g.id] += 1
            else:
                util_zero[g.id] = 0
        
        if (i + 1) % thresh == 0:
            logging.info(util_zero)
            for gid in util_zero:
                if util_zero[gid] >= thresh:
                    alert(gid=gid)
                    util_zero[gid] = 0
        i += 1
        time.sleep(POLL)

if __name__ == '__main__':
    main()