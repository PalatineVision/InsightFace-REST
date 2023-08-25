# Run tests

```bash
cd ./test
unzip if-speedtest.zip
python ./test.py --gpu 0 --mode det+rec --save-file det+rec_benchmark.json --port 18082 -b 64
```

Parameters:

--gpu - num device 0,1, if omitted cpu used

--save-file - file to load from in begin and save results to this file.

--test-types - images that used in test, same name as directory from if-speedtest.zip. 'one', 'billion', 'dir' for det, det+rec modes by default, if rec mode, then 'rec' directory used.

--det-models - list of detection models used in test, all used by default, if 'rec' mode not used.

--rec-models - list of recognition models used in test, all used by default, if 'det' mode not used.

--port - port container start with

--host - host address, in format http://localhost, http://10.144.2.4

--mode - 'det', 'rec', 'det+rec', type of pipeline to test, 'det' - detection, 'rec' - recognition

-b - size of batch 1, 64