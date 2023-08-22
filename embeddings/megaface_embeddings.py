import argparse
import logging
from functools import partial
from pathlib import Path
import copy
import sys

from tqdm import tqdm

sys.path.append('.')
from demo_client import IFRClient, to_chunks, file2base64, to_bool
from save_utils import save_features, check_if_result_exist

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--port', default=18081, type=int, help='Port')
    parser.add_argument('-u', '--uri', default='http://localhost', type=str, help='Server hostname or ip with protocol')
    parser.add_argument('-b', '--batch', default=64, type=int, help='Batch size')
    parser.add_argument('--embed', default='True', type=str, help='Extract embeddings, otherwise run detection only')
    parser.add_argument('--embed_only', default='False', type=str,
                        help='Omit detection step. Expects already cropped 112x112 images')
    parser.add_argument('--megaface-root', required=True, type=str, help='Path to megaface dataset')
    parser.add_argument('--dtype', required=True, type=str, choices=['float32', 'float64'], help='data type to save embeddings')

    args = parser.parse_args()


    client = IFRClient(host=args.uri, port=args.port)

    print('---')
    info = client.server_info(show=True)
    rec_model = info['models']['rec_name']
    print('Configs:')
    print(f"    Embed detected faces:        {args.embed}")
    print(f"    Run in embed only mode:      {args.embed_only}")
    print(f'    Request batch size:          {args.batch}')
    print('---')

    allowed_ext = '.jpeg .jpg .bmp .png .webp .tiff'.split()
    files = (Path(args.megaface_root) / "data").rglob("*.*")
    # if result exists not overwrite
    files = [str(file) for file in files if file.suffix.lower() in allowed_ext and not check_if_result_exist(file, args.megaface_root, rec_model)]
    filepaths = copy.copy(files)
    print('Images will be sent in base64 encoding')
    mode = 'data'
    files = [file2base64(file) for file in files]

    print(f"Total files detected: {len(files)}")
    im_batches = zip(files, filepaths)
    im_batches = to_chunks(im_batches, args.batch)
    im_batches = [list(chunk) for chunk in im_batches]

    _part_extract_vecs = partial(client.extract, extract_embedding=to_bool(args.embed),
                                 embed_only=to_bool(args.embed_only), mode=mode, return_face_data=True)

    kwargs = dict(root=args.megaface_root, rec_model=rec_model, dtype=args.dtype)
    for batch in tqdm(range(0, len(im_batches))):
        imgs = [im[0] for im in im_batches[batch]]
        filepaths = [im[1] for im in im_batches[batch]]
        r = _part_extract_vecs(imgs)
        if to_bool(args.embed_only):
            save_features(r['data'], filepaths, **kwargs)
        else:
            save_features([resp['faces'][0] for resp in r['data']], filepaths, **kwargs)
