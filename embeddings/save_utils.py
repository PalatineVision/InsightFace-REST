import struct
import numpy as np
from pathlib import Path
from typing import List

cv_type_to_dtype = {
5 : np.dtype('float32'),
6 : np.dtype('float64')}
dtype_to_cv_type = {
'float32': 5,
'float64': 6}
dtype_num_bytes = {
'float32': 4,
'float64': 8}

def write_bin(path, feature: List, dtype='float32'):
  # default save float32
  num_bytes = dtype_num_bytes[dtype]
  cv_type = dtype_to_cv_type[dtype]
  with open(path, 'wb') as f:
    # pack len of embedding (rows), cols, stride, type
    f.write(struct.pack('4i', len(feature),1, num_bytes, cv_type))
    if dtype == 'float32':
        f.write(struct.pack("%df"%len(feature), *feature))
    elif dtype == 'float64':
        f.write(struct.pack("%dd"%len(feature), *feature))


def read_mat(f):
    """
    Reads an OpenCV mat from the given file opened in binary mode
    """
    rows, cols, stride, type_ = struct.unpack('4i', f.read(4*4)) # read 4 ints
    mat = np.frombuffer(f.read(rows*stride),dtype=cv_type_to_dtype[type_])
    return mat.reshape(rows,cols)


def compute_megaface_result_path(file, root, rec_model):
    filepath = Path(file) 
    if 'megaface_images' in filepath.parts:
        # file ./megaface/data/megaface_images/100/10000339@N08/4158579397_2.jpg
        # root ./megaface
        # result_file ./megaface/results/ifc_feature_out/megaface/100/10000339@N08/4158579397_0.jpg_{rec_model}.bin
        outpath = Path(root) / "results" / "ifc_features_out" / "megaface" / filepath.parent.parent.name / filepath.parent.name / (filepath.name + f'_{rec_model}.bin')
    elif 'facescrub_images' in filepath.parts:
        # file ./megaface/data/facescrub_images/Adam_Brody/Adam_Brody_241.png
        # root ./megaface
        # ./megaface/results/ifc_feature_out/facescrub/Adam_Brody/Adam_Brody_241.png_{rec_model}.bin
        outpath = Path(root) / "results" / "ifc_features_out" / "facescrub" / filepath.parent.name / (filepath.name + f'_{rec_model}.bin')
    else:
        raise ValueError(f"Wrong filepath {filepath}!") 
    outpath.parent.mkdir(parents=True, exist_ok=True)
    return str(outpath)


def save_features(resp_faces, files, root, rec_model, dtype):
    for file, face in zip(files, resp_faces):
        feature = face['vec']
        write_bin(compute_megaface_result_path(file, root, rec_model), feature, dtype)
