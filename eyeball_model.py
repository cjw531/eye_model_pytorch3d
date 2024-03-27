import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np

device = torch.cuda.current_device()

# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
dtype = torch.float32
print(device)

from Tianfu_MultiviewDeflectometryModel import GenerateSphereSmoothVerts, GenerateSphereSmoothFaces
from pytorch3d.io import save_obj
from tqdm import tqdm


def folder_check(save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

def generate_eyeball_model(save_folder: str, file_iter: str, radius_offset=0, start_index=2, end_index=30):
    latitudes = 100
    longitudes = 100

    verts = torch.zeros(((latitudes + 1) * (longitudes + 1), 3), dtype= dtype,device = device)

    latitudeAngles = torch.linspace(np.pi/2, -np.pi/2, latitudes + 1, dtype= dtype,device = device)
    z = 1.05 * torch.sin(latitudeAngles)[latitudes//2:]
    xy = torch.sqrt(torch.max(( (0.7 + radius_offset) ** 2 - (z + 0.35 - radius_offset) ** 2), (1 - z * z) )) # OR 0.75 and 0.3
    xy[-1] = 0
    xy[-end_index: -start_index + 1] = torch.linspace(xy[-end_index], xy[-start_index], end_index- start_index + 1)
    
    verts = GenerateSphereSmoothVerts(xy = xy, z = z,  latitudes = latitudes, longitudes = longitudes)
    faces = GenerateSphereSmoothFaces(latitudes = latitudes, longitudes = longitudes)
    
    file_path = os.path.join(save_folder, (str(file_iter) + '.obj'))
    save_obj(file_path, verts, faces)


if __name__ =='__main__':
    # base path set & create folders
    save_folder = './dl_data_set_jc/meshes/'
    folder_check(save_folder)

    file_iter = 0
    total_iterations = sum([(31 - i) for i in range(3, 31)])
    with tqdm(total=total_iterations) as pbar:
        for i in range(3, 31):
            for j in range(i, 31):
                generate_eyeball_model(save_folder, str(file_iter), start_index=i, end_index=j)
                file_iter += 1
                pbar.update(1)
