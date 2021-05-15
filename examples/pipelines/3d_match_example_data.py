from glob import glob
from pprint import pprint
import cv2
import numpy as np
from PIL import Image

def cloud_from_depth(depth, K, depth_scale = 1000):
    # z = d / depth_scale
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy

    h, w = depth.shape[:2]

    fx, cx = K[0, 0], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]

    u, v = np.meshgrid(
        np.linspace(0, w-1, w, dtype=np.int16),
        np.linspace(0, h-1, h, dtype=np.int16))

    u, v = u.flatten(), v.flatten()

    z = depth.flatten() / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    mask = np.nonzero(z)[0]

    return np.stack([x, y, z], axis=1)[mask, :]


intrinsic_matrix = np.loadtxt("/home/slam_data/3dmatch-toolbox/data/sample/depth-fusion-demo/camera-intrinsics.txt")
print(intrinsic_matrix)

fs = glob("/home/slam_data/3dmatch-toolbox/data/sample/depth-fusion-demo/rgbd-frames/*")
fsrgb = sorted([f for f in fs if ".color." in f])
fsdepth = sorted([f for f in fs if ".depth." in f])
fspose = sorted([f for f in fs if ".pose." in f])

assert len(fsrgb) == len(fsdepth) == len(fspose) > 0, (len(fsrgb) ,len(fsdepth) , len(fspose))

for frgb, fdepth, fpose in zip(fsrgb, fsdepth, fspose):
    rgb = cv2.imread(frgb)
    depth = cv2.cvtColor(cv2.imread(fdepth), cv2.COLOR_RGB2GRAY)
    cloud = cloud_from_depth(depth, intrinsic_matrix)
    print(cloud.shape)
    # Image.fromarray(rgb).show()
    exit()
