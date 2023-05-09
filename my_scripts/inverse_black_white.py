import cv2
import os

path_dir_src = "/media/tai/12TB/ForFun/Art/SD/CompVis/stable-diffusion/data/images/init_frames/frames_10k_512x512_00"
path_dir_out = "/media/tai/12TB/ForFun/Art/SD/CompVis/stable-diffusion/data/images/init_frames/frames_10k_512x512_00_inversed"

os.makedirs(path_dir_out, exist_ok=True)

for filename in os.listdir(path_dir_src):
    print(filename)
    img_gray = cv2.imread(os.path.join(path_dir_src, filename), 0)
    img_inversed = 255 - img_gray
    cv2.imwrite(os.path.join(path_dir_out, filename), img_inversed)
