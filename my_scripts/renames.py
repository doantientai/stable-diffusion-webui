"""_summary_
    Given a directory with files named as x.png
    Sort them with x
    Rename them as x.png but in order 
"""

import os
import shutil

if __name__ == "__main__":
    # path_dir_in = "/media/tai/12TB/ForFun/Art/SD/CompVis/stable-diffusion/data/images/init_frames/frames_10k_512x512_transformed_0.7"
    # path_dir_out = "/media/tai/12TB/ForFun/Art/SD/CompVis/stable-diffusion/data/images/init_frames/frames_10k_512x512_transformed_0.7_renamed"
    
    # path_dir_in = "/media/tai/12TB/ForFun/Art/SD/CompVis/stable-diffusion/data/images/init_frames/frames_10k_512x512_transformed"
    # path_dir_out = "/media/tai/12TB/ForFun/Art/SD/CompVis/stable-diffusion/data/images/init_frames/frames_10k_512x512_transformed_renamed_"
    
    path_dir_in = "/media/tai/12TB/ForFun/Art/SD/2023/stable-diffusion-webui/outputs/img2img-images/chez-kawtar/frames_10k_512x512_0-evolution-padded"
    path_dir_out = "/media/tai/12TB/ForFun/Art/SD/2023/stable-diffusion-webui/outputs/img2img-images/chez-kawtar/frames_10k_512x512_0-evolution-padded_renamed"
    
    os.makedirs(path_dir_out, exist_ok=True)

    list_names = os.listdir(path_dir_in)
    list_nums = [int(x.split(".")[0]) for x in list_names]
    
    list_nums, list_names = zip(*sorted(zip(list_nums, list_names)))

    for i in range(len(list_names)):
        file_name_old = list_names[i]
        num = file_name_old.split(".")[0]
        num = int(num)
        file_name_new = f"{i}.png"
        shutil.copyfile(
            os.path.join(path_dir_in, file_name_old),
            os.path.join(path_dir_out, file_name_new),
        )
