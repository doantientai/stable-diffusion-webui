# path_dir_images="/home/tai/Desktop/Data on Server (NoMachine)/StableDiffusion/stable-diffusion-webui/data/frames_22k_512x512_red"
path_dir_images="/media/tai/12TB/ForFun/Art/SD/2023/stable-diffusion-webui/outputs/img2img-images/Etienne-simu/frames_1048576_512x512_2_interessant_SD"


ffmpeg -framerate 60 \
       -i "$path_dir_images/%d.png" \
       -c:v libx264 \
       -profile:v high \
       -crf 18 \
       -pix_fmt yuv420p \
       $path_dir_images".mp4"
