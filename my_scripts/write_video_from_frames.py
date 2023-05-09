
import cv2
import os

image_folder = "/home/tai/Desktop/Data on Server (NoMachine)/StableDiffusion/stable-diffusion-webui/data/frames_22k_512x512_red/output_002_str0.7"
video_name = "video.mp4"
fps = 30

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key=lambda x: int(x.split(".")[0]))

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))


cv2.destroyAllWindows()
video.release()
