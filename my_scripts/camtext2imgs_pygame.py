# Import libraries
import cv2
import time
import base64
import requests
import io
import base64
import numpy as np
import argparse
import time
import os
import json
import pygame
import pygame.camera
import threading
from pynput import keyboard


def resquest_img2img(
    img_bgr, prompt, step=5, strength=0.45, seed=-1, server_url="http://127.0.0.1:7860"
):
    img = img_bgr
    # Encode
    img_bytes = cv2.imencode(".jpg", img)[1].tobytes()
    img_base64 = base64.b64encode(img_bytes).decode()

    payload = {
        "prompt": prompt,
        "steps": step,
        "init_images": [img_base64],
        "denoising_strength": strength,
        "seed": seed,
    }
    response = requests.post(url=f"{server_url}/sdapi/v1/img2img", json=payload)
    img_str = response.json()["images"]

    # Decode
    img_bytes = base64.b64decode(img_str[0].split(",", 1)[0])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    res_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return res_rgb

def on_press(key):
    global text
    try:
        # get the digit value of the pressed key
        digit = int(key.char)
        if digit in range(len(list_texts)):
            text = list_texts[digit]
    except AttributeError:
        pass


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, help="The txt containing a style each line")
    parser.add_argument("--step", type=int, help="# of iterations")
    parser.add_argument(
        "--strength", type=float, help="The strength of noise (0. -> 1.)"
    )
    parser.add_argument("--seed", type=int, help="Seed, let -1 for random noise")
    parser.add_argument("--output", type=str, help="Path to save result to file .mp4")
    parser.add_argument(
        "--server", type=str, default="http://127.0.0.1:7860", help="Server address"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    server_url = args.server

    width = 640
    height = 480

    pygame.init()
    pygame.camera.init()
    cam = pygame.camera.Camera("/dev/video0", (width, height))
    cam.start()

    screen = pygame.display.set_mode((width, height))

    # Initialize variables for FPS calculation
    fps = 0
    start_time = time.time()
    frame_count = 0


    if args.output is not None:
        if args.output[-4:] != ".mp4":
            raise "ERROR: --output must end with .mp4"
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        # Define the codec and create a video writer object
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output, codec, 1, (width, height))

        # Save params to json
        params = {
            "text_file": args.text_file,
            "step": args.step,
            "strength": args.strength,
            "seed": args.seed,
        }
        with open(args.output[:-4] + ".json", "w") as f:
            json.dump(params, f)

    running = True

    list_texts = []
    assert len(list_texts) < 10
    with open(args.text_file) as f:
        for _l in f.readlines():
            list_texts.append(_l[:-1])
        print(list_texts)
    text = list_texts[0]

    # create keyboard listener and start listening
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        cam_image = cam.get_image()
        frame = pygame.surfarray.array3d(cam_image)
        frame = np.swapaxes(frame, 0, 1)

        generated = resquest_img2img(
            frame,
            prompt=text,
            step=args.step,
            strength=args.strength,
            seed=args.seed,
            server_url=args.server,
        )
        # Increment the frame count
        frame_count += 1

        generated = cv2.resize(generated, (width, height))
        generated = cv2.flip(generated, 1)

        # Calculate the FPS every second
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0

        print(f"FPS: {fps:.2f}")

        # Show the frame in a window
        generated_rot = np.swapaxes(generated, 1, 0)
        surface = pygame.surfarray.make_surface(generated_rot)
        screen.blit(surface, (0, 0))
        pygame.display.update()

        # Save to the video
        if args.output is not None:
            out.write(cv2.cvtColor(generated, cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"{args.output[:-4]}_{frame_count}.png", frame)

    # Release the video capture object, video writer object and destroy all windows
    out.release() if args.output is not None else print()


"""
    Command example: 
    python my_scripts/camtext2imgs_pygame.py --text_file /media/tai/12TB/ForFun/Art/SD/2023/stable-diffusion-webui/my_scripts/camtext_styles/list_style.txt --step 10 --strength 0.45 --seed 2023 --output /media/tai/12TB/ForFun/Art/SD/2023/stable-diffusion-webui/outputs/CamText2Vid/multi_styles.mp4
"""