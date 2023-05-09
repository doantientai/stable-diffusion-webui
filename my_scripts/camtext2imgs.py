# Import libraries
import cv2
import time
import base64
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import numpy as np
import argparse
import time
import os
import json


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


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="The text")
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

    # Create a video capture object
    cap = cv2.VideoCapture(0)
    print("sleeping a bit before starting")
    time.sleep(5)

    # Get the frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
            "text": args.text,
            "step": args.step,
            "strength": args.strength,
            "seed": args.seed,
        }
        with open(args.output[:-4] + ".json", "w") as f:
            json.dump(params, f)

    # Loop until the user presses 'q' key
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        # Check if the frame is valid
        if ret:
            generated = resquest_img2img(
                frame,
                prompt=args.text,
                step=args.step,
                strength=args.strength,
                seed=args.seed,
                server_url=args.server,
            )
            # Increment the frame count
            frame_count += 1

            generated = cv2.resize(generated, (width, height))
            generated = cv2.flip(generated, 1)
            # concat = np.concatenate((generated, frame), axis=1)
            # frame = concat
            frame = generated

            # Calculate the FPS every second
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0

            print(f"FPS: {fps:.2f}")

            # Show the frame in a window
            cv2.imshow("Camera", frame)

            # Save to the video
            if args.output is not None:
                out.write(frame)
                # cv2.imwrite(f"{args.output[:-4]}_{frame_count}.png", frame)

            cap.set(cv2.CAP_PROP_POS_FRAMES, -1)

        # Check if the user presses 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release the video capture object, video writer object and destroy all windows
    cap.release()
    out.release() if args.output is not None else print()
    cv2.destroyAllWindows()
