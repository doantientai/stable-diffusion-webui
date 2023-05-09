import cv2
import time
import argparse

# Import libraries
import cv2
import time
import base64
import requests
import base64
import numpy as np
import argparse
import time


def resquest_img2img_batch(
    list_img_bgr,
    prompt,
    step=5,
    strength=0.45,
    seed=-1,
    server_url="http://127.0.0.1:7860",
):
    list_img_bgr = list_img_bgr[:10]
    exit("#TODO: batching")
    # Encode
    list_img_encoded = []
    for img in list_img_bgr:
        img_bytes = cv2.imencode(".jpg", img)[1].tobytes()
        img_base64 = base64.b64encode(img_bytes).decode()
        list_img_encoded.append(img_base64)

    payload = {
        "prompt": prompt,
        "steps": step,
        "init_images": list_img_encoded,
        "denoising_strength": strength,
        "seed": seed,
        "batch_size": len(list_img_bgr),
    }
    response = requests.post(url=f"{server_url}/sdapi/v1/img2img", json=payload)
    img_str = response.json()["images"]

    list_outputs = []
    for img_out in img_str:

        # Decode
        img_bytes = base64.b64decode(img_out.split(",", 1)[0])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        res_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        list_outputs.append(res_rgb)

    return list_outputs


if __name__ == "__main__":
    # construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--duration",
        required=True,
        type=int,
        help="duration of the recording in seconds",
    )
    ap.add_argument(
        "-f",
        "--fps",
        required=True,
        type=int,
        help="frames per second of the recording",
    )
    ap.add_argument("-o", "--output", type=str, help="path to the output video file")
    ap.add_argument("--text", type=str, help="The text")
    ap.add_argument("--step", type=int, help="# of iterations")
    ap.add_argument("--strength", type=float, help="The strength of noise (0. -> 1.)")
    ap.add_argument("--seed", type=int, help="Seed, let -1 for random noise")
    ap.add_argument(
        "--server", type=str, default="http://127.0.0.1:7860", help="Server address"
    )
    args = vars(ap.parse_args())

    # set the duration of the video in seconds
    duration = args["duration"]

    # set the frames per second (fps) of the video to be saved
    save_fps = args["fps"]

    # create a VideoCapture object to read from the camera
    cap = cv2.VideoCapture(0)

    # set the width and height of the frame
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # calculate the number of frames to capture
    num_frames = duration * save_fps

    # create an empty list to store the frames
    frames = []

    # loop through the frames and display them in real-time
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # display the frame on the screen
        cv2.imshow("Camera", frame)

        # add the frame to the list
        frames.append(frame)

        # wait for the specified time between frames
        time.sleep(1 / save_fps)

        # print the remaining time
        print(f"Recording in progress: {duration-i//save_fps} seconds left", end="\r")

        # check if the duration has elapsed
        if i == num_frames - 1:
            break

        # check if the user has pressed 'q' to quit
        if cv2.waitKey(1) == ord("q"):
            break

    # release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    ### send images to SD
    list_generated = resquest_img2img_batch(
        frames,
        prompt=args["text"],
        step=args["step"],
        strength=args["strength"],
        seed=args["seed"],
        server_url=args["server"],
    )

    frames = []
    ### display generated images
    for frame in list_generated:
        for i in range(int(save_fps * 24)):
            # display the frame on the screen
            cv2.imshow("Generated", frame)
            if cv2.waitKey(1) == ord("q"):
                break
            # add the frame to the list
        frames.append(frame)

        # wait for the specified time between frames
        # time.sleep(1 / save_fps)

    # if the output file path is specified, save the video
    if args["output"]:
        out = cv2.VideoWriter(
            args["output"], cv2.VideoWriter_fourcc(*"mp4v"), save_fps, (width, height)
        )
        for frame in frames:
            frame = cv2.resize(frame, (width, height))
            out.write(frame)
        out.release()
        print(f"Recording saved as {args['output']}")
    else:
        print("Recording not saved.")

    # print a message when the recording is finished
    print("Recording finished!")
