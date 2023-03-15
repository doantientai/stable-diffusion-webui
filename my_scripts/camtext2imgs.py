# Import libraries
import cv2
import time
import base64
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import numpy as np

# Create a video capture object
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
fps = 0
start_time = time.time()
frame_count = 0


def resquest_img2img(img_bgr, prompt):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Encode
    img_bytes = cv2.imencode(".jpg", img)[1].tobytes()
    img_base64 = base64.b64encode(img_bytes).decode()

    payload = {
        "prompt": prompt,
        "steps": 10,
        "init_images": [img_base64],
    }
    response = requests.post(url=f"{server_url}/sdapi/v1/img2img", json=payload)
    img_str = response.json()["images"]
    # image = Image.open(io.BytesIO(base64.b64decode(r[0].split(",", 1)[0])))
    # image.save("output.png")

    # Decode
    img_bytes = base64.b64decode(img_str[0].split(",", 1)[0])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    res_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("res_bgr.png", res_bgr)
    # exit()
    return res_rgb


if __name__ == "__main__":
    server_url = "https://d8b2f215-34ff-492e.gradio.live"

    # Loop until the user presses 'q' key
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is valid
        if ret:
            frame = resquest_img2img(frame, "a beautiful blue sky")

            # Increment the frame count
            frame_count += 1

            # Calculate the FPS every second
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0

            # Put the FPS text on the frame
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show the frame in a window
            cv2.imshow("Camera", frame)

        # Check if the user presses 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
