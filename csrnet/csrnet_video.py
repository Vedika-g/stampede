import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from csrnet_model import CSRNet


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "../crowd.mp4"
OUTPUT_PATH = "../csrnet_heatmap.mp4"

# Run CSRNet once every 3 frames
FRAME_SKIP = 3

# Smaller input = faster CSRNet inference
INPUT_WIDTH = 640
INPUT_HEIGHT = 360


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("CSRNet - RTX 3050 OPTIMIZED")
print("========================================")

print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD CSRNET WEIGHTS
# ============================================================

print("\nLoading CSRNet weights...")

weights_path = hf_hub_download(
    repo_id="AbdurRahman011/csrnet-indian-metro-crowd-density",
    filename="csrnet_v3_best.pth"
)

model = CSRNet()

checkpoint = torch.load(
    weights_path,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)

model.eval()

print("CSRNet loaded successfully!")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# OPEN VIDEO
# ============================================================

video = cv2.VideoCapture(
    VIDEO_PATH
)

if not video.isOpened():

    print(
        "ERROR: Could not open crowd.mp4"
    )

    exit()


# ============================================================
# VIDEO INFORMATION
# ============================================================

width = int(
    video.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    video.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = video.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("\nVideo information:")
print(
    f"Resolution: {width} x {height}"
)
print(
    f"FPS: {fps:.2f}"
)
print(
    f"Total frames: {total_frames}"
)

print(
    f"\nCSRNet runs every {FRAME_SKIP} frames."
)

print(
    f"CSRNet input: "
    f"{INPUT_WIDTH} x {INPUT_HEIGHT}"
)


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

output = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# VARIABLES
# ============================================================

frame_number = 0

last_heatmap = None

last_count = 0.0

processed_csrnet_frames = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1


    # ========================================================
    # RUN CSRNET ONLY EVERY 3RD FRAME
    # ========================================================

    if frame_number % FRAME_SKIP == 0:

        processed_csrnet_frames += 1

        # ----------------------------------------------------
        # Resize frame for CSRNet
        # ----------------------------------------------------

        small_frame = cv2.resize(
            frame,
            (
                INPUT_WIDTH,
                INPUT_HEIGHT
            )
        )


        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Convert to PIL
        # ----------------------------------------------------

        pil_image = Image.fromarray(
            rgb_frame
        )


        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        input_tensor = transform(
            pil_image
        )

        input_tensor = input_tensor.unsqueeze(
            0
        )

        input_tensor = input_tensor.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # CSRNet inference
        # ----------------------------------------------------

        with torch.inference_mode():

            density_map = model(
                input_tensor
            )


        # ----------------------------------------------------
        # Convert density map
        # ----------------------------------------------------

        density = (
            density_map
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Estimated count
        # ----------------------------------------------------

        last_count = float(
            density.sum()
        )


        # ----------------------------------------------------
        # Normalize density
        # ----------------------------------------------------

        if density.max() > density.min():

            density_normalized = cv2.normalize(
                density,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

        else:

            density_normalized = np.zeros_like(
                density
            )


        density_normalized = (
            density_normalized
            .astype(np.uint8)
        )


        # ----------------------------------------------------
        # Apply heatmap
        # ----------------------------------------------------

        heatmap = cv2.applyColorMap(
            density_normalized,
            cv2.COLORMAP_JET
        )


        # ----------------------------------------------------
        # Resize heatmap to original video size
        # ----------------------------------------------------

        last_heatmap = cv2.resize(
            heatmap,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_LINEAR
        )


    # ========================================================
    # USE PREVIOUS HEATMAP ON SKIPPED FRAMES
    # ========================================================

    if last_heatmap is not None:

        combined = cv2.addWeighted(
            frame,
            0.60,
            last_heatmap,
            0.40,
            0
        )

    else:

        combined = frame.copy()


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        combined,
        (10, 10),
        (520, 125),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        combined,
        f"CSRNet Count: {last_count:.1f}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        combined,
        f"Frame: {frame_number}/{total_frames}",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        combined,
        f"CSRNet Frame: {processed_csrnet_frames}",
        (25, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    output.write(
        combined
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "CSRNet - Crowd Density",
        combined
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print(
            "\nStopped by user."
        )

        break


# ============================================================
# CLEANUP
# ============================================================

video.release()

output.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n========================================")
print("CSRNet PROCESSING COMPLETED")
print("========================================")

print(
    f"Total video frames: {frame_number}"
)

print(
    f"CSRNet frames processed: "
    f"{processed_csrnet_frames}"
)

print(
    f"Frame skip: {FRAME_SKIP}"
)

print(
    f"Output saved to: {OUTPUT_PATH}"
)