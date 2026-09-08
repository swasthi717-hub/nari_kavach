import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFilter
from torchvision import transforms


# ============================================================
# SETTINGS
# ============================================================

# Makes the random initialization reproducible
torch.manual_seed(42)

# Face detector
mtcnn = MTCNN(
    image_size=160,
    margin=0
)

# Face recognition model
model = InceptionResnetV1(
    pretrained='vggface2'
).eval()


# ============================================================
# QUALITY ENHANCEMENT
# ============================================================

def enhance_cloaked_face(face_img):
    """
    Very mild post-processing to reduce visible
    high-frequency artifacts while retaining detail.

    This is intentionally subtle because aggressive
    enhancement can weaken the cloaking effect.
    """

    # Very light smoothing
    smoothed = face_img.filter(
        ImageFilter.GaussianBlur(radius=0.30)
    )

    # Recover a small amount of perceived sharpness
    enhanced = smoothed.filter(
        ImageFilter.UnsharpMask(
            radius=0.75,
            percent=30,
            threshold=4
        )
    )

    return enhanced


# ============================================================
# SOFT FACE MASK
# ============================================================

def create_soft_mask(width, height):
    """
    Creates a soft elliptical mask so the cloaked face
    blends naturally into the surrounding image.
    """

    from PIL import ImageDraw

    mask = Image.new(
        "L",
        (width, height),
        0
    )

    draw = ImageDraw.Draw(mask)

    left = int(width * 0.07)
    top = int(height * 0.05)
    right = int(width * 0.93)
    bottom = int(height * 0.97)

    draw.ellipse(
        (left, top, right, bottom),
        fill=255
    )

    # Soft transition at edges
    blur_radius = max(
        4,
        int(min(width, height) * 0.05)
    )

    mask = mask.filter(
        ImageFilter.GaussianBlur(
            blur_radius
        )
    )

    # Slightly reduce blending strength
    mask = mask.point(
        lambda p: int(p * 0.72)
    )

    return mask


# ============================================================
# FACE CLOAK
# ============================================================

def cloak_face(full_img):
    """
    Takes a full PIL image and returns the full image
    with the detected face cloaked.

    Returns None if no face is detected.
    """

    # --------------------------------------------------------
    # 1. Detect face
    # --------------------------------------------------------

    boxes, _ = mtcnn.detect(full_img)

    if boxes is None:
        return None

    # Use the first detected face
    box = boxes[0]

    x1, y1, x2, y2 = [
        int(b) for b in box
    ]

    # Make sure coordinates stay inside image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(full_img.width, x2)
    y2 = min(full_img.height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    # --------------------------------------------------------
    # 2. Crop face
    # --------------------------------------------------------

    face_crop = full_img.crop(
        (x1, y1, x2, y2)
    )

    original_size = face_crop.size

    # FaceNet input size
    face_resized = face_crop.resize(
        (160, 160),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # 3. Convert face to tensor
    # --------------------------------------------------------

    to_tensor = transforms.ToTensor()

    face_tensor = to_tensor(
        face_resized
    )

    # Normalize [0,1] → [-1,1]
    face_tensor = (
        face_tensor - 0.5
    ) / 0.5

    face_tensor = face_tensor.unsqueeze(0)

    # --------------------------------------------------------
    # 4. Original face embedding
    # --------------------------------------------------------

    with torch.no_grad():

        original_embedding = model(
            face_tensor
        )

        original_embedding = F.normalize(
            original_embedding,
            p=2,
            dim=1
        )

    # --------------------------------------------------------
    # 5. Start with very small random perturbation
    # --------------------------------------------------------

    perturbation = (
        torch.randn_like(face_tensor)
        * 0.002
    )

    perturbation.requires_grad_(True)

    # --------------------------------------------------------
    # 6. Conservative cloaking parameters
    # --------------------------------------------------------

    # Maximum pixel perturbation
    epsilon = 0.012

    # Number of optimization iterations
    steps = 60

    # Size of each update
    alpha = 0.0005

    # Quality-preserving penalties
    pixel_weight = 0.015
    smooth_weight = 0.025

    # --------------------------------------------------------
    # 7. Optimize the face embedding
    # --------------------------------------------------------

    for step in range(steps):

        # Create current adversarial face
        adv_face = (
            face_tensor + perturbation
        )

        # Keep image values valid
        adv_face = torch.clamp(
            adv_face,
            -1.0,
            1.0
        )

        # Current embedding
        current_embedding = model(
            adv_face
        )

        current_embedding = F.normalize(
            current_embedding,
            p=2,
            dim=1
        )

        # ----------------------------------------------------
        # Identity similarity
        # ----------------------------------------------------

        cosine_similarity = F.cosine_similarity(
            current_embedding,
            original_embedding,
            dim=1
        ).mean()

        # We want similarity to decrease
        identity_loss = cosine_similarity

        # ----------------------------------------------------
        # Penalize large pixel changes
        # ----------------------------------------------------

        pixel_loss = torch.mean(
            perturbation ** 2
        )

        # ----------------------------------------------------
        # Penalize noisy high-frequency changes
        # ----------------------------------------------------

        tv_vertical = torch.mean(
            torch.abs(
                perturbation[:, :, 1:, :]
                -
                perturbation[:, :, :-1, :]
            )
        )

        tv_horizontal = torch.mean(
            torch.abs(
                perturbation[:, :, :, 1:]
                -
                perturbation[:, :, :, :-1]
            )
        )

        smooth_loss = (
            tv_vertical
            + tv_horizontal
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        loss = (
            identity_loss
            + pixel_weight * pixel_loss
            + smooth_weight * smooth_loss
        )

        # Clear model gradients
        model.zero_grad()

        if perturbation.grad is not None:
            perturbation.grad.zero_()

        # Calculate gradients
        loss.backward()

        # ----------------------------------------------------
        # Update perturbation
        # ----------------------------------------------------

        with torch.no_grad():

            # Move in direction that reduces similarity
            perturbation -= (
                alpha
                * perturbation.grad.sign()
            )

            # Enforce perturbation budget
            perturbation.clamp_(
                -epsilon,
                epsilon
            )

        # Detach before next iteration
        perturbation = (
            perturbation.detach()
        )

        perturbation.requires_grad_(True)

    # --------------------------------------------------------
    # 8. Construct final cloaked face
    # --------------------------------------------------------

    with torch.no_grad():

        final_face_tensor = (
            face_tensor + perturbation
        )

        final_face_tensor = torch.clamp(
            final_face_tensor,
            -1.0,
            1.0
        )

    # --------------------------------------------------------
    # 9. Convert tensor back to PIL
    # --------------------------------------------------------

    final_face_tensor = (
        final_face_tensor
        .squeeze(0)
        .detach()
    )

    cloaked_face_img = transforms.ToPILImage()(
        (
            final_face_tensor * 0.5
            + 0.5
        ).clamp(0, 1)
    )

    # --------------------------------------------------------
    # 10. Resize to original face dimensions
    # --------------------------------------------------------

    cloaked_face_resized = (
        cloaked_face_img.resize(
            original_size,
            Image.Resampling.LANCZOS
        )
    )

    # --------------------------------------------------------
    # 11. Mild quality enhancement
    # --------------------------------------------------------

    cloaked_face_resized = (
        enhance_cloaked_face(
            cloaked_face_resized
        )
    )

    # --------------------------------------------------------
    # 12. Create soft blending mask
    # --------------------------------------------------------

    mask = create_soft_mask(
        original_size[0],
        original_size[1]
    )

    # --------------------------------------------------------
    # 13. Paste cloaked face into original image
    # --------------------------------------------------------

    final_full_img = full_img.copy()

    final_full_img.paste(
        cloaked_face_resized,
        (x1, y1),
        mask
    )

    return final_full_img


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == '__main__':

    input_path = (
        r'E:\Swasthi\books\face-cloak-project\test_photo.png'
    )

    output_path = (
        'cloaked_full_photo.jpg'
    )

    # Load original
    full_img = Image.open(
        input_path
    ).convert('RGB')

    print("Detecting face and generating cloak...")

    # Cloak
    result = cloak_face(
        full_img
    )

    # Check result
    if result is None:

        print(
            "No face detected."
        )

    else:

        # Save at high JPEG quality
        result.save(
            output_path,
            format='JPEG',
            quality=98,
            subsampling=0
        )

        print(
            f"Saved {output_path}"
        )
