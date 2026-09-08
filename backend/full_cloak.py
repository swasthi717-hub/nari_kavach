import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFilter
from torchvision import transforms


# ============================================================
# SETTINGS
# ============================================================

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
# HELPERS
# ============================================================

def total_variation(x):
    """
    Measures high-frequency variation in the perturbation.
    Higher TV = more noisy / visible perturbation.
    """
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))

    return tv_h + tv_w


def create_soft_mask(width, height):
    """
    Creates a soft elliptical mask so the perturbation
    fades naturally toward the edges of the face.
    """

    mask = Image.new("L", (width, height), 0)

    # Large central ellipse
    ellipse = Image.new("L", (width, height), 0)

    # Leave a small margin around the face crop
    left = int(width * 0.08)
    top = int(height * 0.06)
    right = int(width * 0.92)
    bottom = int(height * 0.96)

    from PIL import ImageDraw

    draw = ImageDraw.Draw(ellipse)

    draw.ellipse(
        (left, top, right, bottom),
        fill=255
    )

    # Blur the mask so the transition is gradual
    blur_radius = max(5, int(min(width, height) * 0.06))

    ellipse = ellipse.filter(
        ImageFilter.GaussianBlur(blur_radius)
    )

    return ellipse


# ============================================================
# MAIN CLOAK FUNCTION
# ============================================================

def cloak_face(full_img):
    """
    Takes a full PIL image and returns the full image
    with a subtle adversarial face perturbation.

    Returns None if no face is detected.
    """

    # --------------------------------------------------------
    # 1. Detect face
    # --------------------------------------------------------

    boxes, _ = mtcnn.detect(full_img)

    if boxes is None:
        return None

    # Use first detected face
    box = boxes[0]

    x1, y1, x2, y2 = [
        int(b) for b in box
    ]

    # Keep coordinates inside image
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

    # FaceNet expects 160 x 160
    face_resized = face_crop.resize(
        (160, 160),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # 3. Convert to tensor
    # --------------------------------------------------------

    to_tensor = transforms.ToTensor()

    face_tensor = to_tensor(face_resized)

    # Normalize to [-1, 1]
    face_tensor = (face_tensor - 0.5) / 0.5

    face_tensor = face_tensor.unsqueeze(0)

    # --------------------------------------------------------
    # 4. Get original face embedding
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
    # 5. Initialize VERY small perturbation
    # --------------------------------------------------------

    perturbation = (
        torch.randn_like(face_tensor) * 0.002
    )

    perturbation.requires_grad_(True)

    # --------------------------------------------------------
    # 6. Conservative attack parameters
    # --------------------------------------------------------

    epsilon = 0.012

    steps = 60

    alpha = 0.0005

    # Penalties controlling visual quality
    pixel_weight = 0.015
    smooth_weight = 0.025

    # --------------------------------------------------------
    # 7. Iterative optimization
    # --------------------------------------------------------

    for step in range(steps):

        # Create adversarial face
        adv_face = face_tensor + perturbation

        adv_face = torch.clamp(
            adv_face,
            -1.0,
            1.0
        )

        # Get current embedding
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

        # We want this value to become smaller.
        identity_loss = cosine_similarity

        # ----------------------------------------------------
        # Keep pixel changes small
        # ----------------------------------------------------

        pixel_loss = torch.mean(
            perturbation ** 2
        )

        # ----------------------------------------------------
        # Discourage noisy / high-frequency artifacts
        # ----------------------------------------------------

        smooth_loss = total_variation(
            perturbation
        )

        # ----------------------------------------------------
        # Combined objective
        # ----------------------------------------------------

        loss = (
            identity_loss
            + pixel_weight * pixel_loss
            + smooth_weight * smooth_loss
        )

        # Clear old gradients
        model.zero_grad()

        if perturbation.grad is not None:
            perturbation.grad.zero_()

        # Backprop
        loss.backward()

        # ----------------------------------------------------
        # Gradient update
        # ----------------------------------------------------

        with torch.no_grad():

            perturbation -= (
                alpha * perturbation.grad.sign()
            )

            # Keep perturbation inside epsilon budget
            perturbation.clamp_(
                -epsilon,
                epsilon
            )

        perturbation = perturbation.detach()
        perturbation.requires_grad_(True)

    # --------------------------------------------------------
    # 8. Create final adversarial face
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
    # 9. Convert back to PIL
    # --------------------------------------------------------

    final_face_tensor = (
        final_face_tensor
        .squeeze(0)
        .detach()
    )

    cloaked_face_img = transforms.ToPILImage()(
        (final_face_tensor * 0.5 + 0.5).clamp(0, 1)
    )

    # --------------------------------------------------------
    # 10. Resize to original face dimensions
    # --------------------------------------------------------

    cloaked_face_resized = cloaked_face_img.resize(
        original_size,
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # 11. Softly blend perturbation into original face
    # --------------------------------------------------------

    mask = create_soft_mask(
        original_size[0],
        original_size[1]
    )

    # Slightly reduce overall strength of the modification
    # to preserve visual quality.
    mask = mask.point(
        lambda p: int(p * 0.72)
    )

    final_full_img = full_img.copy()

    final_full_img.paste(
        cloaked_face_resized,
        (x1, y1),
        mask
    )

    return final_full_img


# ============================================================
# TEST
# ============================================================

if __name__ == '__main__':

    input_path = (
        r'E:\Swasthi\books\face-cloak-project\test_photo.png'
    )

    output_path = 'cloaked_full_photo.jpg'

    full_img = Image.open(
        input_path
    ).convert('RGB')

    result = cloak_face(full_img)

    if result is None:

        print("No face detected.")

    else:

        result.save(
            output_path,
            format='JPEG',
            quality=98,
            subsampling=0
        )

        print(
            f"Saved {output_path}"
        )
