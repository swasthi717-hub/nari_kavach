import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torchvision import transforms

torch.manual_seed(42)

mtcnn = MTCNN(image_size=160, margin=0)
model = InceptionResnetV1(pretrained='vggface2').eval()


def cloak_face(full_img):
    """Return the full image with an imperceptibly modified face."""

    boxes, _ = mtcnn.detect(full_img)

    if boxes is None:
        return None

    box = boxes[0]
    x1, y1, x2, y2 = [int(b) for b in box]

    # Keep coordinates valid
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(full_img.width, x2)
    y2 = min(full_img.height, y2)

    face_crop = full_img.crop((x1, y1, x2, y2))
    original_size = face_crop.size

    # FaceNet input
    face_resized = face_crop.resize((160, 160), Image.LANCZOS)

    to_tensor = transforms.ToTensor()

    face_tensor = to_tensor(face_resized)
    face_tensor = (face_tensor - 0.5) / 0.5
    face_tensor = face_tensor.unsqueeze(0)

    with torch.no_grad():
        original_embedding = model(face_tensor)
        original_embedding = F.normalize(original_embedding, p=2, dim=1)

    # Start with very small noise
    adv_face = face_tensor.clone()

    noise = torch.randn_like(adv_face) * 0.003
    adv_face = adv_face + noise

    # Smaller perturbation budget
    epsilon = 0.012

    # Small iterative updates
    steps = 50
    alpha = 0.0008

    for step in range(steps):

        adv_face.requires_grad_(True)

        embedding = model(adv_face)
        embedding = F.normalize(embedding, p=2, dim=1)

        # Maximize angular distance from original identity embedding
        cosine_similarity = F.cosine_similarity(
            embedding,
            original_embedding,
            dim=1
        )

        loss = cosine_similarity.mean()

        model.zero_grad()

        if adv_face.grad is not None:
            adv_face.grad.zero_()

        loss.backward()

        with torch.no_grad():

            # Gradient direction that reduces similarity
            gradient = adv_face.grad.sign()

            adv_face = adv_face - alpha * gradient

            # Keep perturbation very small
            perturbation = torch.clamp(
                adv_face - face_tensor,
                -epsilon,
                epsilon
            )

            adv_face = face_tensor + perturbation

            # Keep normalized image values valid
            adv_face = torch.clamp(
                adv_face,
                -1.0,
                1.0
            )

        adv_face = adv_face.detach()

    # Convert back to image
    final_face_tensor = adv_face.squeeze(0)

    cloaked_face_img = transforms.ToPILImage()(
        (final_face_tensor * 0.5 + 0.5).clamp(0, 1)
    )

    # Restore original face dimensions
    cloaked_face_resized = cloaked_face_img.resize(
        original_size,
        Image.LANCZOS
    )

    # Paste back into original image
    final_full_img = full_img.copy()

    final_full_img.paste(
        cloaked_face_resized,
        (x1, y1)
    )

    return final_full_img


if __name__ == '__main__':

    full_img = Image.open(
        r'E:\Swasthi\books\face-cloak-project\test_photo.png'
    ).convert('RGB')

    result = cloak_face(full_img)

    if result is not None:
        result.save(
            'cloaked_full_photo.jpg',
            quality=98,
            subsampling=0
        )

        print("Saved cloaked_full_photo.jpg")
    else:
        print("No face detected.")
