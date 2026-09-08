import easyocr
import re
from PIL import Image, ImageFilter


# =========================================================
# Load OCR reader once
# =========================================================

reader = easyocr.Reader(['en'])


# =========================================================
# Check whether OCR text contains sensitive information
# =========================================================

def check_sensitive(text):
    """
    Checks OCR text against sensitive-information patterns.

    Returns category name if matched, otherwise None.
    """

    cleaned = re.sub(r'[\s\-]', '', text).upper()

    # -----------------------------------------------------
    # Phone number
    # Examples:
    # 9876543210
    # +919876543210
    # -----------------------------------------------------

    if re.fullmatch(r'(\+91)?\d{10}', cleaned):
        return "Phone Number"


    # -----------------------------------------------------
    # Aadhaar-style number
    # Example:
    # 123456789012
    # -----------------------------------------------------

    if re.fullmatch(r'\d{12}', cleaned):
        return "ID Number (Aadhaar-style)"


    # -----------------------------------------------------
    # PAN
    # Example:
    # ABCDE1234F
    # -----------------------------------------------------

    if re.search(r'[A-Z]{5}\d{4}[A-Z]', cleaned):
        return "PAN Card Number"


    # -----------------------------------------------------
    # Indian vehicle number plate
    #
    # Normal examples:
    # HP12S0990
    # DL01AB1234
    # MH12AB1234
    #
    # OCR can sometimes produce:
    # PHP1250990
    # instead of:
    # HP12S0990
    # -----------------------------------------------------

    vehicle_pattern = r'[A-Z]{2}\d{1,2}[A-Z0-9]{1,3}\d{4}'

    if re.search(vehicle_pattern, cleaned):
        return "Vehicle Number Plate"


    # -----------------------------------------------------
    # OCR-tolerant vehicle number plate
    #
    # Handles common OCR mistakes where an extra letter
    # gets added at the beginning of the plate.
    #
    # Example:
    # PHP1250990
    #              ↓
    # HP12S0990
    # -----------------------------------------------------

    if len(cleaned) >= 9:

        for start in range(0, min(3, len(cleaned) - 8)):

            candidate = cleaned[start:]

            if re.fullmatch(
                r'[A-Z]{2}\d{1,2}[A-Z0-9]{1,3}\d{4}',
                candidate
            ):
                return "Vehicle Number Plate"


    # -----------------------------------------------------
    # Email
    # -----------------------------------------------------

    if re.search(
        r'[A-Za-z0-9._%+-]+[@o][A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        text
    ):
        return "Email Address"


    # -----------------------------------------------------
    # PIN code
    # Example:
    # 110001
    # -----------------------------------------------------

    if re.fullmatch(r'\d{6}', cleaned):
        return "PIN Code"


    # -----------------------------------------------------
    # Generic ID
    #
    # Any sequence containing 6 or more digits.
    # -----------------------------------------------------

    if re.search(r'\d{6,}', cleaned):
        return "Generic ID Number"


    return None


# =========================================================
# Scan image for sensitive information
# =========================================================

def scan_sensitive(image_path):
    """
    Scan image for sensitive information.

    Does NOT modify the image.

    Returns a list containing:

    - detected OCR text
    - category
    - confidence
    - bounding box
    """

    results = reader.readtext(image_path)

    flagged = []

    for bbox, text, confidence in results:

        # Ignore low-confidence OCR results
        if confidence < 0.4:
            continue

        category = check_sensitive(text)

        if category:

            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]

            left = int(min(xs))
            right = int(max(xs))
            top = int(min(ys))
            bottom = int(max(ys))

            flagged.append({
                "text": text,
                "category": category,
                "confidence": round(float(confidence), 2),
                "bbox": [
                    left,
                    top,
                    right,
                    bottom
                ]
            })

    return flagged


# =========================================================
# Apply blur to selected detections
# =========================================================

def apply_blur(image_path, detections, output_path):
    """
    Blur only the regions selected by the user.

    The detections list comes from the Privacy Mirror
    review/consent step.
    """

    img = Image.open(image_path).convert("RGB")

    for item in detections:

        left, top, right, bottom = item["bbox"]

        # -------------------------------------------------
        # Safety bounds
        # -------------------------------------------------

        left = max(0, left)
        top = max(0, top)

        right = min(img.width, right)
        bottom = min(img.height, bottom)

        if right <= left or bottom <= top:
            continue

        # -------------------------------------------------
        # Crop detected sensitive region
        # -------------------------------------------------

        region = img.crop(
            (
                left,
                top,
                right,
                bottom
            )
        )

        # -------------------------------------------------
        # Apply Gaussian blur
        # -------------------------------------------------

        blurred_region = region.filter(
            ImageFilter.GaussianBlur(
                radius=15
            )
        )

        # -------------------------------------------------
        # Put blurred region back into image
        # -------------------------------------------------

        img.paste(
            blurred_region,
            (left, top)
        )

    # -----------------------------------------------------
    # Save final image
    # -----------------------------------------------------

    img.save(
        output_path,
        quality=95
    )

    return output_path