from pathlib import Path

from PIL import Image

from src.utils.clip_plant_gate import is_plant_image


# ---------------------------------------------------------
# REAL PLANT IMAGES
# ---------------------------------------------------------

PLANT_IMAGES = [
    Path(r"static\samples\tomato_early_blight.jpg"),
    Path(r"static\samples\corn_common_rust.jpg"),
    Path(r"static\samples\potato_late_blight.jpg"),
    Path(r"static\samples\healthy_apple.jpg"),
]


# ---------------------------------------------------------
# NON-PLANT IMAGES
# ---------------------------------------------------------

NEGATIVE_DIR = Path(r"tests\clip_negatives")


# ---------------------------------------------------------
# CHECK FUNCTION
# ---------------------------------------------------------

def check_image(path: Path, category: str):

    try:
        image = Image.open(path).convert("RGB")

        is_plant, confidence = is_plant_image(image)

        print(
            f"{category:10} | "
            f"{path.name:30} | "
            f"Plant={str(is_plant):5} | "
            f"Confidence={confidence:.4f}"
        )

    except Exception as exc:

        print(
            f"{category:10} | "
            f"{path.name:30} | "
            f"ERROR: {exc}"
        )


# ---------------------------------------------------------
# RUN CALIBRATION
# ---------------------------------------------------------

print()
print("=" * 85)
print("CLIP PLANT GATE - CALIBRATION TEST")
print("=" * 85)

print()
print("REAL PLANT IMAGES")
print("-" * 85)

for path in PLANT_IMAGES:

    if path.exists():
        check_image(path, "PLANT")
    else:
        print(f"PLANT      | {path} | FILE NOT FOUND")


print()
print("NON-PLANT IMAGES")
print("-" * 85)

if NEGATIVE_DIR.exists():

    negative_images = sorted(
        [
            p
            for p in NEGATIVE_DIR.iterdir()
            if p.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )

    if not negative_images:
        print("No negative images found.")

    for path in negative_images:
        check_image(path, "NEGATIVE")

else:

    print(f"Folder not found: {NEGATIVE_DIR}")


print()
print("=" * 85)
print("TEST COMPLETE")
print("=" * 85)