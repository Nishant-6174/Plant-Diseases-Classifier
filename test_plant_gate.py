from PIL import Image

from src.utils.plant_gate import is_plant_image


image_path = r"C:\Users\My\OneDrive\Pictures\Screenshots\Screenshot 2026-04-18 182824.png"

image = Image.open(image_path)

is_plant, confidence = is_plant_image(image)

print(f"Is plant: {is_plant}")
print(f"Plant confidence: {confidence:.4f}")