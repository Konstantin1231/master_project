
from PIL import Image
def crop_image(path, left_p = 25,top_per = 40, right_per = 25, bottom_per = 40):
    # Open the image file
    image = Image.open(path)

    # Get the original width and height of the image
    original_width, original_height = image.size

    # Define the cropping percentages
    left_percentage = left_p  # Left edge percentage
    top_percentage = top_per   # Top edge percentage
    right_percentage = right_per  # Right edge percentage
    bottom_percentage = bottom_per # Bottom edge percentage

    # Calculate cropping box coordinates
    left = (left_percentage / 100) * original_width
    top = (top_percentage / 100) * original_height
    right = (right_percentage / 100) * original_width
    bottom = (bottom_percentage / 100) * original_height

    # Crop the image
    cropped_image = image.crop((int(left), int(top), int(right), int(bottom)))

    # Save the cropped image
    cropped_image.save(path)

    # Close the original image
    image.close()
path = f"images/render_toy_1.jpg"
crop_image(path, 30, 45, 30, 45)