from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string
import os
import math

# Directory to save generated CAPTCHAs
output_dir = "generated_captchas"
os.makedirs(output_dir, exist_ok=True)

# Hard-coded list of fonts from /usr/share
FONTS = [
    "/usr/share/fonts/opentype/urw-base35/C059-BdIta.otf",
    "/usr/share/fonts/opentype/urw-base35/C059-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/C059-Italic.otf",
    "/usr/share/fonts/opentype/urw-base35/C059-Roman.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-BoldItalic.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Italic.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-BoldItalic.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-BoldItalic.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Italic.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-BoldOblique.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Oblique.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSansNarrow-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/P052-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/P052-BoldItalic.otf",
    "/usr/share/fonts/opentype/urw-base35/P052-Italic.otf",
    "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf",
    "/usr/share/fonts/opentype/urw-base35/StandardSymbolsPS.otf",
    "/usr/share/fonts/opentype/urw-base35/URWBookman-Demi.otf",
    "/usr/share/fonts/opentype/urw-base35/URWBookman-DemiItalic.otf",
    "/usr/share/fonts/opentype/urw-base35/URWBookman-Light.otf",
    "/usr/share/fonts/opentype/urw-base35/URWBookman-LightItalic.otf",
    "/usr/share/fonts/opentype/urw-base35/URWGothic-Book.otf",
    "/usr/share/fonts/opentype/urw-base35/URWGothic-BookOblique.otf",
    "/usr/share/fonts/opentype/urw-base35/URWGothic-Demi.otf",
    "/usr/share/fonts/opentype/urw-base35/URWGothic-DemiOblique.otf",
    "/usr/share/fonts/opentype/urw-base35/Z003-MediumItalic.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-BoldOblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Oblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-BoldOblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-BoldItalic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-BoldItalic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-BoldItalic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Italic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDisplay-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDisplay-BoldItalic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDisplay-Italic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDisplay-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMath-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSymbols-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansSymbols-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-BoldItalic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Italic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifDisplay-BoldItalic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Italic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSerifDisplay-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoTraditionalNushu-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Italic[wdth,wght].ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-Italic[wght].ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono[wght].ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuSans-Italic[wdth,wght].ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuSansMono-Italic[wght].ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuSansMono[wght].ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu[wdth,wght].ttf"
]

def random_text(min_length=4, max_length=8):
    length = random.randint(min_length, max_length)
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

def create_gradient_background(width, height):
    """
    Create a vertical gradient background using two random light colors.
    """
    start_color = (
        random.randint(150, 255),
        random.randint(150, 255),
        random.randint(150, 255)
    )
    end_color = (
        random.randint(150, 255),
        random.randint(150, 255),
        random.randint(150, 255)
    )
    background = Image.new("RGB", (width, height), start_color)
    draw = ImageDraw.Draw(background)
    for y in range(height):
        ratio = y / height
        r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
        g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
        b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return background

def add_text_with_transformations(image, text, block_width, overlap_factor):
    """
    For each character, create a transparent image, draw a drop shadow and the character 
    with a random color, and apply a random rotation. Each letter is centered in its 
    allocated block. The blocks are larger than before (and overlap), ensuring the entire 
    rotated letter is visible.
    """
    width, height = image.size
    n = len(text)
    
    for i, char in enumerate(text):
        font_path = random.choice(FONTS)
        font_size = random.randint(30, 50)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # Create a transparent image for this letter with extra padding (the block)
        char_img = Image.new("RGBA", (block_width, height), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        
        # Get text size using textbbox
        bbox = char_draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the letter in the block (with a small random offset)
        center_x = block_width // 2
        center_y = height // 2
        rand_offset_x = random.randint(-5, 5)
        rand_offset_y = random.randint(-10, 10)
        x_offset = center_x - text_width // 2 + rand_offset_x
        y_offset = center_y - text_height // 2 + rand_offset_y
        
        # Draw drop shadow (offset by 2 pixels)
        shadow_color = (0, 0, 0, 100)
        shadow_offset = (2, 2)
        char_draw.text((x_offset + shadow_offset[0], y_offset + shadow_offset[1]),
                       char, font=font, fill=shadow_color)
        
        # Draw the actual character in a random vibrant color
        char_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)
        char_draw.text((x_offset, y_offset), char, font=font, fill=char_color)
        
        # Apply random rotation with expand=1 so the entire letter is preserved
        angle = random.uniform(-30, 30)
        rotated_char = char_img.rotate(angle, resample=Image.BICUBIC, expand=1)
        
        # Compute the center for the allocated block in the final image.
        # The blocks overlap by the specified factor.
        block_center_x = int(i * block_width * overlap_factor + block_width // 2)
        block_center_y = height // 2
        
        # Compute paste coordinates so that the rotated letter is centered at the block center.
        paste_x = block_center_x - rotated_char.width // 2
        paste_y = block_center_y - rotated_char.height // 2
        
        # Paste the rotated letter onto the main image.
        image.paste(rotated_char, (paste_x, paste_y), rotated_char)
        
    return image

def add_distortion_and_noise(image):
    """
    Add interference elements:
    - Random lines
    - Random dots
    - Random arcs/curves
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Random lines
    for _ in range(random.randint(1, 4)):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        line_width = random.randint(2, 5)
        line_color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        draw.line((x1, y1, x2, y2), fill=line_color, width=line_width)

    # Random dots
    for _ in range(random.randint(5, 15)):
        x, y = random.randint(0, width), random.randint(0, height)
        dot_radius = random.randint(2, 4)
        left_up = (x - dot_radius, y - dot_radius)
        right_down = (x + dot_radius, y + dot_radius)
        dot_color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
        draw.ellipse([left_up, right_down], fill=dot_color)

    # Random arcs/curves
    for _ in range(random.randint(1, 3)):
        start_x = random.randint(0, width // 2)
        start_y = random.randint(0, height // 2)
        end_x = random.randint(width // 2, width)
        end_y = random.randint(height // 2, height)
        bbox = [start_x, start_y, end_x, end_y]
        start_angle = random.randint(0, 180)
        end_angle = start_angle + random.randint(45, 180)
        arc_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        draw.arc(bbox, start=start_angle, end=end_angle, fill=arc_color, width=random.randint(1, 3))

    return image

def generate_captcha(text, desired_width=200, height=75,
                     enlargement_factor=1.5, overlap_factor=0.7):
    """
    Generates a captcha with an adjusted canvas width so that each letter's block is larger.
    - enlargement_factor: how much larger each letter's allocated block is compared to equal division.
    - overlap_factor: fraction of block width used as spacing between consecutive letters.
      (Values less than 1 cause overlapping.)
    """
    n = len(text)
    # Compute block width based on desired width divided equally, then enlarged.
    base_block_width = desired_width / n
    block_width = int(base_block_width * enlargement_factor)
    # Overall width is computed so that the center of each letter is spaced by (block_width * overlap_factor)
    new_width = int(block_width + (n - 1) * block_width * overlap_factor)
    
    # Create a gradient background with the new dimensions.
    captcha_image = create_gradient_background(new_width, height)
    # Add transformed text using the larger block and overlap.
    captcha_image = add_text_with_transformations(captcha_image, text, block_width, overlap_factor)
    # Add interference elements.
    captcha_image = add_distortion_and_noise(captcha_image)
    return captcha_image

def generate_captcha_batch(num_captchas=100):
    for i in range(num_captchas):
        text = random_text()
        captcha_image = generate_captcha(text, desired_width=200, height=75,
                                         enlargement_factor=1.5, overlap_factor=0.7)
        file_path = os.path.join(output_dir, f"{text}.png")
        captcha_image.save(file_path)
        if (i + 1) % 100 == 0 or i == num_captchas - 1:
            print(f"Generated {i + 1}/{num_captchas}")

if __name__ == "__main__":
    generate_captcha_batch(15000)
