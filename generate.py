from PIL import Image, ImageDraw, ImageFont
import os

from english_words import get_english_words_set
from random import randint
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


words = get_english_words_set(['web2'], lower=True)

BASE_FONT_PATH = os.getenv('FONT_PATH')

font_mappings = {
    "San Serif": f"{BASE_FONT_PATH}/Arial.ttf",
    "Serif": f"{BASE_FONT_PATH}/Times New Roman.ttf",
    "Hand-Painted": f"{BASE_FONT_PATH}/SignPainter.ttc",
    "cursive": f"{BASE_FONT_PATH}/Zapfino.ttf"
}

area_mappings = {
    "North": (100, 0),
    "NorthWest": (50, 0),
    "NorthEast": (200, 0),
    "South": (150, 300),
    "Center": (170, 170)
}

font_keys = list(font_mappings.keys())
font_key_count = len(font_keys)

area_keys = list(area_mappings.keys())
area_key_count = len(area_keys)

def genImage(area: str, font: str, text: str):
    if area not in area_mappings:
        raise Exception("Please use a correct area mapping key")
    
    if font not in font_mappings:
        raise Exception("Please use a correct font mapping key")
    
    font_file = font_mappings[font]
    text_pos = area_mappings[area]

    # Create a 400x400 image
    img = Image.new('L', (400, 400), color=0)

    # Draw text on the image
    d = ImageDraw.Draw(img)

    # Load a font
    font_loaded = ImageFont.truetype(font_file, 30)

    # Draw text with 3d perspective
    d.text(text_pos, text, fill=255, font=font_loaded)

    if area == "North":
        pos = "on the top"
    elif area == "South":
        pos = "at the bottom"
    elif area == "Center":
        pos = "in the middle"
    elif area == "NorthWest":
        pos = "in the top right"
    elif area == "NorthEast":
        pos = "in the top left"
    else:
        pos = ""

    

    return img, f"the {font} text '{text}' {pos}"




if __name__ == "__main__":
    
    df = pd.DataFrame(columns=['caption', 'image-file'])
    
    # Make images directory.
    if not os.path.exists("./imgs/"):
      os.mkdir("imgs")

    for i, word in enumerate(words):
      if (len(word) > 10):
          continue

      if (i == 20):
          break
      
      # Select a cardinality at random.
      area = area_keys[randint(0, area_key_count - 1)]

      # Select style at random.
      font = font_keys[randint(0, font_key_count - 1)]

      img, caption = genImage(area, font, word)

      filename = f'{word}-{area}-{font}.png'

      new_row = pd.DataFrame({ 'caption': [caption], 'image-file': [filename] })

      df = pd.concat([df, new_row], ignore_index=True)

      img.save(f'./imgs/{filename}')
    
    df.to_csv('labels.csv', index=False)