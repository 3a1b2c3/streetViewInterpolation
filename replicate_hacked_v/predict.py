import os
from cog import BasePredictor, Input, Path
import sys
sys.path.append('C:\workspace\cs231n\proj\DynamiCrafter')
os.chdir('C:\workspace\cs231n\proj\DynamiCrafter')

from PIL import Image
import numpy as np
import torch
from scripts.gradio.i2v_test_application import Image2Video

class Predictor(BasePredictor):
    def setup(self) -> None:
        directory = r"C:\workspace\cs231n\proj\DynamiCrafter\output"
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.image2video = Image2Video(r'C:\workspace\cs231n\proj\DynamiCrafter\output', resolution='320_512')

        
    def predict(
        self,
        image1_path: Path = Input(description="Input Image 1"),
        image2_path: Path = Input(description="Input Image 2"),
        prompt: str = Input(default='a smiling girl'),
        steps: int = Input(default=50),
        cfg_scale: float = Input(default=7.5),
        eta: float = Input(default=1.0),
        fs: int = Input(default=5),
        seed: int = Input(default=12306),
    ) -> Path:
        image1 = Image.open(image1_path)
        if image1.mode == 'RGBA':
            image1 = image1.convert('RGB')
        image2 = Image.open(image2_path)
        if image2.mode == 'RGBA':
            image2 = image2.convert('RGB')
        image1_np = np.array(image1)
        image2_np = np.array(image2)
        i2v_output_video = self.image2video.get_image(image=image1_np, prompt=prompt, steps=steps, cfg_scale=cfg_scale, eta=eta, fs=fs, seed=seed, image2=image2_np)
        print(i2v_output_video)
        return i2v_output_video

p=Predictor()
p.setup()
image1_path=r"C:\workspace\cs231n\proj\data\mapillary\iAl09jBrws1vHnR2NOyzzg\780648522639677.jpg"
image2_path=r"C:\workspace\cs231n\proj\data\mapillary\zONqVJujBgyQrelwP7lmAg\2829395483945030.jpg"
res = p.predict( image1_path=image1_path, image2_path=image2_path, prompt="fly through")
# C:\workspace\cs231n\proj\DynamiCrafter\output\fly_through.mp4