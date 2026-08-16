from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
scene=ROOT/'data'/'scene_01'
rgb=Image.open(scene/'rgb.png')
depth=np.array(Image.open(scene/'depth.png')).astype(float)*0.001

plt.figure(figsize=(8,6)); plt.imshow(rgb); plt.axis('off'); plt.title('Synthetic RGB Scene'); plt.tight_layout(); plt.savefig(ROOT/'results'/'rgb_scene_preview.png',dpi=200); plt.show()
plt.figure(figsize=(8,6)); plt.imshow(depth); plt.colorbar(label='Depth (m)'); plt.axis('off'); plt.title('Synthetic Depth Image'); plt.tight_layout(); plt.savefig(ROOT/'results'/'depth_scene_preview.png',dpi=200); plt.show()
