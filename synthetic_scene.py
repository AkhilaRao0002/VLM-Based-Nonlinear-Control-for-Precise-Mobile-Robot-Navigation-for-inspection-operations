from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'

W, H = 640, 480
FX = FY = 250.0
CX, CY = W / 2, H / 2
DEPTH_SCALE = 0.001  # depth PNG stores millimetres

SCENES = {
    'scene_01': {
        'task': 'Navigate to the chair',
        'target': {'label': 'chair', 'robot_xy': [5.0, 4.0], 'size_m': [1.0, 0.9]},
        'obstacles': [
            {'label': 'table', 'robot_rect': [2.0, 3.5, 1.0, 3.0], 'depth_m': 3.0},
        ],
    },
    'scene_02': {
        'task': 'Navigate to the chair',
        'target': {'label': 'chair', 'robot_xy': [4.8, -2.5], 'size_m': [0.9, 0.8]},
        'obstacles': [
            {'label': 'table', 'robot_rect': [2.0, 3.2, -3.5, -1.5], 'depth_m': 2.8},
        ],
    },
    'scene_03': {
        'task': 'Navigate to the chair',
        'target': {'label': 'chair', 'robot_xy': [6.0, 1.8], 'size_m': [1.0, 0.8]},
        'obstacles': [
            {'label': 'table', 'robot_rect': [2.5, 4.0, 0.5, 2.5], 'depth_m': 3.2},
        ],
    },
}


def project_robot_xy(x_forward, y_left, z=None):
    # Camera convention: X right, Z forward. Robot y-left => camera X=-y.
    z = float(z if z is not None else x_forward)
    x_cam = -float(y_left)
    u = FX * x_cam / max(z, 0.1) + CX
    return u


def make_scene(scene_name, spec):
    out = DATA / scene_name
    out.mkdir(parents=True, exist_ok=True)

    rgb = Image.new('RGB', (W, H), (215, 225, 235))
    d = ImageDraw.Draw(rgb)

    # Ceiling/wall/floor for a simple indoor-looking RGB frame.
    d.rectangle([0, 0, W, 230], fill=(205, 215, 225))
    d.polygon([(0, 230), (W, 230), (W, H), (0, H)], fill=(145, 145, 135))
    for y in range(250, H, 45):
        d.line([(0, y), (W, y)], fill=(175, 175, 165), width=1)
    d.line([(0, 230), (W, 230)], fill=(90, 90, 90), width=3)

    depth_mm = np.full((H, W), 10000, dtype=np.uint16)  # background 10 m

    # Draw table/obstacle. We intentionally use clear visual labels so the
    # lightweight local VLM has a simple synthetic scene to interpret.
    gt_obstacles = []
    for obs in spec['obstacles']:
        x1, x2, y1, y2 = obs['robot_rect']
        z = obs['depth_m']
        u1 = project_robot_xy(x1, y2, z)
        u2 = project_robot_xy(x1, y1, z)
        # Perspective-ish vertical placement based on depth.
        top = int(np.clip(210 - 20 * z, 80, 220))
        bottom = int(np.clip(400 - 8 * z, 280, 430))
        left, right = sorted([int(u1), int(u2)])
        left = max(20, left); right = min(W - 20, right)
        d.rectangle([left, top, right, bottom], fill=(120, 80, 45), outline=(50, 30, 15), width=4)
        d.rectangle([left, top, right, top + 18], fill=(150, 100, 55))
        d.text((left + 8, top + 25), obs['label'].upper(), fill=(255, 255, 255))
        
        # Encode a front/back depth gradient vertically so RGB-D geometry can
        # recover the obstacle's forward extent from the bbox samples.
        for yy in range(top, bottom + 1):
            alpha = (yy - top) / max(bottom - top, 1)
            depth_at_row = x1 + alpha * (x2 - x1)
            depth_mm[yy, left:right+1] = int(max(0.5, depth_at_row) * 1000)
        gt_obstacles.append({
            'label': obs['label'],
            'bbox': [left, top, right, bottom],
            'robot_rect': [x1, x2, y1, y2],
        })

    # Draw target chair.
    tx, ty = spec['target']['robot_xy']
    tz = tx
    u = int(project_robot_xy(tx, ty, tz))
    top = int(np.clip(235 - 18 * tz, 100, 235))
    bottom = min(H - 30, top + 125)
    width = 70
    left, right = max(10, u - width // 2), min(W - 10, u + width // 2)
    # chair back + seat + legs
    d.rectangle([left, top, right, top + 60], fill=(50, 110, 190), outline=(20, 50, 100), width=3)
    d.rectangle([left + 8, top + 60, right - 8, top + 85], fill=(65, 125, 205), outline=(20, 50, 100), width=3)
    d.line([(left + 12, top + 85), (left + 5, bottom)], fill=(20, 50, 100), width=6)
    d.line([(right - 12, top + 85), (right - 5, bottom)], fill=(20, 50, 100), width=6)
    d.text((left, max(10, top - 22)), 'CHAIR', fill=(20, 20, 20))
    depth_mm[max(0, top):min(H, bottom+1), left:right+1] = int(tz * 1000)
    target_bbox = [left, top, right, bottom]

    # Add a robot marker in the lower center of the camera image.
    d.ellipse([CX - 12, H - 65, CX + 12, H - 41], fill=(30, 30, 30))
    d.text((CX + 18, H - 65), 'ROBOT', fill=(30, 30, 30))

    rgb_path = out / 'rgb.png'
    depth_path = out / 'depth.png'
    meta_path = out / 'ground_truth.json'
    rgb.save(rgb_path)
    Image.fromarray(depth_mm).save(depth_path)

    metadata = {
        'camera': {'width': W, 'height': H, 'fx': FX, 'fy': FY, 'cx': CX, 'cy': CY, 'depth_scale': DEPTH_SCALE},
        'task': spec['task'],
        'target': {'label': spec['target']['label'], 'bbox': target_bbox, 'robot_xy': spec['target']['robot_xy']},
        'obstacles': gt_obstacles,
        'note': 'Synthetic RGB-D scene for pipeline development. Ground truth is used only by synthetic_perception mode; real_vlm mode must infer detections from RGB.'
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return rgb_path, depth_path, meta_path


def main():
    DATA.mkdir(exist_ok=True)
    for name, spec in SCENES.items():
        make_scene(name, spec)
    print(f'Generated {len(SCENES)} synthetic RGB-D scenes in {DATA}')


if __name__ == '__main__':
    main()
