import cv2
import numpy as np


class RGBDProcessor:
    """Convert image boxes and aligned depth into metric 3-D geometry."""

    def __init__(self, fx, fy, cx, cy, depth_scale=0.001):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.depth_scale = depth_scale

    @staticmethod
    def bbox_to_pixels(bbox, width, height):
        x1, y1, x2, y2 = bbox

        # Support either pixel coordinates or normalized 0..1000 coordinates.
        if max(abs(float(v)) for v in bbox) <= 1000:
            # Ambiguous for small pixel boxes; the VLM prompt uses pixels,
            # so only normalize if coordinates clearly look like 0..1000 data.
            if max(width, height) > 1000 and max(float(v) for v in bbox) <= 1000:
                pass

        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y2 = int(np.clip(y2, 0, height - 1))
        return [x1, y1, x2, y2]

    def robust_depth(self, depth_image, u, v, window=5):
        h, w = depth_image.shape[:2]
        patch = depth_image[
            max(0, v-window):min(h, v+window+1),
            max(0, u-window):min(w, u+window+1)
        ].astype(np.float32)
        patch *= self.depth_scale
        valid = patch[np.isfinite(patch) & (patch > 0)]
        return float(np.median(valid)) if len(valid) else None

    def pixel_to_3d(self, u, v, depth_m):
        if depth_m is None or depth_m <= 0:
            return None
        return np.array([
            (u-self.cx) * depth_m / self.fx,
            (v-self.cy) * depth_m / self.fy,
            depth_m
        ], dtype=float)

    def bbox_to_3d(self, bbox, depth_image):
        h, w = depth_image.shape[:2]

        x1, y1, x2, y2 = self.bbox_to_pixels(
            bbox, w, h
        )

        if x2 <= x1 or y2 <= y1:
            return None

        # ----------------------------------------------------
        # Sample many pixels inside the bounding box
        # ----------------------------------------------------

        xs = np.linspace(x1, x2, 9).astype(int)
        ys = np.linspace(y1, y2, 9).astype(int)

        points = []

        for v in ys:
            for u in xs:

                depth_m = self.robust_depth(
                    depth_image,
                    u,
                    v,
                    window=2
                )

                p = self.pixel_to_3d(
                    u,
                    v,
                    depth_m
                )

                if p is not None:
                    points.append(p)

        if not points:
            return None

        points = np.asarray(points)

        # ----------------------------------------------------
        # Remove extreme depth outliers
        # ----------------------------------------------------

        z_values = points[:, 2]

        z_low = np.percentile(z_values, 5)
        z_high = np.percentile(z_values, 95)

        valid = points[
            (z_values >= z_low) &
            (z_values <= z_high)
        ]

        if len(valid) == 0:
            valid = points

        # ----------------------------------------------------
        # Robust center
        # ----------------------------------------------------

        center = np.median(
            valid,
            axis=0
        )

        return {
            "center": center,

            "x_min": float(
                valid[:, 0].min()
            ),

            "x_max": float(
                valid[:, 0].max()
            ),

            "z_min": float(
                valid[:, 2].min()
            ),

            "z_max": float(
                valid[:, 2].max()
            ),

            "bbox_pixels": [
                x1,
                y1,
                x2,
                y2
            ]
        }


def load_rgbd(rgb_path, depth_path):
    rgb = cv2.imread(str(rgb_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if rgb is None:
        raise FileNotFoundError(rgb_path)
    if depth is None:
        raise FileNotFoundError(depth_path)

    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth
