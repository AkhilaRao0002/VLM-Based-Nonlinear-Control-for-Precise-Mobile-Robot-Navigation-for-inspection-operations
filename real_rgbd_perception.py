from pathlib import Path

import cv2
import numpy as np

from tartanair.reader import TartanAirImageReader


class RealRGBDPerception:

    def __init__(self, dataset_root):
        self.root = Path(dataset_root)

        self.rgb_dir = self.root / "image_lcam_front"
        self.depth_dir = self.root / "depth_lcam_front"

        if not self.rgb_dir.exists():
            raise FileNotFoundError(
                f"RGB directory not found: {self.rgb_dir}"
            )

        if not self.depth_dir.exists():
            raise FileNotFoundError(
                f"Depth directory not found: {self.depth_dir}"
            )

        self.tartan_reader = TartanAirImageReader()

    # --------------------------------------------------------
    # GET MATCHING RGB + DEPTH FRAME
    # --------------------------------------------------------

    def get_frame(self, frame_id="000010"):

        rgb_path = (
            self.rgb_dir
            / f"{frame_id}_lcam_front.png"
        )

        depth_path = (
            self.depth_dir
            / f"{frame_id}_lcam_front_depth.png"
        )

        if not rgb_path.exists():
            raise FileNotFoundError(
                f"RGB frame not found: {rgb_path}"
            )

        if not depth_path.exists():
            raise FileNotFoundError(
                f"Depth frame not found: {depth_path}"
            )

        # RGB
        rgb = cv2.imread(
            str(rgb_path),
            cv2.IMREAD_COLOR
        )

        if rgb is None:
            raise RuntimeError(
                f"Could not read RGB image: {rgb_path}"
            )

        rgb = cv2.cvtColor(
            rgb,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # TartanAir depth decoding
        # ----------------------------------------------------

        depth = self.tartan_reader.read_depth(
            str(depth_path)
        )

        if depth is None:
            raise RuntimeError(
                f"Could not read depth image: {depth_path}"
            )

        return {
            "frame_id": frame_id,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "rgb": rgb,
            "depth": depth
        }

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    def inspect_frame(self, frame_id="000010"):

        frame = self.get_frame(frame_id)

        print("\n" + "=" * 60)
        print("REAL RGB-D FRAME")
        print("=" * 60)

        print("Frame:", frame["frame_id"])

        print(
            "RGB shape:",
            frame["rgb"].shape
        )

        print(
            "RGB dtype:",
            frame["rgb"].dtype
        )

        print(
            "RGB path:",
            frame["rgb_path"]
        )

        print(
            "Depth shape:",
            frame["depth"].shape
        )

        print(
            "Depth dtype:",
            frame["depth"].dtype
        )

        print(
            "Depth min:",
            float(np.min(frame["depth"]))
        )

        print(
            "Depth max:",
            float(np.max(frame["depth"]))
        )

        print(
            "Depth mean:",
            float(np.mean(frame["depth"]))
        )

        print("=" * 60)


if __name__ == "__main__":

    dataset_root = (
        r"C:\Users\Akhil\OneDrive\Desktop"
        r"\company_project\robotics_iisc"
        r"\tartanair_data"
        r"\AbandonedFactory"
        r"\Data_omni"
        r"\P0000"
    )

    perception = RealRGBDPerception(
        dataset_root
    )

    perception.inspect_frame("000010")