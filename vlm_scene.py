import json
import re
import torch
from pathlib import Path
from PIL import Image


class VLMSceneAnalyzer:

    def __init__(
        self,
        model_name="HuggingFaceTB/SmolVLM-256M-Instruct"
    ):

        print(f"[VLM] Loading model: {model_name}")

        from transformers import (
            AutoProcessor,
            AutoModelForMultimodalLM
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"[VLM] Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

        print("[VLM] Model loaded successfully.")

    # --------------------------------------------------------
    # VLM NAVIGATION + INSPECTION
    # --------------------------------------------------------

    def analyze(self, image_path, instruction):

        image_path = Path(image_path)

        # Load RGB image. Keep the original image for the VLM.
        image = Image.open(image_path).convert("RGB")

        prompt = f"""
You are a robot vision system.

Task:
{instruction}

Analyze the provided image carefully.

You must identify:
1. The object that should be inspected.
2. Objects that can obstruct the robot's direct path to that object.
3. Whether the direct path is blocked.
4. Whether inspection is required.
5. The best viewing direction.
6. A suitable inspection distance.

IMPORTANT RULES:

- Look at the image before answering.
- Identify real visible objects only.
- The chair is the inspection target if a chair is visible.
- A table is an obstacle if it lies between the robot and the chair.
- Do not treat the robot as an obstacle.
- Do not invent objects.
- Do not repeat objects.
- Do not explain your reasoning.
- Do not write a paragraph.
- Return ONLY the six lines below.
- Use exactly YES or NO for BLOCKED and INSPECTION.
- If there are no obstacles, write NONE.
- Use a distance such as 1.0.
- VIEW must be one of: front, left, right, back.

Required format:

TARGET: chair
OBSTACLES: table
BLOCKED: YES
INSPECTION: YES
DISTANCE: 1.0
VIEW: front
"""

        # ----------------------------------------------------
        # IMPORTANT FOR TRANSFORMERS 5.x + IDEFICS3:
        #
        # First render the chat template as TEXT, then pass the
        # actual PIL image to the processor separately.
        #
        # This avoids the duplicate-images error encountered
        # when images are passed through apply_chat_template().
        # ----------------------------------------------------

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.processor(
            text=chat_text,
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        print("[VLM] Performing inspection scene analysis...")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                temperature=0.0
            )

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = generated_ids[:, input_length:]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        print("\n" + "=" * 60)
        print("VLM INSPECTION ANALYSIS")
        print("=" * 60)
        print(response)
        print("=" * 60)

        scene = self._parse_compact_response(response)

        if scene is None:
            scene = self._parse_response(response)

        # ----------------------------------------------------
        # Synthetic-data geometry adapter
        # ----------------------------------------------------
        #
        # The VLM is intentionally responsible for SEMANTIC
        # understanding. Precise 3-D geometry comes from RGB-D.
        #
        # For the current camera-free synthetic experiment,
        # ground_truth.json supplies known pixel regions so the
        # downstream RGB-D -> 3D -> NMPC stages can be evaluated
        # even when the small 256M VLM does not produce pixel
        # coordinates.
        #
        # This is explicitly marked as a synthetic geometry
        # adapter and is NOT treated as VLM-generated geometry.
        # ----------------------------------------------------

        if scene is not None:
            scene = self._attach_synthetic_geometry(
                scene,
                image_path
            )
            return scene

        return self._fallback_scene(response, image_path)

    
    def _parse_compact_response(self, response):

        lines = [
            line.strip()
            for line in response.splitlines()
            if line.strip()
        ]

        result = {
            "target": None,
            "obstacles": [],
            "blocked": False,
            "inspection": True,
            "distance": 1.0,
            "view": "front"
        }

        found_fields = set()

        for line in lines:

            upper = line.upper()

            if upper.startswith("TARGET:"):

                value = line.split(":", 1)[1].strip()

                if value:
                    result["target"] = value.lower()
                    found_fields.add("target")

            elif upper.startswith("OBSTACLES:"):

                value = line.split(":", 1)[1].strip()

                if value.upper() != "NONE":

                    obstacles = value.split(",")

                    result["obstacles"] = list(
                        dict.fromkeys(
                            x.strip().lower()
                            for x in obstacles
                            if x.strip()
                        )
                    )

                found_fields.add("obstacles")

            elif upper.startswith("BLOCKED:"):

                value = line.split(":", 1)[1].strip().upper()

                if value in {"YES", "NO"}:
                    result["blocked"] = value == "YES"
                    found_fields.add("blocked")

            elif upper.startswith("INSPECTION:"):

                value = line.split(":", 1)[1].strip().upper()

                if value in {"YES", "NO"}:
                    result["inspection"] = value == "YES"
                    found_fields.add("inspection")

            elif upper.startswith("DISTANCE:"):

                value = line.split(":", 1)[1].strip()

                try:
                    result["distance"] = float(value)
                    found_fields.add("distance")
                except ValueError:
                    pass

            elif upper.startswith("VIEW:"):

                value = line.split(":", 1)[1].strip().lower()

                if value in {"front", "left", "right", "back"}:
                    result["view"] = value
                    found_fields.add("view")

        # Require the important semantic fields.
        required_fields = {
            "target",
            "obstacles",
            "blocked"
        }

        if not required_fields.issubset(found_fields):
            print(
                "[VLM] Compact response incomplete; "
                "using fallback parser."
            )
            return None

        if result["target"] is None:
            return None

        return {
            "navigation_inspection": {
                "scene_navigable": True,
                "direct_path_blocked": result["blocked"],
                "inspection_required": result["inspection"],
                "preferred_distance_m": result["distance"],
                "preferred_view": result["view"],
                "notes": (
                    "Semantic scene understanding generated "
                    "by local VLM."
                ),
                "perception_source": "VLM_semantic"
            },

            "target": {
                "label": result["target"]
            },

            "obstacles": [
                {
                    "label": label,
                    "blocks_direct_path": result["blocked"]
                }
                for label in result["obstacles"]
                if label not in {
                    "robot",
                    "mobile robot"
                }
            ]
        }
    # --------------------------------------------------------
    # JSON PARSER
    # --------------------------------------------------------

    def _parse_response(self, response):

        response = response.strip()

        response = re.sub(
            r"```json",
            "",
            response,
            flags=re.IGNORECASE
        )

        response = re.sub(
            r"```",
            "",
            response
        )

        response = response.strip()

        # Prefer the largest JSON object in the response.
        match = re.search(
            r"\{.*\}",
            response,
            flags=re.DOTALL
        )

        if not match:
            return None

        try:
            data = json.loads(match.group(0))

            print("\nParsed VLM semantic information:")
            print(
                json.dumps(
                    data,
                    indent=4
                )
            )

            return self._normalize_scene(data)

        except json.JSONDecodeError as e:
            print(f"[VLM] JSON parsing error: {e}")
            return None

    # --------------------------------------------------------
    # NORMALIZE VLM SEMANTIC OUTPUT
    # --------------------------------------------------------

    def _normalize_scene(self, data):

        nav = data.get(
            "navigation_inspection",
            {}
        )

        target_label = nav.get(
            "target_label",
            data.get("target_label", None)
        )

        obstacle_labels = nav.get(
            "obstacle_labels",
            data.get("obstacle_labels", [])
        )

        if isinstance(obstacle_labels, str):
            obstacle_labels = [obstacle_labels]

        if target_label is not None:
            target_label = str(target_label).strip().lower()

        obstacle_labels = [
            str(x).strip().lower()
            for x in obstacle_labels
            if str(x).strip()
        ]

        # Remove the robot from obstacle semantics.
        obstacle_labels = [
            x for x in obstacle_labels
            if x not in {"robot", "mobile robot"}
        ]

        return {
            "navigation_inspection": {
                "scene_navigable": bool(
                    nav.get("scene_navigable", True)
                ),
                "direct_path_blocked": bool(
                    nav.get("direct_path_blocked", False)
                ),
                "inspection_required": bool(
                    nav.get("inspection_required", True)
                ),
                "preferred_distance_m": float(
                    nav.get("preferred_distance_m", 1.0)
                ),
                "preferred_view": str(
                    nav.get("preferred_view", "front")
                ),
                "notes": str(
                    nav.get("notes", "")
                ),
                "perception_source": "VLM_semantic"
            },
            "target": {
                "label": target_label
            } if target_label else None,
            "obstacles": [
                {
                    "label": label,
                    "blocks_direct_path": False
                }
                for label in obstacle_labels
            ]
        }

    # --------------------------------------------------------
    # SYNTHETIC GEOMETRY ADAPTER
    # --------------------------------------------------------

    def _attach_synthetic_geometry(
        self,
        scene,
        image_path
    ):

        meta_path = (
            Path(image_path).parent /
            "ground_truth.json"
        )

        if not meta_path.exists():
            print(
                "[VLM] No ground_truth.json found. "
                "Semantic VLM result will be returned without "
                "pixel geometry."
            )
            return scene

        try:
            ground_truth = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as e:
            print(
                f"[VLM] Could not read synthetic geometry: {e}"
            )
            return scene

        gt_target = ground_truth.get("target")
        gt_obstacles = ground_truth.get(
            "obstacles",
            []
        )

        target_label = (
            scene.get("target") or {}
        ).get("label")

        # Match the VLM semantic target to the synthetic
        # geometric annotation.
        if target_label and gt_target:
            gt_label = str(
                gt_target.get("label", "")
            ).lower()

            if target_label.lower() == gt_label:
                scene["target"]["bbox"] = gt_target["bbox"]
                scene["target"][
                    "geometry_source"
                ] = "synthetic_ground_truth"

        # Match semantic obstacle labels to synthetic geometry.
        for obstacle in scene.get("obstacles", []):

            label = str(
                obstacle.get("label", "")
            ).lower()

            for gt_obstacle in gt_obstacles:

                gt_label = str(
                    gt_obstacle.get("label", "")
                ).lower()

                if label == gt_label:

                    obstacle["bbox"] = (
                        gt_obstacle["bbox"]
                    )

                    obstacle[
                        "blocks_direct_path"
                    ] = bool(
                        gt_obstacle.get(
                            "blocks_direct_path",
                            obstacle.get(
                                "blocks_direct_path",
                                False
                            )
                        )
                    )

                    obstacle[
                        "geometry_source"
                    ] = "synthetic_ground_truth"

                    break

        scene[
            "geometry_note"
        ] = (
            "VLM provides semantic target/obstacle "
            "identification. Pixel geometry in this "
            "camera-free synthetic experiment is supplied "
            "by ground_truth.json and then converted to "
            "3-D using RGB-D depth."
        )

        return scene

    # --------------------------------------------------------
    # FALLBACK FOR NON-JSON VLM OUTPUT
    # --------------------------------------------------------

    def _fallback_scene(
        self,
        response,
        image_path
    ):

        text = response.lower()

        target_label = None

        # Use the inspection instruction/image response to
        # recover the common synthetic target if the small VLM
        # describes it in natural language rather than JSON.
        if "chair" in text:
            target_label = "chair"
        elif "table" in text and "inspect" in text:
            target_label = "table"

        obstacle_labels = []

        if (
            "table" in text
            and target_label != "table"
        ):
            obstacle_labels.append("table")

        scene = {
            "navigation_inspection": {
                "scene_navigable": True,
                "direct_path_blocked": bool(
                    obstacle_labels
                ),
                "inspection_required": True,
                "preferred_distance_m": 1.0,
                "preferred_view": "front",
                "notes": (
                    "VLM returned natural-language output; "
                    "semantic labels were recovered from "
                    "the response."
                ),
                "perception_source": "VLM_semantic_fallback"
            },
            "target": (
                {"label": target_label}
                if target_label
                else None
            ),
            "obstacles": [
                {
                    "label": label,
                    "blocks_direct_path": True
                }
                for label in obstacle_labels
            ]
        }

        return self._attach_synthetic_geometry(
            scene,
            image_path
        )


# ============================================================
# SYNTHETIC PERCEPTION
# ============================================================

def synthetic_perception(meta_path=None):

    if meta_path is None:
        raise ValueError(
            "synthetic_perception requires ground_truth.json"
        )

    meta_path = Path(meta_path)

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    ground_truth = json.loads(
        meta_path.read_text(
            encoding='utf-8'
        )
    )

    obstacles = []

    for obstacle in ground_truth.get(
        'obstacles',
        []
    ):

        obstacles.append({
            'label': obstacle.get(
                'label',
                'obstacle'
            ),

            'bbox': obstacle.get(
                'bbox'
            ),

            'robot_rect': obstacle.get(
                'robot_rect'
            ),

            'blocks_direct_path': True
        })

    return {

        'navigation_inspection': {

            'scene_navigable': True,

            'direct_path_blocked': bool(
                obstacles
            ),

            'notes':
                'Synthetic scene annotations '
                'loaded from ground_truth.json.',

            'perception_source':
                'synthetic_ground_truth'
        },

        'target': ground_truth.get(
            'target'
        ),

        'obstacles': obstacles
    }