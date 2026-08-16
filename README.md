# VLM + RGB-D + 3D + NMPC Inspection Architecture

This version is based on the user's working four-file pipeline and changes the
navigation objective from point-to-point target reaching to inspection-pose
navigation.

## Architecture

Inspection instruction
-> VLM scene/inspection analysis
-> RGB-D metric geometry
-> target and obstacle 3-D positions
-> desired inspection pose [xg, yg, theta_g]
-> differential-drive NMPC
-> v, omega
-> robot state feedback

## First test: synthetic perception

Activate the existing virtual environment and run:

python main_pipeline.py --perception synthetic

This tests RGB-D, 3-D geometry, inspection-pose generation and NMPC without
requiring a camera or VLM.

## Second test: local VLM

After the synthetic test succeeds:

python main_pipeline.py --perception vlm

The first VLM run may download the model.

## Outputs

Results are saved in results/:

- scene_result.json
- metrics.csv
- trajectory.csv
- controls.csv
- inspection_trajectory.png
- linear_velocity.png
- angular_velocity.png
- clearance.png

The synthetic run is an integration/validation result. It should not be
reported as VLM detection accuracy.
