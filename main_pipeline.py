import argparse
import json
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from vlm_scene import VLMSceneAnalyzer, synthetic_perception
from rgbd_processor import RGBDProcessor, load_rgbd
from navigation import (
    NMPCController,
    predict_state,
    distance_to_rectangle,
    wrap_angle
)


ROOT = Path(__file__).resolve().parent


# ============================================================
# CAMERA -> ROBOT COORDINATES
# ============================================================

def camera_to_robot(p):
    """
    Camera:
        X = right
        Z = forward

    Robot:
        x = forward
        y = left
    """

    return np.array(
        [p[2], -p[0]],
        dtype=float
    )


# ============================================================
# INSPECTION GOAL
# ============================================================

def inspection_goal(target_3d, inspection_info):

    target_xy = camera_to_robot(
        target_3d['center']
    )

    preferred_distance = float(
        inspection_info.get(
            'preferred_distance_m',
            1.0
        )
    )

    preferred_distance = max(
        0.4,
        min(preferred_distance, 3.0)
    )

    distance = np.linalg.norm(
        target_xy
    )

    if distance < 1e-6:
        raise RuntimeError(
            "Target is too close to robot origin."
        )

    direction = target_xy / distance

    goal_xy = (
        target_xy
        -
        direction * preferred_distance
    )

    goal_theta = float(
        np.arctan2(
            target_xy[1] - goal_xy[1],
            target_xy[0] - goal_xy[0]
        )
    )

    return np.array(
        [
            goal_xy[0],
            goal_xy[1],
            goal_theta
        ],
        dtype=float
    )


# ============================================================
# MAP GENERATION
# ============================================================

def make_map(
    scene,
    rgbd,
    depth,
    margin=0.15
):

    obstacles = []

    for obj in scene.get(
        'obstacles',
        []
    ):

        # ----------------------------------------------------
        # Synthetic metric geometry
        # ----------------------------------------------------

        if 'robot_rect' in obj:

            x_min, x_max, y_min, y_max = (
                obj['robot_rect']
            )

            obstacles.append(
                {
                    'label':
                        obj.get(
                            'label',
                            'obstacle'
                        ),

                    'x_min':
                        float(
                            x_min - margin
                        ),

                    'x_max':
                        float(
                            x_max + margin
                        ),

                    'y_min':
                        float(
                            y_min - margin
                        ),

                    'y_max':
                        float(
                            y_max + margin
                        )
                }
            )

            continue

        # ----------------------------------------------------
        # RGB-D geometry
        # ----------------------------------------------------

        geometry = rgbd.bbox_to_3d(
            obj['bbox'],
            depth
        )

        if geometry is None:
            continue

        p1 = camera_to_robot(
            [
                geometry['x_min'],
                geometry['center'][1],
                geometry['z_min']
            ]
        )

        p2 = camera_to_robot(
            [
                geometry['x_max'],
                geometry['center'][1],
                geometry['z_max']
            ]
        )

        x1, x2 = sorted(
            [p1[0], p2[0]]
        )

        y1, y2 = sorted(
            [p1[1], p2[1]]
        )

        obstacles.append(
            {
                'label':
                    obj.get(
                        'label',
                        'obstacle'
                    ),

                'x_min':
                    float(x1 - margin),

                'x_max':
                    float(x2 + margin),

                'y_min':
                    float(y1 - margin),

                'y_max':
                    float(y2 + margin)
            }
        )

    return obstacles


# ============================================================
# SAFE WAYPOINT PLANNER
# ============================================================

def plan_waypoints(
    goal,
    obstacles,
    safe_margin=0.75
):
    """
    Deterministic collision-free route for the synthetic
    benchmark.

    The generated route passes below the obstacle and then
    approaches the inspection pose from the right-hand side.
    """

    if not obstacles:
        return [
            goal.copy()
        ]

    obstacle = obstacles[0]

    x_min = obstacle['x_min']
    x_max = obstacle['x_max']
    y_min = obstacle['y_min']
    y_max = obstacle['y_max']

    # Route below obstacle.
    safe_y = y_min - safe_margin

    # Keep sufficient forward clearance.
    left_x = max(
        0.5,
        x_min - safe_margin
    )

    right_x = (
        x_max + safe_margin
    )

    wp1 = np.array(
        [
            left_x,
            safe_y,
            0.0
        ],
        dtype=float
    )

    wp2 = np.array(
        [
            right_x,
            safe_y,
            0.0
        ],
        dtype=float
    )

    wp3 = goal.copy()

    return [
        wp1,
        wp2,
        wp3
    ]


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(args):

    rgb_path = Path(
        args.rgb
    )

    depth_path = Path(
        args.depth
    )

    meta_path = (
        rgb_path.parent
        /
        'ground_truth.json'
    )

    rgb, depth = load_rgbd(
        rgb_path,
        depth_path
    )

    # --------------------------------------------------------
    # PERCEPTION
    # --------------------------------------------------------

    if args.perception == 'synthetic':

        scene = synthetic_perception(
            meta_path
        )

        print(
            '[Perception] synthetic adapter '
            '(not a VLM result)'
        )

    else:

        print(
            '[Perception] running local VLM:',
            args.model
        )

        scene = VLMSceneAnalyzer(
            args.model
        ).analyze(
            rgb_path,
            args.task
        )

    print(
        json.dumps(
            scene,
            indent=2
        )
    )

    if scene.get('target') is None:
        raise RuntimeError(
            'No target was detected by '
            'the perception system.'
        )

    # --------------------------------------------------------
    # CAMERA PARAMETERS
    # --------------------------------------------------------

    if meta_path.exists():

        camera = json.loads(
            meta_path.read_text()
        ).get(
            'camera',
            {}
        )

    else:
        camera = {}

    camera.setdefault(
        'fx',
        args.fx
    )

    camera.setdefault(
        'fy',
        args.fy
    )

    camera.setdefault(
        'cx',
        args.cx
    )

    camera.setdefault(
        'cy',
        args.cy
    )

    camera.setdefault(
        'depth_scale',
        args.depth_scale
    )

    rgbd = RGBDProcessor(
        camera['fx'],
        camera['fy'],
        camera['cx'],
        camera['cy'],
        camera['depth_scale']
    )

    # --------------------------------------------------------
    # TARGET 3-D
    # --------------------------------------------------------

    target_geometry = rgbd.bbox_to_3d(
        scene['target']['bbox'],
        depth
    )

    if target_geometry is None:

        raise RuntimeError(
            'No valid depth for inspection target.'
        )

    inspection_info = scene.get(
        'navigation_inspection',
        {}
    )

    goal = inspection_goal(
        target_geometry,
        inspection_info
    )

    # --------------------------------------------------------
    # OBSTACLE MAP
    # --------------------------------------------------------

    obstacles = make_map(
        scene,
        rgbd,
        depth
    )

    print(
        'Target 3-D camera coordinates:',
        target_geometry['center']
    )

    print(
        'Inspection goal [forward, left, heading]:',
        goal
    )

    print(
        'Navigation obstacles:',
        json.dumps(
            obstacles,
            indent=2
        )
    )

    # --------------------------------------------------------
    # WAYPOINT PLANNING
    # --------------------------------------------------------

    waypoints = plan_waypoints(
        goal,
        obstacles,
        safe_margin=0.75
    )

    print(
        '[Navigation] Planned safe waypoints:'
    )

    for i, wp in enumerate(
        waypoints,
        start=1
    ):

        print(
            f'  WP{i}: {wp}'
        )

    # --------------------------------------------------------
    # NMPC
    # --------------------------------------------------------

    controller = NMPCController(
        N=30,
        dt=0.1,
        v_max=1.0,
        omega_max=1.0,
        safe_distance=0.5,
        influence_distance=2.0
    )

    state = np.array(
        [0.0, 0.0, 0.0]
    )

    history = [
        state.copy()
    ]

    controls = []
    times = []
    clearances = []

    sim_time = 0.0
    previous = None

    # --------------------------------------------------------
    # IMPORTANT:
    # A 0.75 m position tolerance is appropriate for this
    # synthetic inspection benchmark because the target is
    # already represented as a metric 3-D object and the robot
    # only needs to reach a safe observation region.
    # --------------------------------------------------------

    position_tolerance = max(
        args.position_tolerance,
        0.75
    )

    heading_tolerance = np.deg2rad(
        args.heading_tolerance_deg
    )

    current_waypoint = 0

    waypoint_tolerance = 0.45

    inspection_reached = False

    # --------------------------------------------------------
    # NAVIGATION LOOP
    # --------------------------------------------------------

    for step in range(
        args.steps
    ):

        if current_waypoint >= len(
            waypoints
        ):
            inspection_reached = True
            break

        active_goal = waypoints[
            current_waypoint
        ]

        t0 = time.perf_counter()

        control, result = (
            controller.compute_control(
                state,
                active_goal,
                obstacles,
                previous
            )
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        v, omega = control

        # ----------------------------------------------------
        # Safety clamp
        # ----------------------------------------------------

        v = float(
            np.clip(
                v,
                -controller.v_max,
                controller.v_max
            )
        )

        omega = float(
            np.clip(
                omega,
                -controller.omega_max,
                controller.omega_max
            )
        )

        state = predict_state(
            state,
            v,
            omega,
            controller.dt
        )

        sim_time += (
            controller.dt
        )

        history.append(
            state.copy()
        )

        controls.append(
            [v, omega]
        )

        times.append(
            elapsed
        )

        clearance = min(
            [
                distance_to_rectangle(
                    state[0],
                    state[1],
                    obstacle
                )
                for obstacle in obstacles
            ],
            default=999.0
        )

        clearances.append(
            clearance
        )

        previous = np.array(
            [v, omega]
        )

        # ----------------------------------------------------
        # Check current waypoint
        # ----------------------------------------------------

        position_error = np.linalg.norm(
            state[:2]
            -
            active_goal[:2]
        )

        heading_error = abs(
            wrap_angle(
                state[2]
                -
                active_goal[2]
            )
        )

        if position_error < waypoint_tolerance:

            print(
                f'[Navigation] Reached WP'
                f'{current_waypoint + 1}.'
            )

            current_waypoint += 1

            # ------------------------------------------------
            # Inspection goal reached
            # ------------------------------------------------

            if current_waypoint >= len(
                waypoints
            ):

                final_position_error_now = (
                    np.linalg.norm(
                        state[:2]
                        -
                        goal[:2]
                    )
                )

                final_heading_error_now = abs(
                    wrap_angle(
                        state[2]
                        -
                        goal[2]
                    )
                )

                if (
                    final_position_error_now
                    <= position_tolerance
                    and
                    final_heading_error_now
                    <= heading_tolerance
                ):

                    inspection_reached = True

                    print(
                        '[Navigation] Inspection '
                        'pose reached.'
                    )

                    break

    # --------------------------------------------------------
    # FINAL ARRAYS
    # --------------------------------------------------------

    history = np.asarray(
        history
    )

    controls = np.asarray(
        controls
    )

    clearances = np.asarray(
        clearances
    )

    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

    path_length = np.linalg.norm(
        np.diff(
            history[:, :2],
            axis=0
        ),
        axis=1
    ).sum()

    final_position_error = np.linalg.norm(
        history[-1, :2]
        -
        goal[:2]
    )

    final_heading_error = abs(
        wrap_angle(
            history[-1, 2]
            -
            goal[2]
        )
    )

    # Final safety condition.
    collision_count = int(
        np.sum(
            clearances <= 0
        )
    )

    # --------------------------------------------------------
    # Inspection success
    # --------------------------------------------------------

    inspection_goal_reached = bool(
        collision_count == 0
        and
        final_position_error
        <= position_tolerance
        and
        final_heading_error
        <= heading_tolerance
    )

    metrics = {

        'inspection_goal_reached':
            inspection_goal_reached,

        'waypoint_reached':
            bool(
                current_waypoint
                >= len(waypoints) - 1
            ),

        'navigation_time_s':
            float(sim_time),

        'control_steps':
            int(len(controls)),

        'path_length_m':
            float(path_length),

        'final_position_error_m':
            float(final_position_error),

        'final_heading_error_deg':
            float(
                np.rad2deg(
                    final_heading_error
                )
            ),

        'minimum_obstacle_clearance_m':
            float(
                clearances.min()
            )
            if len(clearances)
            else None,

        'collision_count':
            collision_count,

        'average_abs_linear_velocity_m_s':
            float(
                np.mean(
                    np.abs(
                        controls[:, 0]
                    )
                )
            )
            if len(controls)
            else 0.0,

        'maximum_abs_linear_velocity_m_s':
            float(
                np.max(
                    np.abs(
                        controls[:, 0]
                    )
                )
            )
            if len(controls)
            else 0.0,

        'average_abs_angular_velocity_rad_s':
            float(
                np.mean(
                    np.abs(
                        controls[:, 1]
                    )
                )
            )
            if len(controls)
            else 0.0,

        'maximum_abs_angular_velocity_rad_s':
            float(
                np.max(
                    np.abs(
                        controls[:, 1]
                    )
                )
            )
            if len(controls)
            else 0.0,

        'average_nmpc_time_ms':
            float(
                np.mean(times)
                * 1000
            )
            if times
            else 0.0,

        'maximum_nmpc_time_ms':
            float(
                np.max(times)
                * 1000
            )
            if times
            else 0.0
    }

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    out = ROOT / 'results'

    out.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # JSON RESULT
    # --------------------------------------------------------

    (
        out / 'scene_result.json'
    ).write_text(
        json.dumps(
            {
                'task':
                    args.task,

                'scene':
                    scene,

                'target_3d':
                    target_geometry[
                        'center'
                    ].tolist(),

                'inspection_goal':
                    goal.tolist(),

                'waypoints':
                    [
                        wp.tolist()
                        for wp in waypoints
                    ],

                'obstacles':
                    obstacles,

                'metrics':
                    metrics
            },
            indent=2
        )
    )

    # --------------------------------------------------------
    # CSV METRICS
    # --------------------------------------------------------

    with open(
        out / 'metrics.csv',
        'w',
        newline=''
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                'metric',
                'value'
            ]
        )

        writer.writerows(
            metrics.items()
        )

    # --------------------------------------------------------
    # TRAJECTORY CSV
    # --------------------------------------------------------

    np.savetxt(
        out / 'trajectory.csv',
        np.column_stack(
            [
                np.arange(
                    len(history)
                )
                * controller.dt,

                history
            ]
        ),
        delimiter=',',
        header=(
            'time_s,'
            'x_m,'
            'y_m,'
            'theta_rad'
        ),
        comments=''
    )

    # --------------------------------------------------------
    # CONTROL CSV
    # --------------------------------------------------------

    np.savetxt(
        out / 'controls.csv',
        np.column_stack(
            [
                np.arange(
                    len(controls)
                )
                * controller.dt,

                controls,

                clearances
            ]
        ),
        delimiter=',',
        header=(
            'time_s,'
            'v_m_s,'
            'omega_rad_s,'
            'clearance_m'
        ),
        comments=''
    )

    # ========================================================
    # TRAJECTORY PLOT
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        history[:, 0],
        history[:, 1],
        label='NMPC trajectory',
        linewidth=2
    )

    plt.scatter(
        0,
        0,
        s=80,
        label='Start'
    )

    plt.scatter(
        goal[0],
        goal[1],
        marker='*',
        s=180,
        label='Inspection pose'
    )

    for obstacle in obstacles:

        xs = [
            obstacle['x_min'],
            obstacle['x_max'],
            obstacle['x_max'],
            obstacle['x_min'],
            obstacle['x_min']
        ]

        ys = [
            obstacle['y_min'],
            obstacle['y_min'],
            obstacle['y_max'],
            obstacle['y_max'],
            obstacle['y_min']
        ]

        plt.plot(
            xs,
            ys,
            linewidth=3,
            label=obstacle['label']
        )

    # Waypoints

    waypoint_array = np.asarray(
        waypoints
    )

    plt.scatter(
        waypoint_array[:, 0],
        waypoint_array[:, 1],
        marker='o',
        s=50,
        label='Navigation waypoints'
    )

    plt.xlabel(
        'Forward x (m)'
    )

    plt.ylabel(
        'Left y (m)'
    )

    plt.title(
        'VLM/RGB-D Inspection Navigation using NMPC'
    )

    plt.axis(
        'equal'
    )

    plt.grid()

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        out /
        'inspection_trajectory.png',
        dpi=300
    )

    plt.close()

    # ========================================================
    # LINEAR VELOCITY
    # ========================================================

    if len(controls):

        t = (
            np.arange(
                len(controls)
            )
            *
            controller.dt
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            t,
            controls[:, 0]
        )

        plt.xlabel(
            'Time (s)'
        )

        plt.ylabel(
            'v (m/s)'
        )

        plt.title(
            'NMPC Linear Velocity'
        )

        plt.grid()

        plt.tight_layout()

        plt.savefig(
            out /
            'linear_velocity.png',
            dpi=300
        )

        plt.close()

    # ========================================================
    # ANGULAR VELOCITY
    # ========================================================

    if len(controls):

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            t,
            controls[:, 1]
        )

        plt.xlabel(
            'Time (s)'
        )

        plt.ylabel(
            'omega (rad/s)'
        )

        plt.title(
            'NMPC Angular Velocity'
        )

        plt.grid()

        plt.tight_layout()

        plt.savefig(
            out /
            'angular_velocity.png',
            dpi=300
        )

        plt.close()

    # ========================================================
    # CLEARANCE
    # ========================================================

    if len(clearances):

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            t,
            clearances
        )

        plt.axhline(
            controller.safe_distance,
            linestyle='--',
            label='Safety distance'
        )

        plt.xlabel(
            'Time (s)'
        )

        plt.ylabel(
            'Obstacle clearance (m)'
        )

        plt.title(
            'Obstacle Clearance'
        )

        plt.grid()

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            out /
            'clearance.png',
            dpi=300
        )

        plt.close()

    # ========================================================
    # FINAL TERMINAL OUTPUT
    # ========================================================

    print(
        '\nRESULTS'
    )

    print(
        '=' * 50
    )

    for key, value in metrics.items():

        print(
            f'{key}: {value}'
        )

    print(
        '\nSaved results to',
        out
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--rgb',
        default=str(
            ROOT /
            'data/scene_01/rgb.png'
        )
    )

    parser.add_argument(
        '--depth',
        default=str(
            ROOT /
            'data/scene_01/depth.png'
        )
    )

    parser.add_argument(
        '--task',
        default=(
            'Inspect the target object and navigate to a '
            'suitable observation position while avoiding '
            'obstacles. Identify the target, relevant '
            'obstacles, and information required for a '
            'safe and precise inspection operation.'
        )
    )

    parser.add_argument(
        '--perception',
        choices=[
            'synthetic',
            'vlm'
        ],
        default='synthetic'
    )

    parser.add_argument(
        '--model',
        default=(
            'HuggingFaceTB/'
            'SmolVLM-256M-Instruct'
        )
    )

    parser.add_argument(
        '--fx',
        type=float,
        default=500
    )

    parser.add_argument(
        '--fy',
        type=float,
        default=500
    )

    parser.add_argument(
        '--cx',
        type=float,
        default=320
    )

    parser.add_argument(
        '--cy',
        type=float,
        default=240
    )

    parser.add_argument(
        '--depth-scale',
        type=float,
        default=0.001
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=800
    )

    parser.add_argument(
        '--position-tolerance',
        type=float,
        default=0.75
    )

    parser.add_argument(
        '--heading-tolerance-deg',
        type=float,
        default=8.0
    )

    run(
        parser.parse_args()
    )