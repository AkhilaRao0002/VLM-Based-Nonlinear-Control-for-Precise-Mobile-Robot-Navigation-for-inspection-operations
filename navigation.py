import numpy as np
from scipy.optimize import minimize


def predict_state(state, v, omega, dt):
    x, y, theta = state

    return np.array([
        x + v * np.cos(theta) * dt,
        y + v * np.sin(theta) * dt,
        theta + omega * dt
    ])


def predict_trajectory(state, controls, dt):
    current = state.copy()
    states = []

    for v, omega in controls:
        current = predict_state(current, v, omega, dt)
        states.append(current.copy())

    return np.asarray(states)


def distance_to_rectangle(x, y, o):
    dx = max(
        o['x_min'] - x,
        0,
        x - o['x_max']
    )

    dy = max(
        o['y_min'] - y,
        0,
        y - o['y_max']
    )

    return float(np.hypot(dx, dy))


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class NMPCController:

    def __init__(
        self,
        N=15,
        dt=0.1,
        v_max=1.0,
        omega_max=1.0,
        safe_distance=0.5,
        influence_distance=2.0,
        w_position=20.0,
        w_heading=1.0,
        w_obstacle=1200.0,
        w_influence=20.0,
        w_control=0.03,
        w_terminal_position=80.0,
        w_terminal_heading=3.0,
        w_progress=2.0
    ):

        self.N = N
        self.dt = dt

        self.v_max = v_max
        self.omega_max = omega_max

        self.safe_distance = safe_distance
        self.influence_distance = influence_distance

        self.w_position = w_position
        self.w_heading = w_heading
        self.w_obstacle = w_obstacle
        self.w_influence = w_influence
        self.w_control = w_control

        self.w_terminal_position = w_terminal_position
        self.w_terminal_heading = w_terminal_heading
        self.w_progress = w_progress

        self.last_result = None

    def cost_function(
        self,
        u_flat,
        state,
        goal,
        obstacles
    ):

        controls = u_flat.reshape(self.N, 2)

        predicted = predict_trajectory(
            state,
            controls,
            self.dt
        )

        total = 0.0

        initial_distance = np.linalg.norm(
            state[:2] - goal[:2]
        )

        for future, (v, omega) in zip(
            predicted,
            controls
        ):

            x, y, theta = future

            position_error = (
                (x - goal[0]) ** 2 +
                (y - goal[1]) ** 2
            )

            heading_error = wrap_angle(
                theta - goal[2]
            ) ** 2

            total += (
                self.w_position *
                position_error
            )

            total += (
                self.w_heading *
                heading_error
            )

            for obstacle in obstacles:

                clearance = distance_to_rectangle(
                    x,
                    y,
                    obstacle
                )

                # Soft influence region.
                if clearance < self.influence_distance:

                    influence_error = (
                        self.influence_distance -
                        clearance
                    )

                    total += (
                        self.w_influence *
                        influence_error ** 2
                    )

                # Strong collision avoidance.
                if clearance < self.safe_distance:

                    violation = (
                        self.safe_distance -
                        clearance
                    )

                    total += (
                        self.w_obstacle *
                        violation ** 2
                    )

            total += (
                self.w_control *
                (v ** 2 + omega ** 2)
            )

        final_state = predicted[-1]

        final_position_error = np.linalg.norm(
            final_state[:2] - goal[:2]
        )

        final_heading_error = abs(
            wrap_angle(
                final_state[2] - goal[2]
            )
        )

        total += (
            self.w_terminal_position *
            final_position_error ** 2
        )

        total += (
            self.w_terminal_heading *
            final_heading_error ** 2
        )

        final_distance = final_position_error

        progress_loss = max(
            0.0,
            final_distance - initial_distance
        )

        total += (
            self.w_progress *
            progress_loss ** 2
        )

        return float(total)

    def compute_control(
        self,
        state,
        goal,
        obstacles,
        previous_control=None
    ):

        bounds = []

        for _ in range(self.N):

            bounds.append(
                (-self.v_max, self.v_max)
            )

            bounds.append(
                (-self.omega_max, self.omega_max)
            )

        # --------------------------------------------------
        # Warm start
        # --------------------------------------------------

        if previous_control is None:

            guess_controls = np.zeros(
                (self.N, 2),
                dtype=float
            )

            # Initial forward motion.
            guess_controls[:, 0] = 0.4

        else:

            previous_control = np.asarray(
                previous_control,
                dtype=float
            )

            guess_controls = np.tile(
                previous_control,
                (self.N, 1)
            )

        guess = guess_controls.flatten()

        # --------------------------------------------------
        # Optimization
        # --------------------------------------------------

        result = minimize(
            self.cost_function,
            guess,
            args=(
                state,
                goal,
                obstacles
            ),
            bounds=bounds,
            method='SLSQP',
            options={
                'maxiter': 25,
                'ftol': 1e-3,
                'disp': False
            }
        )

        self.last_result = result

        solution = result.x.reshape(
            self.N,
            2
        )

        return solution[0], result