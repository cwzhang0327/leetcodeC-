"""
Maneuvering Target Tracking (机动目标跟踪)

This module implements several classic algorithms for tracking maneuvering targets:
1. Kalman Filter (KF) - For linear constant-velocity target tracking
2. Extended Kalman Filter (EKF) - For nonlinear target tracking
3. Interacting Multiple Model (IMM) - For maneuvering target tracking
   using multiple motion models (CV + CT)

Models used:
- CV (Constant Velocity): Assumes the target moves at constant velocity.
- CT (Coordinated Turn): Assumes the target performs a coordinated turn
  with a constant turn rate.

Reference:
- Y. Bar-Shalom, X. R. Li, and T. Kirubarajan,
  "Estimation with Applications to Tracking and Navigation," Wiley, 2001.
- E. Mazor, A. Averbuch, Y. Bar-Shalom, and J. Dahan,
  "Interacting Multiple Model Methods in Target Tracking: A Survey,"
  IEEE Transactions on Aerospace and Electronic Systems, 1998.

Author: cwzhang0327
Date: 2026-03-12
"""

import math
import numpy as np


# ============================================================
# Kalman Filter for Constant Velocity (CV) Model
# ============================================================
class KalmanFilterCV:
    """
    Standard Kalman Filter with a Constant Velocity (CV) motion model.

    State vector: x = [px, vx, py, vy]^T
      - px, py: position
      - vx, vy: velocity

    Measurement: z = [px, py]^T
    """

    def __init__(self, dt=1.0, process_noise_std=1.0, measurement_noise_std=10.0):
        """
        Args:
            dt: Time step between measurements.
            process_noise_std: Standard deviation of process noise acceleration.
            measurement_noise_std: Standard deviation of measurement noise.
        """
        self.dt = dt
        self.n = 4  # state dimension
        self.m = 2  # measurement dimension

        # State transition matrix (CV model)
        self.F = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1]
        ], dtype=float)

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=float)

        # Process noise covariance
        q = process_noise_std ** 2
        self.Q = q * np.array([
            [dt**3/3, dt**2/2,       0,       0],
            [dt**2/2,      dt,       0,       0],
            [      0,       0, dt**3/3, dt**2/2],
            [      0,       0, dt**2/2,      dt]
        ], dtype=float)

        # Measurement noise covariance
        r = measurement_noise_std ** 2
        self.R = r * np.eye(self.m)

        # Initial state and covariance
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 1000.0

    def init_state(self, z):
        """Initialize state from the first measurement."""
        self.x = np.array([z[0], 0.0, z[1], 0.0])
        self.P = np.eye(self.n) * 1000.0

    def predict(self):
        """Predict step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """Update step with measurement z = [px, py]."""
        z = np.asarray(z, dtype=float)
        y = z - self.H @ self.x          # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

    def step(self, z):
        """Run one predict-update cycle."""
        self.predict()
        return self.update(z)


# ============================================================
# Extended Kalman Filter for Coordinated Turn (CT) Model
# ============================================================
class EKFCoordinatedTurn:
    """
    Extended Kalman Filter with a Coordinated Turn (CT) motion model.

    State vector: x = [px, vx, py, vy, omega]^T
      - px, py: position
      - vx, vy: velocity
      - omega: turn rate (rad/s)

    Measurement: z = [px, py]^T
    """

    def __init__(self, dt=1.0, process_noise_std=1.0, omega_noise_std=0.1,
                 measurement_noise_std=10.0):
        self.dt = dt
        self.n = 5
        self.m = 2

        self.q_acc = process_noise_std ** 2
        self.q_omega = omega_noise_std ** 2
        self.R = (measurement_noise_std ** 2) * np.eye(self.m)

        self.H = np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=float)

        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 1000.0

    def init_state(self, z, omega=0.0):
        """Initialize state from first measurement."""
        self.x = np.array([z[0], 0.0, z[1], 0.0, omega])
        self.P = np.eye(self.n) * 1000.0

    def _f(self, x):
        """Nonlinear state transition function for CT model."""
        px, vx, py, vy, omega = x
        dt = self.dt
        if abs(omega) < 1e-6:
            # Degenerate to CV model when omega ≈ 0
            px_new = px + vx * dt
            vx_new = vx
            py_new = py + vy * dt
            vy_new = vy
        else:
            sin_wt = math.sin(omega * dt)
            cos_wt = math.cos(omega * dt)
            px_new = px + (vx * sin_wt - vy * (1 - cos_wt)) / omega
            vx_new = vx * cos_wt - vy * sin_wt
            py_new = py + (vx * (1 - cos_wt) + vy * sin_wt) / omega
            vy_new = vx * sin_wt + vy * cos_wt
        omega_new = omega
        return np.array([px_new, vx_new, py_new, vy_new, omega_new])

    def _jacobian_F(self, x):
        """Compute the Jacobian of f with respect to x."""
        _, vx, _, vy, omega = x
        dt = self.dt
        F = np.eye(self.n)
        if abs(omega) < 1e-6:
            F[0, 1] = dt
            F[2, 3] = dt
        else:
            sin_wt = math.sin(omega * dt)
            cos_wt = math.cos(omega * dt)
            w = omega
            F[0, 1] = sin_wt / w
            F[0, 3] = -(1 - cos_wt) / w
            F[0, 4] = (vx * (w*dt*cos_wt - sin_wt)
                        - vy * (w*dt*sin_wt - 1 + cos_wt)) / w**2
            F[1, 1] = cos_wt
            F[1, 3] = -sin_wt
            F[1, 4] = -vx * dt * sin_wt - vy * dt * cos_wt
            F[2, 1] = (1 - cos_wt) / w
            F[2, 3] = sin_wt / w
            F[2, 4] = (vx * (w*dt*sin_wt - 1 + cos_wt)
                        + vy * (w*dt*cos_wt - sin_wt)) / w**2
            F[3, 1] = sin_wt
            F[3, 3] = cos_wt
            F[3, 4] = vx * dt * cos_wt - vy * dt * sin_wt
        return F

    def _process_noise(self):
        """Build the process noise covariance matrix Q."""
        dt = self.dt
        G = np.array([
            [dt**2 / 2, 0,         0],
            [dt,        0,         0],
            [0,         dt**2 / 2, 0],
            [0,         dt,        0],
            [0,         0,         dt]
        ], dtype=float)
        D = np.diag([self.q_acc, self.q_acc, self.q_omega])
        return G @ D @ G.T

    def predict(self):
        """Predict step using the nonlinear CT model."""
        F_jac = self._jacobian_F(self.x)
        self.x = self._f(self.x)
        Q = self._process_noise()
        self.P = F_jac @ self.P @ F_jac.T + Q
        return self.x.copy()

    def update(self, z):
        """Update step with measurement z = [px, py]."""
        z = np.asarray(z, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

    def step(self, z):
        """Run one predict-update cycle."""
        self.predict()
        return self.update(z)


# ============================================================
# Interacting Multiple Model (IMM) Estimator
# ============================================================
class IMMEstimator:
    """
    Interacting Multiple Model (IMM) estimator for maneuvering target tracking.

    Combines two models:
      - Model 1: Constant Velocity (CV) via KalmanFilterCV
      - Model 2: Coordinated Turn (CT) via EKFCoordinatedTurn

    The IMM algorithm consists of four steps each cycle:
      1. Interaction (mixing)
      2. Model-conditioned prediction
      3. Model-conditioned update
      4. Model probability update and state combination
    """

    def __init__(self, dt=1.0, process_noise_std=1.0, omega_noise_std=0.1,
                 measurement_noise_std=10.0, transition_prob=0.95):
        """
        Args:
            dt: Time step.
            process_noise_std: Process noise standard deviation for acceleration.
            omega_noise_std: Process noise standard deviation for turn rate.
            measurement_noise_std: Measurement noise standard deviation.
            transition_prob: Probability of staying in the same model (diagonal
                             of the Markov transition matrix).
        """
        self.n_models = 2

        # Model filters
        self.filters = [
            KalmanFilterCV(dt, process_noise_std, measurement_noise_std),
            EKFCoordinatedTurn(dt, process_noise_std, omega_noise_std,
                               measurement_noise_std),
        ]

        # Markov transition probability matrix
        p = transition_prob
        self.TPM = np.array([
            [p,     1 - p],
            [1 - p, p    ]
        ])

        # Model probabilities (initially equal)
        self.mu = np.array([0.5, 0.5])

    def init_state(self, z):
        """Initialize all model filters with the first measurement."""
        self.filters[0].init_state(z)
        self.filters[1].init_state(z)

    def _pad_state(self, x, target_dim):
        """Pad a state vector to the target dimension with zeros."""
        if len(x) >= target_dim:
            return x[:target_dim]
        return np.concatenate([x, np.zeros(target_dim - len(x))])

    def _pad_covariance(self, P, target_dim):
        """Pad a covariance matrix to the target dimension."""
        n = P.shape[0]
        if n >= target_dim:
            return P[:target_dim, :target_dim]
        P_new = np.eye(target_dim) * 1000.0
        P_new[:n, :n] = P
        return P_new

    def step(self, z):
        """
        Run one IMM cycle: interaction -> prediction -> update -> combination.

        Args:
            z: Measurement vector [px, py].

        Returns:
            Combined state estimate [px, vx, py, vy].
        """
        z = np.asarray(z, dtype=float)

        # ----- Step 1: Compute mixing probabilities -----
        c_bar = self.TPM.T @ self.mu  # predicted model probabilities
        c_bar = np.maximum(c_bar, 1e-30)  # avoid division by zero

        mu_ij = np.zeros((self.n_models, self.n_models))
        for i in range(self.n_models):
            for j in range(self.n_models):
                mu_ij[i, j] = self.TPM[i, j] * self.mu[i] / c_bar[j]

        # ----- Step 2: State mixing (interaction) -----
        max_dim = max(f.n for f in self.filters)
        states = [self._pad_state(f.x, max_dim) for f in self.filters]
        covs = [self._pad_covariance(f.P, max_dim) for f in self.filters]

        mixed_states = []
        mixed_covs = []
        for j in range(self.n_models):
            x_mixed = np.zeros(max_dim)
            for i in range(self.n_models):
                x_mixed += mu_ij[i, j] * states[i]
            mixed_states.append(x_mixed)

            P_mixed = np.zeros((max_dim, max_dim))
            for i in range(self.n_models):
                diff = states[i] - x_mixed
                P_mixed += mu_ij[i, j] * (covs[i] + np.outer(diff, diff))
            mixed_covs.append(P_mixed)

        # Set mixed states back into filters
        self.filters[0].x = mixed_states[0][:self.filters[0].n]
        self.filters[0].P = mixed_covs[0][:self.filters[0].n, :self.filters[0].n]
        self.filters[1].x = mixed_states[1][:self.filters[1].n]
        self.filters[1].P = mixed_covs[1][:self.filters[1].n, :self.filters[1].n]

        # ----- Step 3: Model-conditioned prediction and update -----
        likelihoods = np.zeros(self.n_models)
        for j, f in enumerate(self.filters):
            f.predict()
            # Compute innovation likelihood
            z_pred = f.H @ f.x
            y = z - z_pred
            S = f.H @ f.P @ f.H.T + f.R
            det_S = np.linalg.det(S)
            if det_S < 1e-30:
                det_S = 1e-30
            inv_S = np.linalg.inv(S)
            exponent = -0.5 * y @ inv_S @ y
            likelihoods[j] = math.exp(exponent) / math.sqrt(
                (2 * math.pi) ** f.m * det_S
            )
            f.update(z)

        # ----- Step 4: Model probability update -----
        self.mu = c_bar * likelihoods
        mu_sum = np.sum(self.mu)
        if mu_sum < 1e-30:
            self.mu = np.ones(self.n_models) / self.n_models
        else:
            self.mu /= mu_sum

        # ----- Step 5: State combination -----
        # Combine into a 4-dim state [px, vx, py, vy]
        out_dim = 4
        combined_x = np.zeros(out_dim)
        for j in range(self.n_models):
            combined_x += self.mu[j] * self._pad_state(
                self.filters[j].x, out_dim
            )

        return combined_x


# ============================================================
# Simulation and Demo
# ============================================================
def generate_maneuvering_trajectory(n_steps=100, dt=1.0):
    """
    Generate a ground-truth trajectory of a maneuvering target.

    The target moves in three phases:
      1. Constant velocity (straight line)
      2. Coordinated turn (circular arc)
      3. Constant velocity (straight line, new heading)

    Returns:
        true_positions: (n_steps, 2) array of [px, py].
        measurements:   (n_steps, 2) array of noisy [px, py].
    """
    measurement_noise_std = 10.0
    true_positions = np.zeros((n_steps, 2))
    px, py = 0.0, 0.0
    vx, vy = 30.0, 10.0  # initial velocity

    for k in range(n_steps):
        true_positions[k] = [px, py]

        if k < 30:
            # Phase 1: constant velocity
            px += vx * dt
            py += vy * dt
        elif k < 70:
            # Phase 2: coordinated turn (omega = 0.05 rad/s)
            omega = 0.05
            cos_w = math.cos(omega * dt)
            sin_w = math.sin(omega * dt)
            vx_new = vx * cos_w - vy * sin_w
            vy_new = vx * sin_w + vy * cos_w
            px += (vx * sin_w - vy * (1 - cos_w)) / omega
            py += (vx * (1 - cos_w) + vy * sin_w) / omega
            vx, vy = vx_new, vy_new
        else:
            # Phase 3: constant velocity (new heading after turn)
            px += vx * dt
            py += vy * dt

    # Add measurement noise
    noise = np.random.randn(n_steps, 2) * measurement_noise_std
    measurements = true_positions + noise

    return true_positions, measurements


def run_demo():
    """
    Demonstrate maneuvering target tracking with KF, EKF, and IMM.

    Prints RMSE (Root Mean Square Error) for each tracker.
    """
    np.random.seed(42)
    n_steps = 100
    dt = 1.0
    measurement_noise_std = 10.0

    true_pos, measurements = generate_maneuvering_trajectory(n_steps, dt)

    # Initialize trackers
    kf = KalmanFilterCV(dt=dt, process_noise_std=1.0,
                        measurement_noise_std=measurement_noise_std)
    ekf = EKFCoordinatedTurn(dt=dt, process_noise_std=1.0,
                             omega_noise_std=0.1,
                             measurement_noise_std=measurement_noise_std)
    imm = IMMEstimator(dt=dt, process_noise_std=1.0,
                       omega_noise_std=0.1,
                       measurement_noise_std=measurement_noise_std)

    kf.init_state(measurements[0])
    ekf.init_state(measurements[0])
    imm.init_state(measurements[0])

    kf_estimates = np.zeros((n_steps, 2))
    ekf_estimates = np.zeros((n_steps, 2))
    imm_estimates = np.zeros((n_steps, 2))

    for k in range(n_steps):
        z = measurements[k]

        x_kf = kf.step(z)
        kf_estimates[k] = [x_kf[0], x_kf[2]]

        x_ekf = ekf.step(z)
        ekf_estimates[k] = [x_ekf[0], x_ekf[2]]

        x_imm = imm.step(z)
        imm_estimates[k] = [x_imm[0], x_imm[2]]

    # Compute RMSE
    def rmse(est, truth):
        return np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1)))

    print("=" * 60)
    print("Maneuvering Target Tracking - Results")
    print("=" * 60)
    print(f"  KF  (Constant Velocity)  RMSE: {rmse(kf_estimates, true_pos):.2f} m")
    print(f"  EKF (Coordinated Turn)   RMSE: {rmse(ekf_estimates, true_pos):.2f} m")
    print(f"  IMM (CV + CT)            RMSE: {rmse(imm_estimates, true_pos):.2f} m")
    print("=" * 60)
    print()
    print("IMM model probabilities at end:")
    print(f"  P(CV) = {imm.mu[0]:.4f},  P(CT) = {imm.mu[1]:.4f}")

    return kf_estimates, ekf_estimates, imm_estimates, true_pos, measurements


if __name__ == "__main__":
    run_demo()
