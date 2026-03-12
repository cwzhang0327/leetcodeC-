/**
 * Maneuvering Target Tracking (机动目标跟踪)
 *
 * This file implements a Kalman Filter with Constant Velocity (CV) model
 * and an Interacting Multiple Model (IMM) estimator combining CV and
 * Coordinated Turn (CT) models for tracking maneuvering targets.
 *
 * Algorithms implemented:
 *   1. Kalman Filter (CV model)  - For straight-line motion
 *   2. Extended Kalman Filter (CT model) - For turning motion
 *   3. IMM Estimator (CV + CT)  - Adaptively switches between models
 *
 * Reference:
 *   - Y. Bar-Shalom, X. R. Li, T. Kirubarajan,
 *     "Estimation with Applications to Tracking and Navigation," Wiley, 2001.
 *
 * Compile: g++ -std=c++17 -O2 -o tracking ManeuveringTargetTracking.cpp -lm
 *
 * Author: cwzhang0327
 * Date: 2026-03-12
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <numeric>
#include <random>
#include <iomanip>

// ============================================================
// Simple Matrix Operations (no external library needed)
// ============================================================

using Vec = std::vector<double>;
using Mat = std::vector<std::vector<double>>;

Mat matZeros(int rows, int cols) {
    return Mat(rows, Vec(cols, 0.0));
}

Mat matEye(int n) {
    Mat I = matZeros(n, n);
    for (int i = 0; i < n; i++) I[i][i] = 1.0;
    return I;
}

Mat matMul(const Mat& A, const Mat& B) {
    int m = A.size(), n = B[0].size(), p = B.size();
    Mat C = matZeros(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < p; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

Vec matVecMul(const Mat& A, const Vec& x) {
    int m = A.size();
    Vec y(m, 0.0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < (int)A[i].size(); j++)
            y[i] += A[i][j] * x[j];
    return y;
}

Mat matTranspose(const Mat& A) {
    int m = A.size(), n = A[0].size();
    Mat AT = matZeros(n, m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            AT[j][i] = A[i][j];
    return AT;
}

Mat matAdd(const Mat& A, const Mat& B) {
    int m = A.size(), n = A[0].size();
    Mat C = matZeros(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Mat matSub(const Mat& A, const Mat& B) {
    int m = A.size(), n = A[0].size();
    Mat C = matZeros(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

Mat matScale(const Mat& A, double s) {
    int m = A.size(), n = A[0].size();
    Mat C = matZeros(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] * s;
    return C;
}

// 2x2 matrix inverse
Mat matInv2x2(const Mat& A) {
    double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (std::abs(det) < 1e-30) det = 1e-30;
    Mat inv = matZeros(2, 2);
    inv[0][0] =  A[1][1] / det;
    inv[0][1] = -A[0][1] / det;
    inv[1][0] = -A[1][0] / det;
    inv[1][1] =  A[0][0] / det;
    return inv;
}

double matDet2x2(const Mat& A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

Mat outerProduct(const Vec& a, const Vec& b) {
    int m = a.size(), n = b.size();
    Mat C = matZeros(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = a[i] * b[j];
    return C;
}

// ============================================================
// Kalman Filter - Constant Velocity (CV) Model
// ============================================================
class KalmanFilterCV {
public:
    int n = 4;  // state dim: [px, vx, py, vy]
    int m = 2;  // meas dim:  [px, py]
    Vec x;
    Mat P, F, H, Q, R;

    KalmanFilterCV(double dt = 1.0, double q_std = 1.0, double r_std = 10.0) {
        double q = q_std * q_std;
        double r = r_std * r_std;

        x = Vec(n, 0.0);
        P = matScale(matEye(n), 1000.0);

        F = matEye(n);
        F[0][1] = dt; F[2][3] = dt;

        H = matZeros(m, n);
        H[0][0] = 1.0; H[1][2] = 1.0;

        Q = matZeros(n, n);
        double dt3 = dt*dt*dt/3.0, dt2 = dt*dt/2.0;
        Q[0][0] = dt3; Q[0][1] = dt2;
        Q[1][0] = dt2; Q[1][1] = dt;
        Q[2][2] = dt3; Q[2][3] = dt2;
        Q[3][2] = dt2; Q[3][3] = dt;
        Q = matScale(Q, q);

        R = matScale(matEye(m), r);
    }

    void initState(double px, double py) {
        x = {px, 0.0, py, 0.0};
        P = matScale(matEye(n), 1000.0);
    }

    void predict() {
        x = matVecMul(F, x);
        P = matAdd(matMul(matMul(F, P), matTranspose(F)), Q);
    }

    void update(double zpx, double zpy) {
        Vec z = {zpx, zpy};
        Vec z_pred = matVecMul(H, x);
        Vec y = {z[0] - z_pred[0], z[1] - z_pred[1]};

        Mat S = matAdd(matMul(matMul(H, P), matTranspose(H)), R);
        Mat S_inv = matInv2x2(S);
        Mat K = matMul(matMul(P, matTranspose(H)), S_inv);

        Vec Ky = matVecMul(K, y);
        for (int i = 0; i < n; i++) x[i] += Ky[i];

        Mat I = matEye(n);
        Mat KH = matMul(K, H);
        P = matMul(matSub(I, KH), P);
    }

    // Return innovation likelihood for IMM
    double likelihood(double zpx, double zpy) const {
        Vec z = {zpx, zpy};
        Vec z_pred = matVecMul(H, x);
        Vec y = {z[0] - z_pred[0], z[1] - z_pred[1]};
        Mat S = matAdd(matMul(matMul(H, P), matTranspose(H)), R);
        double det = matDet2x2(S);
        if (det < 1e-30) det = 1e-30;
        Mat S_inv = matInv2x2(S);
        Vec S_inv_y = matVecMul(S_inv, y);
        double exponent = -0.5 * (y[0]*S_inv_y[0] + y[1]*S_inv_y[1]);
        return std::exp(exponent) / std::sqrt(4.0 * M_PI * M_PI * det);
    }
};

// ============================================================
// Extended Kalman Filter - Coordinated Turn (CT) Model
// ============================================================
class EKFCoordinatedTurn {
public:
    int n = 5;  // state: [px, vx, py, vy, omega]
    int m = 2;
    Vec x;
    Mat P, H, R;
    double q_acc, q_omega, dt;

    EKFCoordinatedTurn(double dt_ = 1.0, double q_std = 1.0,
                       double w_std = 0.1, double r_std = 10.0)
        : dt(dt_), q_acc(q_std*q_std), q_omega(w_std*w_std)
    {
        x = Vec(n, 0.0);
        P = matScale(matEye(n), 1000.0);

        H = matZeros(m, n);
        H[0][0] = 1.0; H[1][2] = 1.0;

        R = matScale(matEye(m), r_std * r_std);
    }

    void initState(double px, double py, double omega = 0.0) {
        x = {px, 0.0, py, 0.0, omega};
        P = matScale(matEye(n), 1000.0);
    }

    Vec transitionFunc(const Vec& s) const {
        double px = s[0], vx = s[1], py = s[2], vy = s[3], w = s[4];
        Vec s_new(n);
        if (std::abs(w) < 1e-6) {
            s_new = {px + vx*dt, vx, py + vy*dt, vy, w};
        } else {
            double sw = std::sin(w*dt), cw = std::cos(w*dt);
            s_new[0] = px + (vx*sw - vy*(1-cw)) / w;
            s_new[1] = vx*cw - vy*sw;
            s_new[2] = py + (vx*(1-cw) + vy*sw) / w;
            s_new[3] = vx*sw + vy*cw;
            s_new[4] = w;
        }
        return s_new;
    }

    Mat jacobianF(const Vec& s) const {
        double vx = s[1], vy = s[3], w = s[4];
        Mat Fj = matEye(n);
        if (std::abs(w) < 1e-6) {
            Fj[0][1] = dt; Fj[2][3] = dt;
        } else {
            double sw = std::sin(w*dt), cw = std::cos(w*dt);
            Fj[0][1] = sw / w;
            Fj[0][3] = -(1 - cw) / w;
            Fj[0][4] = (vx*(w*dt*cw - sw) - vy*(w*dt*sw - 1 + cw)) / (w*w);
            Fj[1][1] = cw; Fj[1][3] = -sw;
            Fj[1][4] = -vx*dt*sw - vy*dt*cw;
            Fj[2][1] = (1 - cw) / w;
            Fj[2][3] = sw / w;
            Fj[2][4] = (vx*(w*dt*sw - 1 + cw) + vy*(w*dt*cw - sw)) / (w*w);
            Fj[3][1] = sw; Fj[3][3] = cw;
            Fj[3][4] = vx*dt*cw - vy*dt*sw;
        }
        return Fj;
    }

    Mat processNoise() const {
        // G matrix (5x3)
        Mat G = matZeros(n, 3);
        G[0][0] = dt*dt/2; G[1][0] = dt;
        G[2][1] = dt*dt/2; G[3][1] = dt;
        G[4][2] = dt;
        // D = diag(q_acc, q_acc, q_omega)
        Mat D = matZeros(3, 3);
        D[0][0] = q_acc; D[1][1] = q_acc; D[2][2] = q_omega;
        return matMul(matMul(G, D), matTranspose(G));
    }

    void predict() {
        Mat Fj = jacobianF(x);
        x = transitionFunc(x);
        Mat Qn = processNoise();
        P = matAdd(matMul(matMul(Fj, P), matTranspose(Fj)), Qn);
    }

    void update(double zpx, double zpy) {
        Vec z = {zpx, zpy};
        Vec z_pred = matVecMul(H, x);
        Vec y = {z[0] - z_pred[0], z[1] - z_pred[1]};
        Mat S = matAdd(matMul(matMul(H, P), matTranspose(H)), R);
        Mat S_inv = matInv2x2(S);
        Mat K = matMul(matMul(P, matTranspose(H)), S_inv);
        Vec Ky = matVecMul(K, y);
        for (int i = 0; i < n; i++) x[i] += Ky[i];
        Mat I = matEye(n);
        P = matMul(matSub(I, matMul(K, H)), P);
    }

    double likelihood(double zpx, double zpy) const {
        Vec z = {zpx, zpy};
        Vec z_pred = matVecMul(H, x);
        Vec y = {z[0] - z_pred[0], z[1] - z_pred[1]};
        Mat S = matAdd(matMul(matMul(H, P), matTranspose(H)), R);
        double det = matDet2x2(S);
        if (det < 1e-30) det = 1e-30;
        Mat S_inv = matInv2x2(S);
        Vec Si_y = matVecMul(S_inv, y);
        double exponent = -0.5 * (y[0]*Si_y[0] + y[1]*Si_y[1]);
        return std::exp(exponent) / std::sqrt(4.0 * M_PI * M_PI * det);
    }
};

// ============================================================
// IMM Estimator (CV + CT)
// ============================================================
class IMMEstimator {
public:
    KalmanFilterCV cv;
    EKFCoordinatedTurn ct;
    double mu[2];       // model probabilities
    double TPM[2][2];   // transition probability matrix

    IMMEstimator(double dt = 1.0, double q_std = 1.0, double w_std = 0.1,
                 double r_std = 10.0, double p_stay = 0.95)
        : cv(dt, q_std, r_std), ct(dt, q_std, w_std, r_std)
    {
        mu[0] = 0.5; mu[1] = 0.5;
        TPM[0][0] = p_stay;   TPM[0][1] = 1 - p_stay;
        TPM[1][0] = 1 - p_stay; TPM[1][1] = p_stay;
    }

    void initState(double px, double py) {
        cv.initState(px, py);
        ct.initState(px, py);
    }

    void step(double zpx, double zpy, double* out_px, double* out_py) {
        // Step 1: Predicted model probabilities
        double c_bar[2];
        for (int j = 0; j < 2; j++) {
            c_bar[j] = 0;
            for (int i = 0; i < 2; i++)
                c_bar[j] += TPM[i][j] * mu[i];
            if (c_bar[j] < 1e-30) c_bar[j] = 1e-30;
        }

        // Mixing probabilities
        double mu_ij[2][2];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                mu_ij[i][j] = TPM[i][j] * mu[i] / c_bar[j];

        // Step 2: State mixing
        // Pad CV state (4-dim) to match CT (5-dim)
        int maxDim = 5;
        Vec states[2];
        Mat covs[2];
        states[0] = cv.x; states[0].resize(maxDim, 0.0);
        states[1] = ct.x;
        covs[0] = matScale(matEye(maxDim), 1000.0);
        for (int i = 0; i < cv.n; i++)
            for (int j = 0; j < cv.n; j++)
                covs[0][i][j] = cv.P[i][j];
        covs[1] = ct.P;

        Vec mixed_x[2];
        Mat mixed_P[2];
        for (int j = 0; j < 2; j++) {
            mixed_x[j] = Vec(maxDim, 0.0);
            for (int i = 0; i < 2; i++)
                for (int d = 0; d < maxDim; d++)
                    mixed_x[j][d] += mu_ij[i][j] * states[i][d];

            mixed_P[j] = matZeros(maxDim, maxDim);
            for (int i = 0; i < 2; i++) {
                Vec diff(maxDim);
                for (int d = 0; d < maxDim; d++)
                    diff[d] = states[i][d] - mixed_x[j][d];
                Mat spread = outerProduct(diff, diff);
                Mat term = matAdd(covs[i], spread);
                for (int r = 0; r < maxDim; r++)
                    for (int c = 0; c < maxDim; c++)
                        mixed_P[j][r][c] += mu_ij[i][j] * term[r][c];
            }
        }

        // Set mixed states back
        cv.x = Vec(mixed_x[0].begin(), mixed_x[0].begin() + cv.n);
        cv.P = matZeros(cv.n, cv.n);
        for (int i = 0; i < cv.n; i++)
            for (int j = 0; j < cv.n; j++)
                cv.P[i][j] = mixed_P[0][i][j];

        ct.x = mixed_x[1];
        ct.P = mixed_P[1];

        // Step 3: Predict
        cv.predict();
        ct.predict();

        // Compute likelihoods
        double L[2];
        L[0] = cv.likelihood(zpx, zpy);
        L[1] = ct.likelihood(zpx, zpy);

        // Step 4: Update
        cv.update(zpx, zpy);
        ct.update(zpx, zpy);

        // Step 5: Update model probabilities
        mu[0] = c_bar[0] * L[0];
        mu[1] = c_bar[1] * L[1];
        double mu_sum = mu[0] + mu[1];
        if (mu_sum < 1e-30) { mu[0] = 0.5; mu[1] = 0.5; }
        else { mu[0] /= mu_sum; mu[1] /= mu_sum; }

        // Step 6: Combine state estimates
        *out_px = mu[0] * cv.x[0] + mu[1] * ct.x[0];
        *out_py = mu[0] * cv.x[2] + mu[1] * ct.x[2];
    }
};

// ============================================================
// Trajectory Generation
// ============================================================
struct Point { double x, y; };

void generateManeuveringTrajectory(std::vector<Point>& truth,
                                   std::vector<Point>& meas,
                                   int n_steps, double dt,
                                   double meas_noise_std) {
    truth.resize(n_steps);
    meas.resize(n_steps);

    std::mt19937 gen(42);
    std::normal_distribution<> noise(0.0, meas_noise_std);

    double px = 0, py = 0, vx = 30, vy = 10;

    for (int k = 0; k < n_steps; k++) {
        truth[k] = {px, py};

        if (k < 30) {
            // Phase 1: constant velocity
            px += vx * dt;
            py += vy * dt;
        } else if (k < 70) {
            // Phase 2: coordinated turn (omega = 0.05 rad/s)
            double omega = 0.05;
            double sw = std::sin(omega * dt), cw = std::cos(omega * dt);
            double vx_new = vx * cw - vy * sw;
            double vy_new = vx * sw + vy * cw;
            px += (vx * sw - vy * (1 - cw)) / omega;
            py += (vx * (1 - cw) + vy * sw) / omega;
            vx = vx_new; vy = vy_new;
        } else {
            // Phase 3: constant velocity (new heading)
            px += vx * dt;
            py += vy * dt;
        }

        meas[k] = {truth[k].x + noise(gen), truth[k].y + noise(gen)};
    }
}

// ============================================================
// Main: Demo
// ============================================================
int main() {
    const int N = 100;
    const double dt = 1.0;
    const double meas_noise = 10.0;

    std::vector<Point> truth, meas;
    generateManeuveringTrajectory(truth, meas, N, dt, meas_noise);

    // Trackers
    KalmanFilterCV kf(dt, 1.0, meas_noise);
    EKFCoordinatedTurn ekf(dt, 1.0, 0.1, meas_noise);
    IMMEstimator imm(dt, 1.0, 0.1, meas_noise);

    kf.initState(meas[0].x, meas[0].y);
    ekf.initState(meas[0].x, meas[0].y);
    imm.initState(meas[0].x, meas[0].y);

    double kf_sse = 0, ekf_sse = 0, imm_sse = 0;

    for (int k = 0; k < N; k++) {
        // KF
        kf.predict();
        kf.update(meas[k].x, meas[k].y);
        double kf_ex = kf.x[0] - truth[k].x, kf_ey = kf.x[2] - truth[k].y;
        kf_sse += kf_ex*kf_ex + kf_ey*kf_ey;

        // EKF
        ekf.predict();
        ekf.update(meas[k].x, meas[k].y);
        double ekf_ex = ekf.x[0] - truth[k].x, ekf_ey = ekf.x[2] - truth[k].y;
        ekf_sse += ekf_ex*ekf_ex + ekf_ey*ekf_ey;

        // IMM
        double imm_px, imm_py;
        imm.step(meas[k].x, meas[k].y, &imm_px, &imm_py);
        double imm_ex = imm_px - truth[k].x, imm_ey = imm_py - truth[k].y;
        imm_sse += imm_ex*imm_ex + imm_ey*imm_ey;
    }

    double kf_rmse  = std::sqrt(kf_sse / N);
    double ekf_rmse = std::sqrt(ekf_sse / N);
    double imm_rmse = std::sqrt(imm_sse / N);

    std::cout << std::string(60, '=') << "\n";
    std::cout << "Maneuvering Target Tracking - Results\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  KF  (Constant Velocity)  RMSE: " << kf_rmse  << " m\n";
    std::cout << "  EKF (Coordinated Turn)   RMSE: " << ekf_rmse << " m\n";
    std::cout << "  IMM (CV + CT)            RMSE: " << imm_rmse << " m\n";
    std::cout << std::string(60, '=') << "\n\n";
    std::cout << "IMM model probabilities at end:\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  P(CV) = " << imm.mu[0] << ",  P(CT) = " << imm.mu[1] << "\n";

    return 0;
}
