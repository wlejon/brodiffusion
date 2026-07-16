#include "brodiffusion/ardy/motion_rep.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace brodiffusion::ardy {

// ── G1 skeleton data ─────────────────────────────────────────────────────
// Neutral joint offsets (float64) from ardy/assets/skeletons/g1skel34/joints.p.
// This is fixed model-architecture geometry (the G1 rest pose), not trained
// weights — baked in so motion decode is self-contained.
namespace {

constexpr int J = G1Skeleton::kNumJoints;  // 34

const double kNeutralJoints[J][3] = {
    {0.0, 0.0, 0.0},                                                             // 0  pelvis
    {0.064452, -0.10270000000000001, -3.7115199802428826e-17},                  // 1  left_hip_pitch
    {0.116452, -0.13316499999999998, -5.5426108147571535e-17},                  // 2  left_hip_roll
    {0.116452, -0.257285, 0.025000999999999916},                                // 3  left_hip_yaw
    {0.11860089999999998, -0.434625, -0.05327200000000012},                     // 4  left_knee
    {0.11850645499999997, -0.734635, -0.05327200000000018},                     // 5  left_ankle_pitch
    {0.11850645499999997, -0.752193, -0.05327200000000018},                     // 6  left_ankle_roll
    {0.11850645499999997, -0.787193, 0.0867279999999998},                       // 7  left_toe_base
    {-0.064452, -0.10270000000000001, -8.49276204917261e-18},                   // 8  right_hip_pitch
    {-0.116452, -0.13316499999999998, -3.711031482112068e-18},                  // 9  right_hip_roll
    {-0.116452, -0.257285, 0.025000999999999968},                               // 10 right_hip_yaw
    {-0.11860090000000001, -0.434625, -0.05327200000000006},                    // 11 right_knee
    {-0.11850645500000001, -0.734635, -0.05327200000000013},                    // 12 right_ankle_pitch
    {-0.11850645500000001, -0.752193, -0.05327200000000013},                    // 13 right_ankle_roll
    {-0.11850645500000001, -0.787193, 0.08672799999999986},                     // 14 right_toe_base
    {0.0, 0.0, 0.0},                                                             // 15 waist_yaw
    {-8.800737916203616e-19, 0.04400000000000004, -0.00396349999999999},        // 16 waist_roll
    {-8.800737916203616e-19, 0.04400000000000004, -0.00396349999999999},        // 17 waist_pitch
    {0.10022, 0.29178000000000004, -7.199999999956797e-06},                     // 18 left_shoulder_pitch
    {0.13821999999999998, 0.2779490000000001, -7.199999999968305e-06},          // 19 left_shoulder_roll
    {0.14446, 0.17474900000000004, -7.199999999992606e-06},                     // 20 left_shoulder_yaw
    {0.14446, 0.09423099999999995, 0.01577579999999999},                        // 21 left_elbow
    {0.14634791000000003, 0.08423100000000006, 0.1157758},                      // 22 left_wrist_roll
    {0.14634791000000003, 0.08423100000000006, 0.1537758},                      // 23 left_wrist_pitch
    {0.14634791000000003, 0.08423100000000006, 0.1997758},                      // 24 left_wrist_yaw
    {0.14634791000000003, 0.08423100000000006, 0.29977580000000004},            // 25 left_hand_roll
    {-0.10021, 0.29178000000000004, -7.199999999912388e-06},                    // 26 right_shoulder_pitch
    {-0.13820999999999997, 0.2779490000000001, -7.199999999907021e-06},         // 27 right_shoulder_roll
    {-0.14445, 0.17474900000000004, -7.19999999992855e-06},                     // 28 right_shoulder_yaw
    {-0.14445, 0.09423099999999995, 0.01577580000000005},                       // 29 right_elbow
    {-0.14633790999999996, 0.08423100000000006, 0.11577580000000004},           // 30 right_wrist_roll
    {-0.14633790999999996, 0.08423100000000006, 0.15377580000000007},           // 31 right_wrist_pitch
    {-0.14633790999999996, 0.08423100000000006, 0.1997758000000001},            // 32 right_wrist_yaw
    {-0.14633790999999996, 0.08423100000000006, 0.2997758000000001},            // 33 right_hand_roll
};

// Parent index per joint (derived from bone_order_names_with_parents), root -1.
const int kJointParents[J] = {
    -1,  0,  1,  2,  3,  4,  5,  6,   // pelvis, left leg chain + toe
     0,  8,  9, 10, 11, 12, 13,       // right leg chain + toe
     0, 15, 16,                        // waist yaw/roll/pitch
    17, 18, 19, 20, 21, 22, 23, 24,   // left arm chain + hand
    17, 26, 27, 28, 29, 30, 31, 32,   // right arm chain + hand
};

// ── tiny double-precision 3x3 / rigid helpers (row-major 3x3) ──────────────

inline void mat3_mul(const double* a, const double* b, double* out) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c] +
                             a[r * 3 + 1] * b[1 * 3 + c] +
                             a[r * 3 + 2] * b[2 * 3 + c];
}

// out = A^T @ B
inline void mat3_Tmul(const double* a, const double* b, double* out) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out[r * 3 + c] = a[0 * 3 + r] * b[0 * 3 + c] +
                             a[1 * 3 + r] * b[1 * 3 + c] +
                             a[2 * 3 + r] * b[2 * 3 + c];
}

// v_out = M @ v (3-vec)
inline void mat3_vec(const double* m, const double* v, double* out) {
    out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

inline void vec3_cross(const double* a, const double* b, double* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

inline double vec3_norm(const double* v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

}  // namespace

const double* G1Skeleton::neutral_joints() { return &kNeutralJoints[0][0]; }
const int* G1Skeleton::joint_parents() { return kJointParents; }

// ── cont6d <-> matrix ──────────────────────────────────────────────────────
// cont6d = [column0, column1]; rebuild via Gram-Schmidt exactly as
// ardy/geometry.py cont6d_to_matrix.
void cont6d_to_matrix(const double* c6, double* m) {
    const double* x_raw = c6;      // 3
    const double* y_raw = c6 + 3;  // 3
    double x[3], y[3], z[3];

    double nx = vec3_norm(x_raw);
    for (int i = 0; i < 3; ++i) x[i] = x_raw[i] / nx;

    vec3_cross(x, y_raw, z);
    double nz = vec3_norm(z);
    for (int i = 0; i < 3; ++i) z[i] /= nz;

    vec3_cross(z, x, y);

    // columns [x, y, z] -> row-major m[r*3 + c]
    for (int r = 0; r < 3; ++r) {
        m[r * 3 + 0] = x[r];
        m[r * 3 + 1] = y[r];
        m[r * 3 + 2] = z[r];
    }
}

void matrix_to_cont6d(const double* m, double* c6) {
    // first two columns
    for (int r = 0; r < 3; ++r) {
        c6[r]     = m[r * 3 + 0];
        c6[3 + r] = m[r * 3 + 1];
    }
}

// ── forward kinematics ──────────────────────────────────────────────────────
void fk(const double* local_rot_mats, const double* root_positions, int T,
        std::vector<double>& global_rot_mats,
        std::vector<double>& posed_joints,
        std::vector<double>& posed_norootpos) {
    const int* parents = kJointParents;
    global_rot_mats.assign(static_cast<size_t>(T) * J * 9, 0.0);
    posed_joints.assign(static_cast<size_t>(T) * J * 3, 0.0);
    posed_norootpos.assign(static_cast<size_t>(T) * J * 3, 0.0);

    for (int t = 0; t < T; ++t) {
        const double* R = local_rot_mats + static_cast<size_t>(t) * J * 9;
        const double* root = root_positions + static_cast<size_t>(t) * 3;
        double* G = global_rot_mats.data() + static_cast<size_t>(t) * J * 9;
        double* P = posed_norootpos.data() + static_cast<size_t>(t) * J * 3;

        // Root: transform = [R0 | rel0], rel0 = neutral[0] - neutral[0] = 0.
        for (int k = 0; k < 9; ++k) G[k] = R[k];
        P[0] = 0.0; P[1] = 0.0; P[2] = 0.0;  // pelvis offset removed (G1 root == 0)

        // Joints are topologically sorted, so a single forward pass suffices.
        for (int j = 1; j < J; ++j) {
            const int p = parents[j];
            const double* Rj = R + j * 9;
            // rel = neutral[j] - neutral[parent]
            double rel[3] = {
                kNeutralJoints[j][0] - kNeutralJoints[p][0],
                kNeutralJoints[j][1] - kNeutralJoints[p][1],
                kNeutralJoints[j][2] - kNeutralJoints[p][2],
            };
            // Compose rigid transforms: G[j] = G[p] @ Rj ; t[j] = G[p]@rel + t[p]
            double* Gj = G + j * 9;
            const double* Gp = G + p * 9;
            mat3_mul(Gp, Rj, Gj);
            double rot_rel[3];
            mat3_vec(Gp, rel, rot_rel);
            P[j * 3 + 0] = rot_rel[0] + P[p * 3 + 0];
            P[j * 3 + 1] = rot_rel[1] + P[p * 3 + 1];
            P[j * 3 + 2] = rot_rel[2] + P[p * 3 + 2];
        }

        // posed = posed_norootpos + root
        double* PJ = posed_joints.data() + static_cast<size_t>(t) * J * 3;
        for (int j = 0; j < J; ++j) {
            PJ[j * 3 + 0] = P[j * 3 + 0] + root[0];
            PJ[j * 3 + 1] = P[j * 3 + 1] + root[1];
            PJ[j * 3 + 2] = P[j * 3 + 2] + root[2];
        }
    }
}

void global_rots_to_local_rots(const double* global_rot_mats, int T,
                               std::vector<double>& local_rot_mats) {
    const int* parents = kJointParents;
    local_rot_mats.assign(static_cast<size_t>(T) * J * 9, 0.0);
    static const double kEye[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    for (int t = 0; t < T; ++t) {
        const double* G = global_rot_mats + static_cast<size_t>(t) * J * 9;
        double* Lm = local_rot_mats.data() + static_cast<size_t>(t) * J * 9;
        for (int j = 0; j < J; ++j) {
            const double* parent = (j == G1Skeleton::kRootIdx)
                                       ? kEye
                                       : (G + parents[j] * 9);
            // local[j] = parent^T @ global[j]
            mat3_Tmul(parent, G + j * 9, Lm + j * 9);
        }
    }
}

// ── ArdyMotionRep ───────────────────────────────────────────────────────────
namespace {
// velocities: fps * (pos[t+1] - pos[t]); last frame repeats the previous.
void compute_vel_xyz(const std::vector<double>& posed_joints, int T, double fps,
                     std::vector<double>& vel) {
    vel.assign(static_cast<size_t>(T) * J * 3, 0.0);
    for (int t = 0; t < T - 1; ++t) {
        const double* p0 = posed_joints.data() + static_cast<size_t>(t) * J * 3;
        const double* p1 = posed_joints.data() + static_cast<size_t>(t + 1) * J * 3;
        double* v = vel.data() + static_cast<size_t>(t) * J * 3;
        for (int k = 0; k < J * 3; ++k) v[k] = fps * (p1[k] - p0[k]);
    }
    if (T >= 2) {
        // repeat last: vel[T-1] = vel[T-2]
        const double* src = vel.data() + static_cast<size_t>(T - 2) * J * 3;
        double* dst = vel.data() + static_cast<size_t>(T - 1) * J * 3;
        for (int k = 0; k < J * 3; ++k) dst[k] = src[k];
    }
}
}  // namespace

void ArdyMotionRep::forward(const double* local_rot_mats,
                            const double* root_positions, int T,
                            std::vector<double>& out) const {
    if (T < 2) throw std::runtime_error("ArdyMotionRep::forward needs T >= 2");

    std::vector<double> global_rot_mats, posed_joints, posed_norootpos;
    fk(local_rot_mats, root_positions, T, global_rot_mats, posed_joints,
       posed_norootpos);

    std::vector<double> velocities;
    compute_vel_xyz(posed_joints, T, fps_, velocities);

    out.assign(static_cast<size_t>(T) * kFeatureDim, 0.0);

    for (int t = 0; t < T; ++t) {
        double* f = out.data() + static_cast<size_t>(t) * kFeatureDim;
        const double* root = root_positions + static_cast<size_t>(t) * 3;
        const double* PJ = posed_joints.data() + static_cast<size_t>(t) * J * 3;
        const double* PN = posed_norootpos.data() + static_cast<size_t>(t) * J * 3;
        const double* G = global_rot_mats.data() + static_cast<size_t>(t) * J * 9;
        const double* V = velocities.data() + static_cast<size_t>(t) * J * 3;

        int off = 0;
        // root_pos [3]
        f[off++] = root[0]; f[off++] = root[1]; f[off++] = root[2];

        // global_root_heading [2] — from hip vector (right - left)
        const double* rh = PJ + G1Skeleton::kHipRight * 3;
        const double* lh = PJ + G1Skeleton::kHipLeft * 3;
        double dx = rh[0] - lh[0], dz = rh[2] - lh[2];
        double heading = std::atan2(dz, -dx);
        f[off++] = std::cos(heading);
        f[off++] = std::sin(heading);

        // local_joints_positions [(J-1)*3] — non-root, pelvis-relative + root.y
        for (int j = 1; j < J; ++j) {
            f[off++] = PN[j * 3 + 0];
            f[off++] = PN[j * 3 + 1] + root[1];  // ground offset = root height
            f[off++] = PN[j * 3 + 2];
        }

        // global_rot_data [J*6]
        for (int j = 0; j < J; ++j) {
            matrix_to_cont6d(G + j * 9, f + off);
            off += 6;
        }

        // velocities [J*3]
        for (int k = 0; k < J * 3; ++k) f[off++] = V[k];

        // foot_contacts [4] — L{6,7}, R{13,14}: |vel| < 0.15 & height < 0.10
        const int feet[4] = {G1Skeleton::kLeftFoot[0], G1Skeleton::kLeftFoot[1],
                             G1Skeleton::kRightFoot[0], G1Skeleton::kRightFoot[1]};
        for (int fi = 0; fi < 4; ++fi) {
            const int jf = feet[fi];
            const double* v = V + jf * 3;
            double vmag = vec3_norm(v);
            double h = PJ[jf * 3 + 1];
            f[off++] = (vmag < kFootVelThresh && h < kFootHeightThresh) ? 1.0 : 0.0;
        }
    }
}

ArdyMotionRep::Decoded ArdyMotionRep::inverse(const double* features, int T,
                                              bool posed_from_rot) const {
    Decoded d;
    d.T = T;

    // Slice offsets within a 414-d feature row.
    constexpr int kRootPos   = 0;                         // 3
    constexpr int kHeading   = kRootPos + 3;              // 2
    constexpr int kLocalPos  = kHeading + 2;              // (J-1)*3 = 99
    constexpr int kGlobalRot = kLocalPos + (J - 1) * 3;   // J*6 = 204
    // velocities [J*3] and foot_contacts [4] follow but are only needed for
    // foot contacts here.
    constexpr int kFoot = kGlobalRot + J * 6 + J * 3;     // last 4

    // global_rot_data -> global rotation matrices
    std::vector<double> global_rot_mats(static_cast<size_t>(T) * J * 9);
    d.root_positions.assign(static_cast<size_t>(T) * 3, 0.0);
    d.global_root_heading.assign(static_cast<size_t>(T) * 2, 0.0);
    d.foot_contacts.assign(static_cast<size_t>(T) * 4, 0.0);

    for (int t = 0; t < T; ++t) {
        const double* f = features + static_cast<size_t>(t) * kFeatureDim;
        double* root = d.root_positions.data() + static_cast<size_t>(t) * 3;
        root[0] = f[kRootPos + 0];
        root[1] = f[kRootPos + 1];
        root[2] = f[kRootPos + 2];
        d.global_root_heading[t * 2 + 0] = f[kHeading + 0];
        d.global_root_heading[t * 2 + 1] = f[kHeading + 1];

        double* G = global_rot_mats.data() + static_cast<size_t>(t) * J * 9;
        for (int j = 0; j < J; ++j)
            cont6d_to_matrix(f + kGlobalRot + j * 6, G + j * 9);

        for (int fi = 0; fi < 4; ++fi)
            d.foot_contacts[t * 4 + fi] = (f[kFoot + fi] > 0.5) ? 1.0 : 0.0;
    }

    global_rots_to_local_rots(global_rot_mats.data(), T, d.local_rot_mats);
    d.global_rot_mats = std::move(global_rot_mats);

    if (posed_from_rot) {
        std::vector<double> gr, pn;  // FK re-derives global rots identically
        fk(d.local_rot_mats.data(), d.root_positions.data(), T, gr,
           d.posed_joints, pn);
    } else {
        // Lift stored local_joints_positions: dummy root joint + planar root.
        d.posed_joints.assign(static_cast<size_t>(T) * J * 3, 0.0);
        for (int t = 0; t < T; ++t) {
            const double* f = features + static_cast<size_t>(t) * kFeatureDim;
            const double* root = d.root_positions.data() + static_cast<size_t>(t) * 3;
            double* PJ = d.posed_joints.data() + static_cast<size_t>(t) * J * 3;
            PJ[0] = root[0]; PJ[1] = 0.0; PJ[2] = root[2];  // dummy root
            for (int j = 1; j < J; ++j) {
                PJ[j * 3 + 0] = f[kLocalPos + (j - 1) * 3 + 0] + root[0];
                PJ[j * 3 + 1] = f[kLocalPos + (j - 1) * 3 + 1];
                PJ[j * 3 + 2] = f[kLocalPos + (j - 1) * 3 + 2] + root[2];
            }
        }
    }

    return d;
}

}  // namespace brodiffusion::ardy
