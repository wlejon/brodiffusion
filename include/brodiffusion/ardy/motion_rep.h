#pragma once
//
// ARDY motion representation — G1 skeleton + forward kinematics + the
// "global root / global joint" feature codec that ARDY's diffusion denoiser
// operates on.
//
// ARDY (nvidia/ARDY-G1-RP) denoises motion in a feature space, then decodes
// those features back to joint rotations/positions for a Unitree G1 humanoid.
// This module is that codec plus the kinematics it rests on — pure host math,
// no network weights. It is the deterministic geometric floor the rest of the
// port (FSQ autoencoder, two-stage denoiser, sampler, rollout) sits on top of.
//
// Reference: ardy/motion_rep/reps/ardy_motionrep.py (ArdyMotionRep),
// ardy/skeleton/{definitions,base,kinematics,transforms}.py, ardy/geometry.py.
//
// Feature layout for the 34-joint G1 skeleton (motion_rep_dim = 414):
//   root_pos              [3]            world root translation (x, y, z)
//   global_root_heading   [2]            (cos theta, sin theta) from the hips
//   local_joints_positions[(J-1)*3 = 99] non-root joints, pelvis-relative,
//                                        lifted by the root height
//   global_rot_data       [J*6   = 204]  global joint rotations, 6D (cont6d)
//   velocities            [J*3   = 102]  global joint velocities (fps * dpos)
//   foot_contacts         [4]            L-heel, L-toe, R-heel, R-toe (0/1)
//
// All math is double precision to stay bit-close to the float64 PyTorch
// reference; the production FP16/FP32 path lives in the denoiser, not here.

#include <vector>

namespace brodiffusion::ardy {

// Unitree G1 skeleton as ARDY defines it: 32 articulated joints + 2 toe
// endpoints, root (pelvis) at index 0. Neutral joint offsets are baked in from
// ardy/assets/skeletons/g1skel34/joints.p (model architecture data, not trained
// weights), so decoding a motion needs no external asset file.
struct G1Skeleton {
    static constexpr int kNumJoints = 34;
    static constexpr int kRootIdx   = 0;

    // neutral_joints()[j*3 + {0,1,2}] — per-joint rest offset in the parent-less
    // neutral pose (root is the origin).
    static const double* neutral_joints();
    // joint_parents()[j] — parent joint index, -1 for the root. Guaranteed
    // topologically sorted (parent index < child index) so FK is a single pass.
    static const int* joint_parents();

    // Semantic joint groups used by heading + foot-contact detection.
    // Hips are ordered [right, left] to match compute_heading_angle.
    static constexpr int kHipRight = 8;   // right_hip_pitch_skel
    static constexpr int kHipLeft  = 1;   // left_hip_pitch_skel
    // Foot chains: left {ankle_roll, toe}, right {ankle_roll, toe}.
    static constexpr int kLeftFoot[2]  = {6, 7};
    static constexpr int kRightFoot[2] = {13, 14};
};

// Forward kinematics for the G1 hierarchy.
//   local_rot_mats:   (T, J, 3, 3) row-major local joint rotation matrices
//   root_positions:   (T, 3) world root translation
// Outputs (each caller-sized to the noted length, resized if needed):
//   global_rot_mats:  (T, J, 3, 3) global joint rotations
//   posed_joints:     (T, J, 3) world joint positions (root translation added)
//   posed_norootpos:  (T, J, 3) joint positions before adding root translation
// root_positions are treated as global (the pelvis offset — zero for G1 — is
// removed from the neutral pose, matching root_positions_is_global=True).
void fk(const double* local_rot_mats, const double* root_positions, int T,
        std::vector<double>& global_rot_mats,
        std::vector<double>& posed_joints,
        std::vector<double>& posed_norootpos);

// 6D rotation <-> matrix (Zhou et al. cont6d), matching ardy/geometry.py.
//   cont6d layout = [column0(3), column1(3)] of the rotation matrix.
void cont6d_to_matrix(const double* c6, double* m3x3);   // 6 -> 9 (row-major)
void matrix_to_cont6d(const double* m3x3, double* c6);   // 9 -> 6

// Convert global joint rotations back to local rotations for the G1 hierarchy:
// local[j] = global[parent[j]]^T @ global[j], with the root's parent = I.
//   global_rot_mats / local_rot_mats: (T, J, 3, 3) row-major.
void global_rots_to_local_rots(const double* global_rot_mats, int T,
                               std::vector<double>& local_rot_mats);

// The ARDY motion-representation codec for the G1 skeleton.
class ArdyMotionRep {
public:
    static constexpr int kNumJoints  = G1Skeleton::kNumJoints;   // 34
    static constexpr int kFeatureDim = 414;                      // motion_rep_dim

    explicit ArdyMotionRep(double fps = 25.0) : fps_(fps) {}

    double fps() const { return fps_; }

    // Motion normalization stats for inverse(is_normalized=true). ARDY bundles
    // the motion stats as [global_root 5, local_root 4, body 409] = 418 entries,
    // but the explicit 414-dim feature is [global_root 5, body 409]; so
    // unnormalization uses the stats sliced to indices [0:5] ++ [9:418] (exactly
    // all_stats.sliced(...) in the ardy motionrep base). Pass the full 418-entry
    // mean/std; eps matches the ardy Stats default (1e-5). Without stats set,
    // inverse(is_normalized=true) throws.
    void set_motion_stats(const float* mean, const float* std, int n_full,
                          float eps = 1e-5f);
    bool has_motion_stats() const { return !feat_mean_.empty(); }

    // Encode local rotations + root positions into ARDY features (unnormalized).
    //   local_rot_mats: (T, J, 3, 3), root_positions: (T, 3)
    //   out_features:   (T, kFeatureDim), resized if needed.
    // Needs T >= 2 (velocities are finite differences).
    void forward(const double* local_rot_mats, const double* root_positions,
                 int T, std::vector<double>& out_features) const;

    // Decode ARDY features (unnormalized) back into motion.
    struct Decoded {
        std::vector<double> local_rot_mats;       // (T, J, 3, 3)
        std::vector<double> global_rot_mats;      // (T, J, 3, 3)
        std::vector<double> posed_joints;         // (T, J, 3)
        std::vector<double> root_positions;       // (T, 3)
        std::vector<double> foot_contacts;        // (T, 4) — 0/1 (contact > 0.5)
        std::vector<double> global_root_heading;  // (T, 2)
        int T = 0;
    };
    // is_normalized: the features are in normalized space (the detokenize /
    // hybrid output) and must be unnormalized with the motion stats first,
    // matching motion_rep.inverse(motion, is_normalized=True). Requires
    // set_motion_stats(). posed_joints_from_rotations: re-run FK from the decoded
    // rotations (true, the ARDY default) vs. lift the stored
    // local_joints_positions (false).
    Decoded inverse(const double* features, int T,
                    bool is_normalized = false,
                    bool posed_joints_from_rotations = true) const;

    // Foot-contact heuristic thresholds (ArdyMotionRep uses 0.15 / 0.10).
    static constexpr double kFootVelThresh    = 0.15;
    static constexpr double kFootHeightThresh = 0.10;

private:
    double fps_;
    // 414-dim sliced motion stats (double), eps-folded denom. Empty until set.
    std::vector<double> feat_mean_;    // [kFeatureDim]
    std::vector<double> feat_stdeps_;  // [kFeatureDim] = sqrt(std^2 + eps)
};

}  // namespace brodiffusion::ardy
