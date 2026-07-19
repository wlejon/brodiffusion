#pragma once
//
// terrain-diffusion EDM2 "magnitude-preserving" UNet (EDMUnet2D).
//
// Port of xandergos/terrain-diffusion (MIT) —
// terrain_diffusion/models/{edm_unet.py, unet_block.py, mp_layers.py}. The
// pipeline runs three of these nets, all the same architecture with different
// configs: `coarse` (16x16 single-level, 5 scalar conditioners), `base`
// (4-level, one 58-dim tensor conditioner, midblock attention) and `decoder`
// (4-level, unconditional besides the noise level).
//
// EDM2's magnitude-preserving idea: no normalisation layers with learned
// affines and no biases anywhere — instead every combining operation is
// rescaled so unit-variance activations stay unit-variance. That means
//   * mp_silu(x) = silu(x)/0.596 (the constant that restores unit variance),
//   * mp_sum / mp_concat divide by the L2 norm of their blend weights,
//   * MPConv normalises its weight to unit RMS at inference,
//   * a constant all-ones channel is appended to the input to stand in for the
//     bias the network is otherwise denied.
//
// WEIGHTS ARE PRE-FOLDED. scripts/convert-terrain-diffusion.py has already
// applied MPConv's inference-time weight transform (the normalise + 1/sqrt(fan)
// scale) and folded the scalar `out_gain` / per-block `emb_gain` parameters
// into out_conv.weight and emb_linear.weight. So every MPConv here is a PLAIN
// conv2d (or a plain matmul for the kernel=[] linear form), there are no bias,
// gain or scale tensors in the checkpoint, and this class applies no weight
// transform at load or at run time.
//
// The module *construction order* below mirrors the Python dict insertion order
// exactly, because that order is what pairs decoder blocks with encoder skips
// (skips are popped from the back). The safetensors keys are those dict keys,
// e.g. "enc.16x16_conv.weight", "dec.64x64_in0.attn_qkv.weight".

#include "brotensor/tensor.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::terrain {

// Mirrors the `models.<stage>` object of a converted checkpoint's config.json.
// The converter has already resolved every diffusers None-default into a
// concrete number, so nothing here re-derives a default.
struct MPUNetConfig {
    struct CondInput {
        std::string kind;      // "float" (MPFourier -> MPConv) or "tensor" (bare MPConv)
        int         dim = 0;   // fourier channels for "float", vector width for "tensor"
        float       weight = 1.0f;
    };

    int image_size     = 0;
    int in_channels    = 0;
    int out_channels   = 0;
    int model_channels = 0;
    std::vector<int> model_channel_mults;
    std::vector<int> layers_per_block;   // one entry per level
    int emb_channels   = 0;
    int noise_emb_dims = 0;
    std::vector<int> attn_resolutions;
    bool  midblock_attention = true;
    float concat_balance     = 0.5f;
    std::vector<CondInput> conditional_inputs;
    // conditional_weights[0] is the noise embedding's fixed weight of 1.0;
    // entries 1.. line up with conditional_inputs.
    std::vector<float> conditional_weights;
    std::string fourier_scale = "pos";   // "pos" => MPPositionalEmbedding for the noise level
    float res_balance     = 0.3f;
    float attn_balance    = 0.3f;
    float clip_act        = 256.0f;
    bool  has_clip_act    = true;
    int   channels_per_head = 64;

    // Parse `models.<stage>` out of a converted checkpoint's config.json.
    static MPUNetConfig from_config_json(const std::string& config_path,
                                         const std::string& stage);
};

class MPUNet {
public:
    explicit MPUNet(const MPUNetConfig& cfg);
    ~MPUNet();

    MPUNet(const MPUNet&) = delete;
    MPUNet& operator=(const MPUNet&) = delete;

    const MPUNetConfig& config() const { return cfg_; }

    // Load one stage's safetensors (coarse.safetensors / base.safetensors /
    // decoder.safetensors). Throws on any missing tensor.
    void load_weights(const brotensor::safetensors::File& f);

    // Run the denoiser.
    //   x:             (N, in_channels*S*S) NCHW device tensor at compute dtype.
    //   N, S:          batch and spatial size. S need not equal image_size —
    //                  the module *names* come from image_size, the activations
    //                  follow S.
    //   noise_labels:  host, N values (t on the TrigFlow arc).
    //   cond:          one host vector per entry of cfg.conditional_inputs. A
    //                  "float" input carries N values; a "tensor" input carries
    //                  N*dim values, row-major.
    //   out:           (N, out_channels*S*S). Caller syncs before reading.
    void forward(const brotensor::Tensor& x,
                 int N, int S,
                 const float* noise_labels,
                 const std::vector<std::vector<float>>& cond,
                 brotensor::Tensor& out);

private:
    enum class Resample { Keep, Down, Up };

    // One entry of the enc/dec ModuleDict. `plain_conv` marks the encoder stem,
    // which is a bare MPConv rather than a UNetBlock (its weight lives in
    // conv_res0 and nothing else is populated).
    struct Block {
        std::string key;          // e.g. "enc.16x16_block0"
        int  in_ch  = 0;
        int  out_ch = 0;
        bool plain_conv = false;
        bool dec_mode   = false;  // false = 'enc', true = 'dec'
        Resample resample = Resample::Keep;
        int  num_heads = 0;       // 0 = no attention

        brotensor::Tensor conv_res0, conv_res1, conv_skip, emb_linear;
        brotensor::Tensor attn_qkv, attn_proj;
        bool has_skip = false, has_emb = false;
    };

    // Conditioning branch. `freqs`/`phases` are the RANDOM buffers baked at
    // training init (MPFourier) or the fixed log-spaced ones (MPPositionalEmbedding);
    // both are read from the checkpoint, never recomputed.
    struct CondLayer {
        bool is_tensor = false;             // "tensor" kind: bare MPConv, mp_silu applied
        std::vector<float> freqs, phases;   // host; empty for the "tensor" kind
        brotensor::Tensor  W;               // (emb_channels, dim)
    };

    void build_modules_();
    void run_block_(const Block& b, brotensor::Tensor& x,
                    const brotensor::Tensor& emb, bool have_emb,
                    int N, int& C, int& S);
    void attention_(const Block& b, brotensor::Tensor& x, int N, int C, int S);
    const brotensor::Tensor& ones_filter_(int C);

    MPUNetConfig cfg_;
    std::vector<Block> enc_, dec_;
    brotensor::Tensor  out_conv_;

    std::vector<float> noise_freqs_;      // MPPositionalEmbedding / MPFourier buffer
    std::vector<float> noise_phases_;     // empty when fourier_scale == "pos"
    brotensor::Tensor  noise_linear_;     // (emb_channels, noise_emb_dims)
    bool has_noise_ = false;
    std::vector<CondLayer> cond_layers_;

    // Depthwise all-ones 1x1 filters for the `down` resample, keyed by channel
    // count (upstream's stride-2 subsample is exactly this convolution).
    std::unordered_map<int, brotensor::Tensor> ones_filters_;
};

}  // namespace brodiffusion::terrain
