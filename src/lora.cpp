#include "brodiffusion/lora.h"

#include "brodiffusion/safetensors.h"

#include "brotensor/tensor.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brodiffusion::lora {

namespace st = ::brodiffusion::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("lora: " + msg);
}

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}
bool ends_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() &&
           s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// Strip a recognized LoRA suffix (".lora_down.weight" etc.) and return the
// kind: 0 = none, 1 = down, 2 = up, 3 = alpha. `out_prefix` receives the part
// before the suffix.
int strip_suffix(const std::string& key, std::string& out_prefix) {
    static const char* down_sfxs[] = {
        ".lora_down.weight",
        ".lora_A.weight",
        ".lora_A.default.weight",
    };
    static const char* up_sfxs[] = {
        ".lora_up.weight",
        ".lora_B.weight",
        ".lora_B.default.weight",
    };
    for (const char* s : down_sfxs) {
        if (ends_with(key, s)) {
            out_prefix = key.substr(0, key.size() - std::char_traits<char>::length(s));
            return 1;
        }
    }
    for (const char* s : up_sfxs) {
        if (ends_with(key, s)) {
            out_prefix = key.substr(0, key.size() - std::char_traits<char>::length(s));
            return 2;
        }
    }
    if (ends_with(key, ".alpha")) {
        out_prefix = key.substr(0, key.size() - 6);
        return 3;
    }
    return 0;
}

// Parse a non-negative decimal integer starting at `pos` in `s`. On success
// updates `pos` past the digits and returns true.
bool parse_uint(const std::string& s, std::size_t& pos, int& out) {
    if (pos >= s.size() || !std::isdigit(static_cast<unsigned char>(s[pos]))) return false;
    int v = 0;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
        v = v * 10 + (s[pos] - '0');
        ++pos;
    }
    out = v;
    return true;
}

// Tails that live INSIDE a transformer_blocks.0 scope (attention + FF).
std::string match_block_tail_kohya(const std::string& tail) {
    static const std::pair<const char*, const char*> map[] = {
        {"attn1_to_q",       "attn1.to_q"},
        {"attn1_to_k",       "attn1.to_k"},
        {"attn1_to_v",       "attn1.to_v"},
        {"attn1_to_out_0",   "attn1.to_out.0"},
        {"attn2_to_q",       "attn2.to_q"},
        {"attn2_to_k",       "attn2.to_k"},
        {"attn2_to_v",       "attn2.to_v"},
        {"attn2_to_out_0",   "attn2.to_out.0"},
        {"ff_net_0_proj",    "ff.net.0.proj"},
        {"ff_net_2",         "ff.net.2"},
    };
    for (const auto& kv : map) {
        if (tail == kv.first) return kv.second;
    }
    return {};
}

// Tails that live directly under an attentions.<j> Transformer2D scope (the
// 1x1 conv wrappers around the BasicTransformerBlock).
std::string match_attn_wrapper_tail_kohya(const std::string& tail) {
    if (tail == "proj_in")  return "proj_in";
    if (tail == "proj_out") return "proj_out";
    return {};
}

// Tails inside a resnets.<k> scope.
std::string match_resnet_tail_kohya(const std::string& tail) {
    if (tail == "conv1")          return "conv1";
    if (tail == "conv2")          return "conv2";
    if (tail == "conv_shortcut")  return "conv_shortcut";
    if (tail == "time_emb_proj")  return "time_emb_proj";
    return {};
}

const char* match_clip_proj_kohya(const std::string& proj) {
    if (proj == "q_proj"   || proj == "k_proj" ||
        proj == "v_proj"   || proj == "out_proj") {
        return proj.c_str();  // same name in diffusers
    }
    return nullptr;
}

// After the kohya block prefix (e.g. "lora_unet_down_blocks_3_") parse one of:
//   attentions_<j>_transformer_blocks_0_<tail>     -> .attentions.<j>.transformer_blocks.0.<tail>
//   attentions_<j>_(proj_in|proj_out)              -> .attentions.<j>.<proj>
//   resnets_<k>_(conv1|conv2|conv_shortcut|time_emb_proj)
//   (downsamplers|upsamplers)_0_conv
// Returns the diffusers tail (everything after the block prefix in dot form)
// or empty if no recognized pattern. `down_or_up` is "down" or "up"; only
// used to distinguish downsamplers vs upsamplers.
std::string parse_kohya_block_sub(const std::string& rest, std::size_t pos,
                                  const char* down_or_up) {
    // attentions_<j>_...
    static const std::string atts = "attentions_";
    if (rest.compare(pos, atts.size(), atts) == 0) {
        std::size_t p = pos + atts.size();
        int j = 0;
        if (!parse_uint(rest, p, j)) return {};
        if (p >= rest.size() || rest[p] != '_') return {};
        ++p;
        // transformer_blocks_0_<tail>
        static const std::string tb = "transformer_blocks_0_";
        if (rest.compare(p, tb.size(), tb) == 0) {
            std::string tail = rest.substr(p + tb.size());
            std::string dtail = match_block_tail_kohya(tail);
            if (dtail.empty()) return {};
            return "attentions." + std::to_string(j) +
                   ".transformer_blocks.0." + dtail;
        }
        // proj_in / proj_out
        std::string tail = rest.substr(p);
        std::string dtail = match_attn_wrapper_tail_kohya(tail);
        if (!dtail.empty()) {
            return "attentions." + std::to_string(j) + "." + dtail;
        }
        return {};
    }
    // resnets_<k>_<tail>
    static const std::string rs = "resnets_";
    if (rest.compare(pos, rs.size(), rs) == 0) {
        std::size_t p = pos + rs.size();
        int k = 0;
        if (!parse_uint(rest, p, k)) return {};
        if (p >= rest.size() || rest[p] != '_') return {};
        ++p;
        std::string tail = rest.substr(p);
        std::string dtail = match_resnet_tail_kohya(tail);
        if (dtail.empty()) return {};
        return "resnets." + std::to_string(k) + "." + dtail;
    }
    // (downsamplers|upsamplers)_0_conv
    const std::string sampler = std::string(down_or_up) + "samplers_0_conv";
    if (rest.compare(pos, sampler.size(), sampler) == 0 &&
        pos + sampler.size() == rest.size()) {
        return std::string(down_or_up) + "samplers.0.conv";
    }
    return {};
}

// Parse a kohya prefix (everything before .lora_down.weight etc.) into a
// (domain, target_path) pair. Returns false if not a recognized SD1.5 target.
bool parse_kohya_prefix(const std::string& p,
                        std::string& domain,
                        std::string& target_path) {
    if (starts_with(p, "lora_unet_")) {
        std::string rest = p.substr(10);
        for (const char* side : {"down", "up"}) {
            std::string head = std::string(side) + "_blocks_";
            if (starts_with(rest, head)) {
                std::size_t pos = head.size();
                int i = 0;
                if (!parse_uint(rest, pos, i)) return false;
                if (pos >= rest.size() || rest[pos] != '_') return false;
                ++pos;
                std::string sub = parse_kohya_block_sub(rest, pos, side);
                if (sub.empty()) return false;
                domain = "unet";
                target_path = std::string(side) + "_blocks." +
                              std::to_string(i) + "." + sub;
                return true;
            }
        }
        // mid_block_(attentions_0_... | resnets_<k>_...).
        static const std::string mid = "mid_block_";
        if (starts_with(rest, mid)) {
            std::size_t pos = mid.size();
            // mid_block has no sampler — pass "down" as a no-op marker.
            std::string sub = parse_kohya_block_sub(rest, pos, "down");
            if (sub.empty()) return false;
            // Reject the spurious sampler match (mid block has none).
            if (sub.find("samplers") != std::string::npos) return false;
            domain = "unet";
            target_path = "mid_block." + sub;
            return true;
        }
        return false;
    }
    if (starts_with(p, "lora_te_")) {
        std::string rest = p.substr(8);
        static const std::string head = "text_model_encoder_layers_";
        if (!starts_with(rest, head)) return false;
        std::size_t pos = head.size();
        int i = 0;
        if (!parse_uint(rest, pos, i)) return false;
        static const std::string sa = "_self_attn_";
        if (rest.compare(pos, sa.size(), sa) != 0) return false;
        pos += sa.size();
        std::string proj = rest.substr(pos);
        const char* p_diff = match_clip_proj_kohya(proj);
        if (!p_diff) return false;
        domain = "text_encoder";
        target_path = "text_model.encoder.layers." + std::to_string(i) +
                      ".self_attn." + std::string(p_diff);
        return true;
    }
    return false;
}

// Parse a diffusers / PEFT prefix.
//
// Recognized layouts (with optional leading domain prefix):
//   unet.<unet_path>.<attn>.<proj>             [.processor]?
//   text_encoder.<clip_path>
//
// PEFT sometimes inserts ".processor" between the attention name and the
// projection, but only in xformers-style processor-key checkpoints. SD1.5
// LoRAs in the wild generally don't include it; skip the variant for now.
bool parse_diffusers_prefix(const std::string& p,
                            std::string& domain,
                            std::string& target_path) {
    auto strip_domain = [&](std::string_view d) -> std::string_view {
        if (starts_with(p, std::string(d) + ".")) {
            return std::string_view(p).substr(d.size() + 1);
        }
        return {};
    };
    std::string_view sv = strip_domain("unet");
    if (!sv.empty()) {
        // Accept any path whose suffix is a known UNet target tail. The set
        // is the union of: attention/FF tails inside transformer_blocks.0,
        // attentions.<j>.proj_(in|out), resnets.<k>.(conv1|conv2|conv_shortcut|
        // time_emb_proj), and (downsamplers|upsamplers).0.conv.
        static const char* known_tails[] = {
            ".attn1.to_q", ".attn1.to_k", ".attn1.to_v", ".attn1.to_out.0",
            ".attn2.to_q", ".attn2.to_k", ".attn2.to_v", ".attn2.to_out.0",
            ".ff.net.0.proj", ".ff.net.2",
            ".proj_in", ".proj_out",
            ".conv1", ".conv2", ".conv_shortcut", ".time_emb_proj",
            ".downsamplers.0.conv", ".upsamplers.0.conv",
        };
        for (const char* tail : known_tails) {
            std::string_view k(tail);
            if (sv.size() >= k.size() &&
                sv.compare(sv.size() - k.size(), k.size(), k) == 0) {
                domain = "unet";
                target_path = std::string(sv);
                return true;
            }
        }
        return false;
    }
    sv = strip_domain("text_encoder");
    if (!sv.empty()) {
        // CLIP target ends in q_proj/k_proj/v_proj/out_proj.
        for (const char* proj : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            std::string_view k(proj);
            if (sv.size() >= k.size() &&
                sv.compare(sv.size() - k.size(), k.size(), k) == 0) {
                domain = "text_encoder";
                target_path = std::string(sv);
                return true;
            }
        }
        return false;
    }
    return false;
}

// Heuristic: is this key bookkeeping we should silently skip? Things like
// rank metadata, dora scale, etc. — these aren't errors.
bool is_lora_bookkeeping(const std::string& key) {
    return key.find(".dora_scale") != std::string::npos ||
           key.find("__metadata__") != std::string::npos;
}

}  // namespace

// ─── public API ────────────────────────────────────────────────────────────

Format detect_format(const st::File& f) {
    int kohya = 0, diffusers = 0;
    for (const st::TensorView& tv : f.tensors()) {
        const std::string& k = tv.name;
        if (starts_with(k, "lora_unet_") || starts_with(k, "lora_te_")) ++kohya;
        else if (starts_with(k, "unet.") || starts_with(k, "text_encoder.")) ++diffusers;
    }
    if (kohya == 0 && diffusers == 0) {
        fail("file contains no recognizable LoRA keys (need 'lora_unet_'/'lora_te_' "
             "or 'unet.'/'text_encoder.' prefixes)");
    }
    return (kohya >= diffusers) ? Format::Kohya : Format::Diffusers;
}

std::vector<TargetPath> known_targets() {
    // Enumerate SD1.5 default layout: 4 down/up blocks, layers_per_block=2,
    // mid block. The full set is small enough (~80 paths) to inline.
    std::vector<TargetPath> out;
    static const char* tails[] = {
        "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
        "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
        "ff.net.0.proj", "ff.net.2",
    };
    const int nb = 4;
    const int lpb = 2;  // layers_per_block in SD1.5
    for (int i = 0; i < nb - 1; ++i) {
        for (int j = 0; j < lpb; ++j) {
            for (const char* t : tails) {
                out.push_back({"unet",
                    "down_blocks." + std::to_string(i) + ".attentions." +
                    std::to_string(j) + ".transformer_blocks.0." + t});
            }
        }
    }
    for (const char* t : tails) {
        out.push_back({"unet", std::string("mid_block.attentions.0.transformer_blocks.0.") + t});
    }
    // Up blocks: index 0 has no attention; indices 1..nb-1 have it.
    for (int i = 1; i < nb; ++i) {
        for (int j = 0; j < lpb + 1; ++j) {
            for (const char* t : tails) {
                out.push_back({"unet",
                    "up_blocks." + std::to_string(i) + ".attentions." +
                    std::to_string(j) + ".transformer_blocks.0." + t});
            }
        }
    }
    // CLIP: 12 layers in SD1.5's ViT-L/14.
    for (int i = 0; i < 12; ++i) {
        for (const char* p : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            out.push_back({"text_encoder",
                "text_model.encoder.layers." + std::to_string(i) + ".self_attn." + p});
        }
    }
    return out;
}

std::string diffusers_to_kohya_prefix(const std::string& domain,
                                      const std::string& target_path) {
    std::string head;
    if (domain == "unet")              head = "lora_unet_";
    else if (domain == "text_encoder") head = "lora_te_";
    else fail("diffusers_to_kohya_prefix: unknown domain '" + domain + "'");

    std::string body = target_path;
    std::replace(body.begin(), body.end(), '.', '_');
    return head + body;
}

bool kohya_to_diffusers(const std::string& kohya_prefix,
                        std::string& domain,
                        std::string& target_path) {
    return parse_kohya_prefix(kohya_prefix, domain, target_path);
}

std::vector<Triple> enumerate(const st::File& f) {
    const Format fmt = detect_format(f);

    struct Entry {
        std::string domain;
        std::string target_path;
        std::string down_key;
        std::string up_key;
        std::string alpha_key;  // empty if absent
        int rank = 0;
    };
    // Keyed by (domain + '/' + target_path) for stable dedup.
    std::unordered_map<std::string, Entry> map;
    std::vector<std::string> unrecognized;

    for (const st::TensorView& tv : f.tensors()) {
        const std::string& key = tv.name;
        if (is_lora_bookkeeping(key)) continue;

        std::string prefix;
        int kind = strip_suffix(key, prefix);
        if (kind == 0) {
            // Not a LoRA tensor name at all — skip silently. Could be a
            // network_module hint, training config, etc.
            continue;
        }

        std::string domain, target_path;
        bool ok = (fmt == Format::Kohya)
                      ? parse_kohya_prefix(prefix, domain, target_path)
                      : parse_diffusers_prefix(prefix, domain, target_path);
        if (!ok) {
            unrecognized.push_back(key);
            continue;
        }

        const std::string id = domain + "/" + target_path;
        Entry& e = map[id];
        if (e.domain.empty()) {
            e.domain = domain;
            e.target_path = target_path;
        }
        if (kind == 1) e.down_key = key;
        else if (kind == 2) e.up_key = key;
        else if (kind == 3) e.alpha_key = key;
    }

    if (!unrecognized.empty()) {
        std::string msg = "unsupported LoRA target keys (first few): ";
        const std::size_t lim = std::min<std::size_t>(unrecognized.size(), 5);
        for (std::size_t i = 0; i < lim; ++i) {
            if (i) msg += ", ";
            msg += unrecognized[i];
        }
        if (unrecognized.size() > lim) {
            msg += " (+" + std::to_string(unrecognized.size() - lim) + " more)";
        }
        fail(msg);
    }

    std::vector<Triple> out;
    out.reserve(map.size());
    for (auto& kv : map) {
        Entry& e = kv.second;
        if (e.down_key.empty() || e.up_key.empty()) {
            fail("incomplete LoRA pair for '" + e.domain + "/" + e.target_path +
                 "' (missing " + (e.down_key.empty() ? "lora_down" : "lora_up") + ")");
        }
        const st::TensorView& down = f.get(e.down_key);
        // rank = lora_down.shape[0]. For conv weights this is still
        // shape[0]; we don't care here, the UNet/CLIP merge code handles it.
        if (down.shape.empty()) {
            fail("lora_down '" + e.down_key + "' has empty shape");
        }
        e.rank = static_cast<int>(down.shape[0]);
        if (e.rank <= 0) {
            fail("lora_down '" + e.down_key + "' has non-positive rank " +
                 std::to_string(e.rank));
        }

        float alpha;
        if (e.alpha_key.empty()) {
            alpha = static_cast<float>(e.rank);
        } else {
            const st::TensorView& av = f.get(e.alpha_key);
            // .alpha is a scalar (shape [] or [1]) in F32 or F16. Read one value.
            if (av.numel() != 1) {
                fail("alpha tensor '" + e.alpha_key + "' must have a single element (got " +
                     std::to_string(av.numel()) + ")");
            }
            if (av.dtype == st::Dtype::F32) {
                alpha = *reinterpret_cast<const float*>(av.data);
            } else if (av.dtype == st::Dtype::F16) {
                uint16_t bits = *reinterpret_cast<const uint16_t*>(av.data);
                alpha = brotensor::fp16_bits_to_fp32(bits);
            } else {
                fail("alpha tensor '" + e.alpha_key + "' has unsupported dtype " +
                     st::dtype_name(av.dtype));
            }
        }

        out.push_back(Triple{
            e.domain, e.target_path, e.down_key, e.up_key, alpha, e.rank
        });
    }
    // Deterministic order (by domain then target_path) for reproducibility.
    std::sort(out.begin(), out.end(),
              [](const Triple& a, const Triple& b) {
                  if (a.domain != b.domain) return a.domain < b.domain;
                  return a.target_path < b.target_path;
              });
    return out;
}

}  // namespace brodiffusion::lora
