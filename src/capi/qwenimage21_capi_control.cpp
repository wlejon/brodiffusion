// qwenimage21_capi — the conditioning control dictionary and the between-step
// control schedule.
//
// See include/brodiffusion/qwenimage21_capi.h for the contract and
// control_schedule.h for why the schedule exists at all.
//
// The one thing worth saying here that the header does not: this API has no
// prime(), so there is no moment at which the library could capture "the
// conditioning this generation is running under". qi_encode_text is that
// moment instead — it is the call that turns encoder rows into the txt buffer
// qi_forward consumes, so whatever it was handed IS the base, and every
// scheduled step rebuilds from it. A caller who edits the embeds and
// re-projects them simply moves the base, which is the behaviour a research
// loop wants.

#include "qwenimage21_capi_detail.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using qi_capi::download_fp32;
using qi_capi::guarded;

namespace {

namespace bt = ::brotensor;
namespace bd = ::brodiffusion;

bd::ControlScheduleSlot slot_from(std::string name, std::vector<float> dir,
                                  float scale, const float* alpha, int n_alpha,
                                  int lo_step, int hi_step, const char* who) {
    if (!alpha || n_alpha <= 0) {
        throw std::runtime_error(std::string(who) +
                                 ": alpha/n_alpha required (one coefficient "
                                 "per step)");
    }
    bd::ControlScheduleSlot s;
    s.name    = std::move(name);
    s.dir     = std::move(dir);
    s.scale   = scale;
    s.alpha.assign(alpha, alpha + n_alpha);
    s.lo_step = lo_step < 0 ? 0 : lo_step;
    s.hi_step = hi_step;
    return s;
}

// The named form resolves the direction out of the loaded dictionary; the
// explicit form takes it as given. Both then go through the same add/set.
bd::ControlScheduleSlot named_slot(qi_ctx* c, const char* name,
                                   const float* alpha, int n_alpha,
                                   int lo_step, int hi_step, const char* who) {
    if (!name) {
        throw std::runtime_error(std::string(who) + ": name required");
    }
    if (!c->ctl.loaded()) {
        throw std::runtime_error(
            std::string(who) +
            ": no control dictionary loaded — qi_load_control_dictionary() "
            "first, or use the _dir form");
    }
    // Throws naming the axis when the bank does not hold it.
    std::vector<float> dir = c->ctl.direction(name);
    const float scale = c->ctl.axis_scale(name);
    return slot_from(name, std::move(dir), scale, alpha, n_alpha, lo_step,
                     hi_step, who);
}

bd::ControlScheduleSlot dir_slot(const float* dir, int dim, float scale,
                                 const float* alpha, int n_alpha, int lo_step,
                                 int hi_step, const char* who) {
    if (!dir || dim <= 0) {
        throw std::runtime_error(std::string(who) +
                                 ": dir/dim required (the encoder's hidden "
                                 "width)");
    }
    return slot_from(std::string(), std::vector<float>(dir, dir + dim), scale,
                     alpha, n_alpha, lo_step, hi_step, who);
}

}  // namespace

extern "C" {

int qi_load_control_dictionary(qi_ctx* c, const char* path, int merge) {
    return guarded([&] {
        if (!path) {
            throw std::runtime_error("qi_load_control_dictionary: path "
                                     "required");
        }
        c->ctl.load(path, merge != 0);
    });
}

int qi_control_axis_count(qi_ctx* c) {
    return static_cast<int>(c->ctl.names().size());
}

int qi_control_axis_name(qi_ctx* c, int index, char* out, int cap) {
    int len = -1;
    const int rc = guarded([&] {
        const auto& n = c->ctl.names();
        if (index < 0 || static_cast<std::size_t>(index) >= n.size()) {
            throw std::runtime_error("qi_control_axis_name: index " +
                                     std::to_string(index) + " out of range (" +
                                     std::to_string(n.size()) + " axes)");
        }
        const std::string& s = n[static_cast<std::size_t>(index)];
        len = static_cast<int>(s.size());
        if (out && cap > 0) {
            const int take = len < cap - 1 ? len : cap - 1;
            std::memcpy(out, s.data(), static_cast<std::size_t>(take));
            out[take] = '\0';
        }
    });
    return rc == 0 ? len : -1;
}

int qi_control_axis_vector(qi_ctx* c, const char* name, float* dir_out,
                           float* scale_out) {
    return guarded([&] {
        if (!name) {
            throw std::runtime_error("qi_control_axis_vector: name required");
        }
        const std::vector<float> d = c->ctl.direction(name);
        if (dir_out) {
            std::memcpy(dir_out, d.data(), sizeof(float) * d.size());
        }
        if (scale_out) *scale_out = c->ctl.axis_scale(name);
    });
}

int qi_set_control_schedule(qi_ctx* c, const char* name, const float* alpha,
                            int n_alpha, int lo_step, int hi_step) {
    int slot = -1;
    const int rc = guarded([&] {
        slot = c->ctl_sched.set(named_slot(c, name, alpha, n_alpha, lo_step,
                                           hi_step,
                                           "qi_set_control_schedule"));
    });
    return rc == 0 ? slot : -1;
}

int qi_add_control_schedule(qi_ctx* c, const char* name, const float* alpha,
                            int n_alpha, int lo_step, int hi_step) {
    int slot = -1;
    const int rc = guarded([&] {
        slot = c->ctl_sched.add(named_slot(c, name, alpha, n_alpha, lo_step,
                                           hi_step,
                                           "qi_add_control_schedule"));
    });
    return rc == 0 ? slot : -1;
}

int qi_set_control_schedule_dir(qi_ctx* c, const float* dir, int dim,
                                float scale, const float* alpha, int n_alpha,
                                int lo_step, int hi_step) {
    int slot = -1;
    const int rc = guarded([&] {
        slot = c->ctl_sched.set(dir_slot(dir, dim, scale, alpha, n_alpha,
                                         lo_step, hi_step,
                                         "qi_set_control_schedule_dir"));
    });
    return rc == 0 ? slot : -1;
}

int qi_add_control_schedule_dir(qi_ctx* c, const float* dir, int dim,
                                float scale, const float* alpha, int n_alpha,
                                int lo_step, int hi_step) {
    int slot = -1;
    const int rc = guarded([&] {
        slot = c->ctl_sched.add(dir_slot(dir, dim, scale, alpha, n_alpha,
                                         lo_step, hi_step,
                                         "qi_add_control_schedule_dir"));
    });
    return rc == 0 ? slot : -1;
}

int qi_clear_control_schedules(qi_ctx* c) {
    return guarded([&] { c->ctl_sched.clear(); });
}

int qi_control_schedule_count(qi_ctx* c) { return c->ctl_sched.count(); }

int qi_control_step(qi_ctx* c, int step, float* txt_out) {
    int rows = 0;
    const int rc = guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("qi_control_step: DiT not loaded (open "
                                     "with QI_LOAD_DIT)");
        }
        if (c->ctl_sched.empty() && c->ctl_sched.at_base()) return;
        if (c->ctl_base_rows <= 0) {
            throw std::runtime_error(
                "qi_control_step: no base conditioning — call qi_encode_text() "
                "once before stepping, so the schedule knows what to rebuild "
                "from");
        }
        if (!txt_out) {
            throw std::runtime_error("qi_control_step: txt_out required");
        }
        bd::CondControl delta;
        if (!c->ctl_sched.advance(step, c->ctl.budget(), delta)) return;

        const int th = c->mc.qwenimage21.transformer.context_in_dim;
        bt::Tensor emb =
            bt::Tensor::from_host(c->ctl_base.data(), c->ctl_base_rows, th)
                .to(bt::default_device());
        // The same apply() the prime-time seam uses, on the same rows, with
        // the 2.1 row policy (no BOS row to protect).
        delta.apply(emb, /*row_end=*/-1, /*row_start=*/0);
        bt::Tensor txt;
        c->dit->encode_text(emb, txt);
        download_fp32(txt, txt_out);
        // The text half of the joint sequence changed; what the cache holds
        // describes the old one.
        c->cache.reset();
        rows = txt.rows;
    });
    return rc == 0 ? rows : -1;
}

}  // extern "C"
