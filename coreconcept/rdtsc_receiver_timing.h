// ============================================================================
// rdtsc_receiver_timing.h
//
// Receiver-side counterpart to the sender's rdtsc_timing.h -- same core
// (RDTSC + LFENCE, cycles-per-ms calibrated once against wall-clock time),
// different stages, matched to what the receiver's decode loop actually
// does per frame: decode, canvas apply, and the two AI-upscale paths
// (cheap per-rect bicubic vs. the full FSRCNN-s refresh).
// ============================================================================
#pragma once

#include "rdtsc_timing.h" // reuses timing::rdtscFenced / calibrateCyclesPerMs

namespace timing {

enum ReceiveStage : int {
    RECV_DECODE = 0,     // MJPEG+FFV1 decode + AddResidualPlane reconstruct (full-frame path)
    RECV_DELTA_APPLY,    // copy-rect apply + LZ4 decompress + row paste (delta path)
    RECV_CHEAP_UPSCALE,  // CheapRectUpscale + FeatherPasteRect (the ~1.1ms per-rect case)
    RECV_FULL_UPSCALE,   // refreshMasterFull -- the real FSRCNN-s pass (the ~21ms occasional case)
    RECV_STAGE_COUNT
};

class ReceiveBudgetTracker {
public:
    explicit ReceiveBudgetTracker(double targetFps) : targetFps_(targetFps) {
        cyclesPerMs_ = calibrateCyclesPerMs();
    }

    class Scope {
    public:
        Scope(ReceiveBudgetTracker &owner, ReceiveStage stage)
            : owner_(owner), stage_(stage), start_(rdtscFenced()) {}
        ~Scope() {
            uint64_t elapsed = rdtscFenced() - start_;
            owner_.totalCycles_[stage_] += elapsed;
            owner_.count_[stage_]++;
        }
        Scope(const Scope &) = delete;
    private:
        ReceiveBudgetTracker &owner_;
        ReceiveStage stage_;
        uint64_t start_;
    };

    Scope scope(ReceiveStage s) { return Scope(*this, s); }

    // Reports against the SAME kFramePeriod (~41.67ms at 24fps) the sender
    // uses -- both sides are being held to the same one-second/24-frame
    // budget, just measuring different work.
    void maybeReport(int64_t frameIndex, double framePeriodMs, int every = 24) const {
        if (frameIndex == 0 || frameIndex % every != 0) return;
        static const char *names[RECV_STAGE_COUNT] = {
            "decode+residual", "delta apply", "cheap upscale", "full FSRCNN refresh"
        };
        double sumMs = 0.0;
        std::fprintf(stderr, "[timing][recv] --- budget check @ frame %lld (%.3fms/frame budget) ---\n",
                     static_cast<long long>(frameIndex), framePeriodMs);
        for (int s = 0; s < RECV_STAGE_COUNT; ++s) {
            if (count_[s] == 0) continue;
            double avgMs = (double)totalCycles_[s] / count_[s] / cyclesPerMs_;
            sumMs += avgMs;
            std::fprintf(stderr, "[timing][recv]   %-20s avg %.3fms  (%llu samples)\n",
                         names[s], avgMs, (unsigned long long)count_[s]);
        }
        std::fprintf(stderr, "[timing][recv]   sum of per-frame-typical stages vs budget: %.3fms / %.3fms\n",
                     sumMs, framePeriodMs);
    }

private:
    double targetFps_;
    double cyclesPerMs_;
    uint64_t totalCycles_[RECV_STAGE_COUNT] = {};
    uint64_t count_[RECV_STAGE_COUNT] = {};
};

} // namespace timing
