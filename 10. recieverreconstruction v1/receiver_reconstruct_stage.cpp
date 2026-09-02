// ============================================================================
// receiver_reconstruct_stage.cpp
//
// Excerpt from receiver_server.cpp's decode loop: reads MSG_FRAME_FULL vs
// MSG_FRAME_DELTA off the TCP stream and reconstructs the persistent BGRA
// canvas accordingly. RDTSC timing added around both reconstruction paths,
// same cycle-accurate approach as the sender-side posts.
//
// Memory alignment / cache behavior on THIS side:
//   - The canvas is one persistent std::vector<uint8_t> allocated ONCE at
//     connect time (sized from cfg.fullW/fullH) and mutated in place for
//     the entire session -- never reallocated per frame, so its address
//     stays stable and its pages stay resident/hot across frames.
//   - Delta paste is necessarily row-by-row (canvas stride and rect width
//     differ in the general case), but each row copy is one contiguous
//     memcpy -- sequential access the hardware prefetcher handles well,
//     same reasoning as the sender's staging readback.
//   - Full-frame reconstruction's hot loop (AddResidualPlane) uses
//     UNALIGNED AVX2 loads/stores (_mm256_loadu_si256 / storeu), not
//     aligned -- worth being precise here: unlike diff_detector's
//     alignas(32) scratch buffer, decoded FFmpeg plane buffers and the
//     canvas don't come with a 32-byte alignment guarantee, so loadu/storeu
//     is the correct choice, not a missed optimization.
//
// SIMD on the receiver:
//   AddResidualPlane is this project's OWN AVX2 code (not FFmpeg's) --
//   same dispatch pattern as everywhere else: architecture.cpp detects the
//   CPU once, simd_residual.cpp points a function pointer at the AVX2 path.
//   It does 16-bit widened add + saturate in one instruction sequence:
//   unpack to 16-bit -> add bias -> _mm256_packus_epi16 saturates back to
//   [0,255] and restores byte order, avoiding an explicit per-pixel clamp.
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "wire_protocol.h"
#include "codec_common.h"
#include "lz4_rect_codec.h"
#include "rdtsc_timing.h"

// Lightweight receiver-side budget tracker, same RDTSC+LFENCE core as the
// sender's FrameBudgetTracker, sized for the receiver's own two hot paths.
namespace timing {
enum ReceiveStage : int { RECV_FULL_DECODE = 0, RECV_DELTA_PASTE = 1, RECV_STAGE_COUNT };

class ReceiveBudgetTracker {
public:
    ReceiveBudgetTracker() { cyclesPerMs_ = calibrateCyclesPerMs(); }
    struct Scope {
        Scope(ReceiveBudgetTracker &o, ReceiveStage s) : owner_(o), stage_(s), start_(rdtscFenced()) {}
        ~Scope() {
            uint64_t elapsed = rdtscFenced() - start_;
            owner_.totalCycles_[stage_] += elapsed;
            owner_.count_[stage_]++;
        }
        ReceiveBudgetTracker &owner_; ReceiveStage stage_; uint64_t start_;
    };
    Scope scope(ReceiveStage s) { return Scope(*this, s); }

    void maybeReport(int64_t frameIndex, int every = 24) {
        if (frameIndex == 0 || frameIndex % every != 0) return;
        static const char *names[RECV_STAGE_COUNT] = {"full decode+residual", "delta decompress+paste"};
        for (int s = 0; s < RECV_STAGE_COUNT; ++s) {
            if (count_[s] == 0) continue;
            double avgMs = (double)totalCycles_[s] / count_[s] / cyclesPerMs_;
            std::fprintf(stderr, "[timing][recv]   %-24s avg %.3fms (%llu samples)\n",
                         names[s], avgMs, (unsigned long long)count_[s]);
        }
    }
private:
    double cyclesPerMs_;
    uint64_t totalCycles_[RECV_STAGE_COUNT] = {};
    uint64_t count_[RECV_STAGE_COUNT] = {};
};
} // namespace timing

// ---- The actual decode-loop dispatch, from receiver_server.cpp's main
// while(true) { recvMessage(...) } loop -- switch on header type, reconstruct
// the persistent canvas, publish it. ----
void runReceiveLoop(TcpReceiverLike &server, wire::ParsedConfig &cfg,
                     std::vector<uint8_t> &canvas, FrameCodec &fullCodec,
                     std::vector<uint8_t> &rectDecompressScratch,
                     timing::ReceiveBudgetTracker &budget,
                     const std::function<void(int64_t)> &publishCanvas) {
    const int strideBytes = cfg.fullW * 4;

    enum class Mode { None, Full, Delta };
    Mode mode = Mode::None;
    int64_t currentFrame = -1;
    std::vector<uint8_t> baseBuf, enhBuf;
    bool haveBase = false, haveEnh = false;

    wire::Header hdr;
    std::vector<uint8_t> payload;

    while (true) {
        // recvMessage() loops recv() internally until the full header and
        // payload have arrived -- TCP gives no guarantee a single recv()
        // returns a whole message (see the wire-protocol post).
        if (!server.recvMessage(hdr, payload)) break; // sender disconnected
        if (hdr.type == wire::MSG_END) break;

        switch (hdr.type) {
            case wire::MSG_FRAME_FULL:
                // Just a marker -- the real reconstruction happens once
                // MSG_BASE + MSG_ENH have both arrived (see MSG_ENH case).
                mode = Mode::Full;
                currentFrame = hdr.frameIndex;
                haveBase = haveEnh = false;
                break;

            case wire::MSG_FRAME_DELTA: {
                mode = Mode::Delta;
                currentFrame = hdr.frameIndex;
                if (payload.size() < wire::RECT_HEADER_SIZE) break; // malformed, drop

                auto rect = wire::parseDeltaPayloadHeader(payload);
                if (rect.rectW == 0 || rect.rectH == 0) break; // unchanged: zero work, by design

                // Bounds-check BEFORE touching canvas memory -- a corrupt
                // rect_x/rect_w exceeding cfg.fullW would otherwise write
                // past the canvas buffer.
                if (rect.rectX < 0 || rect.rectY < 0 ||
                    rect.rectX + rect.rectW > cfg.fullW || rect.rectY + rect.rectH > cfg.fullH) {
                    break;
                }

                auto _timingScope = budget.scope(timing::RECV_DELTA_PASTE);

                const uint8_t *compressed = payload.data() + wire::RECT_HEADER_SIZE;
                const size_t compressedSize = payload.size() - wire::RECT_HEADER_SIZE;
                const size_t rawRectBytes = static_cast<size_t>(rect.rectW) * 4 * rect.rectH;

                lz4rect::DecompressRect(compressed, compressedSize,
                                         rectDecompressScratch.data(), rawRectBytes);

                // Row-by-row paste: canvas stride and the rect's packed
                // width differ in general, so a single whole-rect memcpy
                // only works when rect.rectW == cfg.fullW. Still O(height)
                // calls, each one a contiguous sequential memcpy.
                const int rectRowBytes = rect.rectW * 4;
                for (int row = 0; row < rect.rectH; ++row) {
                    uint8_t *dstRow = canvas.data() + static_cast<size_t>(rect.rectY + row) * strideBytes
                                                     + static_cast<size_t>(rect.rectX) * 4;
                    const uint8_t *srcRow = rectDecompressScratch.data() + static_cast<size_t>(row) * rectRowBytes;
                    std::memcpy(dstRow, srcRow, rectRowBytes);
                }
                publishCanvas(currentFrame);
                break;
            }

            case wire::MSG_BASE:
                baseBuf = std::move(payload);
                haveBase = true;
                break;

            case wire::MSG_ENH:
                enhBuf = std::move(payload);
                haveEnh = true;
                if (haveBase && haveEnh && mode == Mode::Full) {
                    auto _timingScope = budget.scope(timing::RECV_FULL_DECODE);
                    // Decodes MJPEG base + FFV1 residual, then reconstructs
                    // via AddResidualPlane (this project's own AVX2 code --
                    // see notes above) directly into the canvas buffer.
                    fullCodec.decodeFrame(baseBuf, enhBuf, canvas.data(), strideBytes);
                    publishCanvas(currentFrame);
                }
                break;

            default:
                break;
        }

        budget.maybeReport(currentFrame);
    }
}
