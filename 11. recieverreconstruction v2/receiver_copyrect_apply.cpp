// ============================================================================
// receiver_copyrect_apply.cpp  (excerpt from receiver_server.cpp, verbatim)
//
// Applies a batch of copy rects (content the sender's motion detector found
// already existed elsewhere in the previous frame) to BOTH the native-res
// canvas and the already-AI-upscaled masterCanvas.
//
// Why two phases (extract everything, THEN write everything) instead of
// copy-rect-by-copy-rect: if a delta message carries more than one copy
// rect and two happen to have overlapping src/dst regions -- two unrelated
// moving elements in the same frame is all it takes -- applying them one at
// a time could let a later rect read data an earlier rect already
// overwrote. That's wrong output that would never surface testing a single
// dragged window; it only appears with >=2 overlapping moves in one frame.
// Extracting every source region against the UNTOUCHED starting canvas
// first guarantees the whole batch behaves as one atomic transform,
// matching what the sender actually computed it against.
//
// Cost: a small, bounded (by copyRects.size(), not pixel count) amount of
// per-message heap allocation -- a deliberate, disclosed departure from
// this file's otherwise zero-alloc hot path, traded for batch correctness.
// Copy-rect counts per frame are small (a handful, after coalescing on the
// sender side), so this isn't a meaningful cost.
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "wire_protocol.h"

auto applyCopyRects = [&](const std::vector<wire::CopyRect> &copyRects) -> bool {
    // ---- Validate the WHOLE batch before touching either canvas. If any
    // single rect is out of bounds, the entire delta message is dropped --
    // no partial, half-applied state on the canvas. ----
    for (const auto &cr : copyRects) {
        if (cr.w == 0 || cr.h == 0 ||
            cr.srcX + cr.w > static_cast<uint32_t>(cfg.fullW) || cr.srcY + cr.h > static_cast<uint32_t>(cfg.fullH) ||
            cr.dstX + cr.w > static_cast<uint32_t>(cfg.fullW) || cr.dstY + cr.h > static_cast<uint32_t>(cfg.fullH)) {
            std::fprintf(stderr, "[receiver] copy rect (%u,%u)->(%u,%u) %ux%u out of canvas bounds, dropping delta\n",
                         cr.srcX, cr.srcY, cr.dstX, cr.dstY, cr.w, cr.h);
            return false;
        }
    }

    const int scale = upscaler.scale();
    const int masterStrideBytes = masterW * 4;

    std::vector<std::vector<uint8_t>> canvasExtract(copyRects.size());
    std::vector<std::vector<uint8_t>> masterExtract(copyRects.size());

    // ---- Phase 1: extract every source region from the UNTOUCHED canvases. ----
    for (size_t i = 0; i < copyRects.size(); ++i) {
        const auto &cr = copyRects[i];
        const int rowBytes = static_cast<int>(cr.w) * 4;
        canvasExtract[i].resize(static_cast<size_t>(rowBytes) * cr.h);
        for (uint32_t row = 0; row < cr.h; ++row) {
            const uint8_t *srcRow = canvas.data() + static_cast<size_t>(cr.srcY + row) * strideBytes
                                                   + static_cast<size_t>(cr.srcX) * 4;
            std::memcpy(canvasExtract[i].data() + static_cast<size_t>(row) * rowBytes, srcRow, rowBytes);
        }

        // masterCanvas already holds valid, already-upscaled pixels for
        // the OLD position of this content -- "moved" means same pixels,
        // different place, by definition -- so extracting at the SCALED
        // offset and copying it to the scaled destination reproduces
        // correct upscaled output with ZERO FSRCNN work for this region.
        // Cheaper even than the cheap bicubic path used for new content.
        const int mw = static_cast<int>(cr.w) * scale, mh = static_cast<int>(cr.h) * scale;
        const int mSrcX = static_cast<int>(cr.srcX) * scale, mSrcY = static_cast<int>(cr.srcY) * scale;
        const int mRowBytes = mw * 4;
        masterExtract[i].resize(static_cast<size_t>(mRowBytes) * mh);
        for (int row = 0; row < mh; ++row) {
            const uint8_t *srcRow = masterCanvas.data() + static_cast<size_t>(mSrcY + row) * masterStrideBytes
                                                         + static_cast<size_t>(mSrcX) * 4;
            std::memcpy(masterExtract[i].data() + static_cast<size_t>(row) * mRowBytes, srcRow, mRowBytes);
        }
    }

    // ---- Phase 2: write every destination region. Safe now regardless of
    // overlap -- every source byte this batch needed was already captured
    // in phase 1, before any write happened. ----
    for (size_t i = 0; i < copyRects.size(); ++i) {
        const auto &cr = copyRects[i];
        const int rowBytes = static_cast<int>(cr.w) * 4;
        for (uint32_t row = 0; row < cr.h; ++row) {
            uint8_t *dstRow = canvas.data() + static_cast<size_t>(cr.dstY + row) * strideBytes
                                             + static_cast<size_t>(cr.dstX) * 4;
            std::memcpy(dstRow, canvasExtract[i].data() + static_cast<size_t>(row) * rowBytes, rowBytes);
        }

        const int mw = static_cast<int>(cr.w) * scale, mh = static_cast<int>(cr.h) * scale;
        const int mDstX = static_cast<int>(cr.dstX) * scale, mDstY = static_cast<int>(cr.dstY) * scale;
        const int mRowBytes = mw * 4;
        for (int row = 0; row < mh; ++row) {
            uint8_t *dstRow = masterCanvas.data() + static_cast<size_t>(mDstY + row) * masterStrideBytes
                                                   + static_cast<size_t>(mDstX) * 4;
            std::memcpy(dstRow, masterExtract[i].data() + static_cast<size_t>(row) * mRowBytes, mRowBytes);
        }
    }
    return true;
};

// ---- Call site, from the MSG_FRAME_DELTA case in the main recv loop:
// copy rects are applied BEFORE the leftover raw rect is decompressed and
// pasted, and a copy-rect failure drops the whole delta (same posture as
// the raw-rect bounds check). ----
//
// if (!copyRects.empty() && !applyCopyRects(copyRects)) {
//     break; // malformed/out-of-bounds copy rect; whole delta dropped
// }
