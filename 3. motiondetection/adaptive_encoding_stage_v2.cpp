// ============================================================================
// adaptive_encoding_stage_v2.cpp
//
// UPDATE to the earlier adaptive-encoding post: the bounding box computed
// after the >50%-dirty check no longer goes straight to LZ4. It's now
// handed to MotionDetector first, which tries to EXPLAIN the box as
// content that already existed somewhere in the previous frame (a dragged
// window, a scrolled region) before falling back to sending raw pixels for
// whatever's left. Excerpted verbatim from sender_client.cpp's networkLoop
// and sendDeltaFrame -- only the surrounding class members are omitted.
// ============================================================================

#include "diff_detector.h"
#include "motion_detector.h"
#include "wire_protocol.h"
#include "lz4_rect_codec.h"

// ---- Members added to ScreenCaptureSender for this update ----
//
// Full-RESOLUTION previous frame, kept ONLY for motion_detector.h's
// moved-block search -- diff_detector.h's own dirty-detection never needs
// this, it runs entirely on the downscaled proxies. Rolling-hash motion
// search needs pixel-exact content: the downscaled proxy's box-average
// blur changes under translation unless the shift happens to be an exact
// multiple of the downscale factor, so a full-res copy is unavoidable here.
// std::vector<uint8_t>        prevFullFrame_;
// bool                        havePrevFullFrame_ = false;
// motiondet::MotionDetector   motionDetector_;
// std::vector<wire::CopyRect> copyRectsScratch_;

// ---- The decision block itself, from networkLoop's delta-path branch ----
//
// Everything above this (8x8 SAD diff, >50%-dirty fallback, bounding box
// computation) is UNCHANGED from the original post. This is what runs
// once execution reaches "dirtyCount > 0 and not a full-frame fallback."
void onBoundingBoxReady(diffdet::BoundingBox fullBox, const uint8_t *bgra, int strideBytes, int64_t fi,
                         motiondet::MotionDetector &motionDetector_,
                         std::vector<uint8_t> &prevFullFrame_, bool havePrevFullFrame_,
                         std::vector<wire::CopyRect> &copyRectsScratch_,
                         int fullW_, int fullH_) {
    copyRectsScratch_.clear();
    diffdet::BoundingBox leftoverBox = fullBox; // identity default: nothing explained by motion

    // NEW: try to explain the dirty box as moved content before treating
    // it as new pixel data. Only runs once a previous full-res frame
    // exists (never on the very first delta after a full-frame reset).
    if (havePrevFullFrame_) {
        motionDetector_.DetectMotion(prevFullFrame_.data(), bgra, strideBytes,
                                      fullW_, fullH_, fullBox,
                                      copyRectsScratch_, leftoverBox);
        // DetectMotion internally skips the search (leaving copyRectsScratch_
        // empty and leftoverBox == fullBox unchanged) if the search region
        // would exceed motiondet::kMaxSearchRegionPixels -- behaves exactly
        // like the pre-motion-detection pipeline in that case, not an error.
    }

    // sendDeltaFrame's signature changed to take BOTH the copy-rect list
    // and the (possibly smaller, possibly empty) leftover box, instead of
    // just the single box the original post described.
    // sendDeltaFrame(bgra, strideBytes, fi, copyRectsScratch_, leftoverBox);
}

// ---- sendDeltaFrame itself: now sends copy rects first, then LZ4-compresses
// ONLY the leftover box -- the part motion search couldn't explain. ----
void sendDeltaFrameV2(const uint8_t *bgra, int strideBytes, int64_t fi,
                       const std::vector<wire::CopyRect> &copyRects,
                       const diffdet::BoundingBox &leftoverBox,
                       std::vector<uint8_t> &deltaPayload_,
                       std::vector<uint8_t> &rectPackScratch_,
                       std::vector<uint8_t> &rectCompressedScratch_) {
    // (sendEmptyDelta(fi) call omitted here; unchanged from the original post)

    deltaPayload_.clear();
    wire::putCopyRects(deltaPayload_, copyRects); // NEW: prefix the copy-rect list

    if (leftoverBox.empty()) {
        // Every dirty pixel was explained by motion -- zero new pixel
        // content this frame, just the copy-rect list above. This is the
        // case that didn't exist before: "moved but didn't change."
        wire::appendRawRectHeader(deltaPayload_, 0, 0, 0, 0);
        return; // sender_.sendMessage(...) call omitted here
    }

    // ---- Everything below here is UNCHANGED from the original post --
    // only now it operates on `leftoverBox` (whatever motion search
    // couldn't explain) instead of the full dirty box. ----
    const int rectRowBytes = leftoverBox.width * 4;
    for (int row = 0; row < leftoverBox.height; ++row) {
        const uint8_t *srcRow = bgra + static_cast<size_t>(leftoverBox.y + row) * strideBytes
                                      + static_cast<size_t>(leftoverBox.x) * 4;
        std::memcpy(rectPackScratch_.data() + static_cast<size_t>(row) * rectRowBytes,
                    srcRow, rectRowBytes);
    }
    const size_t rectBytes = static_cast<size_t>(rectRowBytes) * leftoverBox.height;

    const size_t compressedSize = lz4rect::CompressRect(
        rectPackScratch_.data(), rectBytes,
        rectCompressedScratch_.data(), rectCompressedScratch_.size());

    wire::appendRawRectHeader(deltaPayload_, leftoverBox.x, leftoverBox.y,
                               leftoverBox.width, leftoverBox.height);
    size_t off = deltaPayload_.size();
    deltaPayload_.resize(off + compressedSize);
    std::memcpy(deltaPayload_.data() + off, rectCompressedScratch_.data(), compressedSize);
    // sender_.sendMessage(wire::MSG_FRAME_DELTA, fi, deltaPayload_, kDeltaTileIndex); omitted
}
