// ============================================================================
// bandwidth_tracker.h (sender side)
//
// For every delta frame, computes what actually gets sent (copy-rect list +
// leftover LZ4 bytes) alongside what the OLD design (no motion detection --
// the full original dirty box, LZ4-compressed as one raw rect) would have
// cost for that same real frame content. The hypothetical path is genuinely
// recomputed (real LZ4 call on real pixels), never estimated -- so the
// comparison is apples-to-apples, not a guess.
// ============================================================================
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "wire_protocol.h"
#include "diff_detector.h"
#include "lz4_rect_codec.h"

class BandwidthTracker {
public:
    // fullBox: the original dirty bounding box BEFORE motion search split
    // it into copyRects + leftoverBox -- this is what the pre-motion-
    // detection pipeline would have sent as its single raw rect.
    //
    // NOTE: this recomputes LZ4 compression on leftoverBox independently
    // of sendDeltaFrame's own compression of the same region -- redundant
    // CPU work, done deliberately so this tracker doesn't need to change
    // sendDeltaFrame's signature just to smuggle its compressed size out.
    // Correctness of the measurement matters more than avoiding one extra
    // compression call here; this is instrumentation, not the hot path
    // itself (it doesn't touch what actually gets sent on the wire).
    void recordDeltaFrame(const std::vector<wire::CopyRect> &copyRects,
                           const diffdet::BoundingBox &leftoverBox,
                           const diffdet::BoundingBox &fullBox,
                           const uint8_t *bgra, int strideBytes,
                           std::vector<uint8_t> &packScratch,
                           std::vector<uint8_t> &compressScratch) {
        const size_t copyRectBytes = wire::COPY_RECT_COUNT_SIZE + copyRects.size() * wire::COPY_RECT_SIZE;

        // ---- Actual: real leftover bytes, genuinely recompressed here ----
        size_t actualLeftoverBytes = 0;
        if (!leftoverBox.empty()) {
            actualLeftoverBytes = compressRectFor(leftoverBox, bgra, strideBytes, packScratch, compressScratch);
        }
        const size_t actual = copyRectBytes + wire::RECT_HEADER_SIZE + actualLeftoverBytes;
        actualBytesSent_ += actual;
        copyRectBytesTotal_ += copyRectBytes;
        if (!copyRects.empty()) framesWithMotion_++;
        frameCount_++;

        // ---- Hypothetical: old design, same real frame content ----
        size_t hypotheticalBytes = 0;
        if (!fullBox.empty()) {
            hypotheticalBytes = compressRectFor(fullBox, bgra, strideBytes, packScratch, compressScratch);
        }
        hypotheticalBytesNoMotion_ += wire::RECT_HEADER_SIZE + hypotheticalBytes;
    }

    void report(int64_t frameIndex, int everyN = 24) const {
        if (frameIndex == 0 || frameIndex % everyN != 0) return;
        std::fprintf(stderr, "[bandwidth][send] --- motion detection impact @ frame %lld ---\n",
                     static_cast<long long>(frameIndex));
        std::fprintf(stderr, "[bandwidth][send]   frames with motion explained: %llu / %llu delta frames\n",
                     (unsigned long long)framesWithMotion_, (unsigned long long)frameCount_);
        std::fprintf(stderr, "[bandwidth][send]   actual bytes sent:            %llu\n",
                     (unsigned long long)actualBytesSent_);
        std::fprintf(stderr, "[bandwidth][send]   hypothetical (no motion det): %llu\n",
                     (unsigned long long)hypotheticalBytesNoMotion_);
        if (hypotheticalBytesNoMotion_ > 0) {
            double reduction = 100.0 * (1.0 - double(actualBytesSent_) / double(hypotheticalBytesNoMotion_));
            std::fprintf(stderr, "[bandwidth][send]   reduction: %.1f%% of what the old design would have sent\n",
                         reduction);
        }
    }

private:
    // Shared by both the actual-leftover and hypothetical-fullBox paths --
    // pack the rect's rows into a contiguous buffer, LZ4-compress it,
    // return the compressed size. Identical logic to sendDeltaFrame's own
    // packing step, duplicated here rather than shared, since this header
    // has no access to the class's private scratch buffers or method.
    static size_t compressRectFor(const diffdet::BoundingBox &box, const uint8_t *bgra, int strideBytes,
                                   std::vector<uint8_t> &packScratch, std::vector<uint8_t> &compressScratch) {
        const int rowBytes = box.width * 4;
        const size_t rawBytes = static_cast<size_t>(rowBytes) * box.height;
        packScratch.resize(rawBytes);
        for (int row = 0; row < box.height; ++row) {
            const uint8_t *srcRow = bgra + static_cast<size_t>(box.y + row) * strideBytes
                                          + static_cast<size_t>(box.x) * 4;
            std::memcpy(packScratch.data() + static_cast<size_t>(row) * rowBytes, srcRow, rowBytes);
        }
        return lz4rect::CompressRect(packScratch.data(), rawBytes, compressScratch.data(), compressScratch.size());
    }

    uint64_t frameCount_ = 0;
    uint64_t framesWithMotion_ = 0;
    uint64_t actualBytesSent_ = 0;
    uint64_t hypotheticalBytesNoMotion_ = 0;
    uint64_t copyRectBytesTotal_ = 0;
};
