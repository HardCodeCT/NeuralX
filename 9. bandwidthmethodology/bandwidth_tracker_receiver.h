// ============================================================================
// bandwidth_tracker.h
//
// NOT A SIMULATION: unlike the sender-side hypothetical comparison, this
// tracks bytes the receiver ACTUALLY pulled off the wire, per message type,
// as a running total -- a real measurement of what this session cost in
// bandwidth, with the copy-rect share broken out specifically so the
// motion-detection savings are visible directly from the receive side.
// ============================================================================
#pragma once

#include <cstdint>
#include <cstdio>

class BandwidthTracker {
public:
    // Called once per recvMessage() success, with the header + payload
    // sizes actually read off the socket for that message.
    void recordMessage(uint8_t msgType, size_t headerBytes, size_t payloadBytes) {
        totalBytes_ += headerBytes + payloadBytes;
        messageCount_++;

        switch (msgType) {
            case /*MSG_BASE*/      2: baseBytes_ += payloadBytes; break;
            case /*MSG_ENH*/       3: enhBytes_ += payloadBytes; break;
            case /*MSG_FRAME_FULL*/5: fullMarkerBytes_ += payloadBytes; break;
            case /*MSG_FRAME_DELTA*/6: deltaBytes_ += payloadBytes; deltaCount_++; break;
            default: otherBytes_ += payloadBytes; break;
        }
    }

    // Called from the MSG_FRAME_DELTA handler specifically, so the
    // copy-rect share of a delta's payload is visible separately from the
    // LZ4-compressed leftover share -- this is the number that actually
    // demonstrates motion detection's wire-bandwidth effect on THIS side.
    void recordDeltaBreakdown(size_t copyRectListBytes, size_t leftoverBytes) {
        copyRectBytesTotal_ += copyRectListBytes;
        leftoverBytesTotal_ += leftoverBytes;
    }

    void report(int64_t frameIndex, int everyN = 24) const {
        if (frameIndex == 0 || frameIndex % everyN != 0) return;
        std::fprintf(stderr, "[bandwidth][recv] --- received @ frame %lld ---\n",
                     static_cast<long long>(frameIndex));
        std::fprintf(stderr, "[bandwidth][recv]   total: %llu bytes across %llu messages\n",
                     (unsigned long long)totalBytes_, (unsigned long long)messageCount_);
        std::fprintf(stderr, "[bandwidth][recv]   full-frame path (BASE+ENH): %llu bytes\n",
                     (unsigned long long)(baseBytes_ + enhBytes_));
        std::fprintf(stderr, "[bandwidth][recv]   delta path: %llu messages, %llu bytes total\n",
                     (unsigned long long)deltaCount_, (unsigned long long)deltaBytes_);
        std::fprintf(stderr, "[bandwidth][recv]     of which copy-rect lists: %llu bytes (motion, ~0 pixels)\n",
                     (unsigned long long)copyRectBytesTotal_);
        std::fprintf(stderr, "[bandwidth][recv]     of which leftover/LZ4:   %llu bytes (genuinely new pixels)\n",
                     (unsigned long long)leftoverBytesTotal_);
    }

private:
    uint64_t totalBytes_ = 0;
    uint64_t messageCount_ = 0;
    uint64_t baseBytes_ = 0;
    uint64_t enhBytes_ = 0;
    uint64_t fullMarkerBytes_ = 0;
    uint64_t deltaBytes_ = 0;
    uint64_t deltaCount_ = 0;
    uint64_t otherBytes_ = 0;
    uint64_t copyRectBytesTotal_ = 0;
    uint64_t leftoverBytesTotal_ = 0;
};
