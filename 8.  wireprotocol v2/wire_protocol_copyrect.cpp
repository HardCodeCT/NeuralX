// ============================================================================
// wire_protocol_copyrect.cpp  (excerpt from wire_protocol.h, verbatim)
//
// UPDATE to the earlier wire-protocol post: MSG_FRAME_DELTA's payload is no
// longer just [rect header][LZ4 bytes]. It now carries an optional list of
// CopyRect entries FIRST, followed by the same trailing raw-rect format as
// before (unchanged in byte layout, only its offset within the payload moved).
// ============================================================================

#include <cstdint>
#include <vector>

namespace wire {

// "Receiver, copy your own canvas content from (srcX,srcY) to (dstX,dstY)."
// Zero new pixel bytes cross the wire for this region -- just 24 bytes of
// coordinates, regardless of how large w/h actually are.
struct CopyRect {
    uint32_t srcX, srcY, dstX, dstY, w, h;
};
constexpr size_t COPY_RECT_SIZE = 24;       // 6 x uint32
constexpr size_t COPY_RECT_COUNT_SIZE = 4;  // uint32 count prefix

inline void putCopyRects(std::vector<uint8_t> &out, const std::vector<CopyRect> &rects) {
    putU32(out, uint32_t(rects.size()));
    for (const auto &r : rects) {
        putU32(out, r.srcX); putU32(out, r.srcY);
        putU32(out, r.dstX); putU32(out, r.dstY);
        putU32(out, r.w);    putU32(out, r.h);
    }
}

// Parses the copy-rect list starting at p (which must have at least
// availableBytes valid bytes after it). Returns false (and leaves outRects
// untouched) if the declared count would run past availableBytes -- caught
// as a malformed-payload rejection, BEFORE a single entry is read, not an
// out-of-bounds read partway through. On success, *outBytesConsumed is set
// to COPY_RECT_COUNT_SIZE + count*COPY_RECT_SIZE, i.e. the offset where the
// trailing raw-rect header begins.
inline bool parseCopyRects(const uint8_t *p, size_t availableBytes,
                            std::vector<CopyRect> &outRects, size_t *outBytesConsumed) {
    if (availableBytes < COPY_RECT_COUNT_SIZE) return false;
    const uint32_t count = getU32(p);
    const size_t needed = COPY_RECT_COUNT_SIZE + size_t(count) * COPY_RECT_SIZE;
    if (needed > availableBytes) return false; // malformed/truncated -- bail before reading OOB

    outRects.clear();
    outRects.reserve(count);
    const uint8_t *q = p + COPY_RECT_COUNT_SIZE;
    for (uint32_t i = 0; i < count; ++i) {
        CopyRect r;
        r.srcX = getU32(q + 0);  r.srcY = getU32(q + 4);
        r.dstX = getU32(q + 8);  r.dstY = getU32(q + 12);
        r.w    = getU32(q + 16); r.h    = getU32(q + 20);
        outRects.push_back(r);
        q += COPY_RECT_SIZE;
    }
    *outBytesConsumed = needed;
    return true;
}

// ---- Trailing raw-rect header: SAME 16-byte layout as the original post --
// only its OFFSET within the payload changed (now after the copy-rect
// list, not at byte 0). rect_w/rect_h == 0 means "no new pixels, nothing
// follows" -- this can now legitimately happen even when copy rects are
// present (pure motion, nothing new), not just in the fully-static case. ----
constexpr size_t RECT_HEADER_SIZE = 16; // 4 x uint32

// Empty delta: zero copy rects, zero new pixels -- the fully-static-frame
// case, unchanged in spirit from the original post, just with the new
// numCopyRects=0 prefix added.
inline std::vector<uint8_t> buildEmptyDeltaPayload() {
    std::vector<uint8_t> b;
    putU32(b, 0); // numCopyRects = 0
    putU32(b, 0); putU32(b, 0); putU32(b, 0); putU32(b, 0); // x,y,w=0,h=0
    return b;
}

inline void appendRawRectHeader(std::vector<uint8_t> &out, int rectX, int rectY, int rectW, int rectH) {
    putU32(out, uint32_t(rectX)); putU32(out, uint32_t(rectY));
    putU32(out, uint32_t(rectW)); putU32(out, uint32_t(rectH));
    // caller appends the LZ4-compressed bytes after this call, if rectW/rectH > 0
}

struct ParsedDeltaHeader { int rectX, rectY, rectW, rectH; };
inline ParsedDeltaHeader parseRawRectHeaderAt(const uint8_t *p) {
    ParsedDeltaHeader h;
    h.rectX = int(getU32(p + 0));  h.rectY = int(getU32(p + 4));
    h.rectW = int(getU32(p + 8));  h.rectH = int(getU32(p + 12));
    return h;
}

} // namespace wire
