// ============================================================================
// rolling_hash_internals.cpp  (excerpt from motion_detector.cpp, verbatim)
//
// The actual 2D rolling hash: builds a table over the previous frame's
// search region in O(regionW*regionH) total, not O(regionW*regionH*256) --
// the second (vertical) rolling pass over the first (horizontal) pass's
// output is what gets it there. Every pixel-offset block position gets
// hashed, not just block-grid-aligned ones, so pixel-exact drags/scrolls
// are actually findable, not just motion that happens to be a multiple of
// kBlockSize (16px).
// ============================================================================

constexpr uint64_t kRollBase = 1000000007ULL;
constexpr int kBlockSize = 16;

inline uint32_t pixelSymbol(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, 4); // avoids alignment/strict-aliasing UB; compiles to a plain load
    return v;
}

// ---- Building the table: two rolling passes ----
//
// Bpow_w / Bpow_h: kRollBase^(kBlockSize-1) -- the weight the OUTGOING
// element carried, needed to subtract its contribution before rolling in
// the new one. Same formula, computed once per axis.
uint64_t Bpow_w = 1; for (int i = 0; i < kBlockSize - 1; ++i) Bpow_w *= kRollBase;
uint64_t Bpow_h = 1; for (int i = 0; i < kBlockSize - 1; ++i) Bpow_h *= kRollBase;

// Pass 1 -- roll HORIZONTALLY across each row of the previous frame's
// search region. outW = number of valid block-start x-positions (every
// pixel offset, region width minus kBlockSize, plus one).
rowHashScratch_.assign(static_cast<size_t>(regionH) * outW, 0);
for (int y = 0; y < regionH; ++y) {
    const uint8_t *rowPtr = prevFull + static_cast<size_t>(regionY + y) * strideBytes
                                      + static_cast<size_t>(regionX) * 4;
    // Horner's rule: seed the hash for the very first block position (x=0)
    // by hashing kBlockSize pixels the "slow" way, once.
    uint64_t h = 0;
    for (int i = 0; i < kBlockSize; ++i)
        h = h * kRollBase + pixelSymbol(rowPtr + static_cast<size_t>(i) * 4);
    rowHashScratch_[static_cast<size_t>(y) * outW + 0] = h;

    // The actual ROLLING step: every subsequent x-position reuses the
    // previous hash instead of recomputing from scratch -- subtract the
    // pixel that's leaving the window (weighted by Bpow_w, since it was
    // the most-significant term), multiply everything up one place, add
    // the pixel that's entering. O(1) per shift instead of O(kBlockSize).
    for (int x = 1; x < outW; ++x) {
        uint32_t leaving  = pixelSymbol(rowPtr + static_cast<size_t>(x - 1) * 4);
        uint32_t entering = pixelSymbol(rowPtr + static_cast<size_t>(x + kBlockSize - 1) * 4);
        h = (h - static_cast<uint64_t>(leaving) * Bpow_w) * kRollBase + entering;
        rowHashScratch_[static_cast<size_t>(y) * outW + x] = h;
    }
}

// Pass 2 -- roll VERTICALLY over pass 1's output. This is what makes the
// WHOLE table O(regionW*regionH): each column of rowHashScratch_ already
// summarizes a full 16-pixel-wide row-hash, so rolling down it combines 16
// of THOSE (not 16x16 raw pixels) into each block hash -- the row pass
// already paid the cost of the horizontal dimension.
blockHashScratch_.assign(static_cast<size_t>(outH) * outW, 0);
for (int x = 0; x < outW; ++x) {
    uint64_t h = 0;
    for (int i = 0; i < kBlockSize; ++i)
        h = h * kRollBase + rowHashScratch_[static_cast<size_t>(i) * outW + x];
    blockHashScratch_[static_cast<size_t>(0) * outW + x] = h;

    for (int y = 1; y < outH; ++y) {
        uint64_t leaving  = rowHashScratch_[static_cast<size_t>(y - 1) * outW + x];
        uint64_t entering = rowHashScratch_[static_cast<size_t>(y + kBlockSize - 1) * outW + x];
        h = (h - leaving * Bpow_h) * kRollBase + entering;
        blockHashScratch_[static_cast<size_t>(y) * outW + x] = h;
    }
}

// ---- The query side: NOT rolling, deliberately ----
//
// Each dirty-box block is queried exactly once -- no sliding window to
// amortize on this side -- so it's hashed fresh with a plain weighted sum.
// CRITICAL: this MUST produce the exact same value the rolling passes
// above would produce for identical content, or every lookup silently
// misses. An earlier version used a different (XOR/shift-based) hash here
// and shipped exactly that bug: it looked fine, compiled fine, and simply
// never matched anything -- caught only by testing against a real dragged
// window, not by inspection.
uint64_t computeQueryBlockHash(const uint8_t *base, int strideBytes) {
    static const auto weights = [] {
        struct W { uint64_t col[kBlockSize], row[kBlockSize]; } w{};
        w.col[kBlockSize - 1] = 1;
        for (int i = kBlockSize - 2; i >= 0; --i) w.col[i] = w.col[i + 1] * kRollBase;
        w.row[kBlockSize - 1] = 1;
        for (int i = kBlockSize - 2; i >= 0; --i) w.row[i] = w.row[i + 1] * kRollBase;
        return w;
    }();

    uint64_t blockHash = 0;
    for (int row = 0; row < kBlockSize; ++row) {
        const uint8_t *r = base + static_cast<size_t>(row) * strideBytes;
        uint64_t rowHash = 0;
        for (int col = 0; col < kBlockSize; ++col)
            rowHash += static_cast<uint64_t>(pixelSymbol(r + static_cast<size_t>(col) * 4)) * weights.col[col];
        blockHash += rowHash * weights.row[row];
    }
    return blockHash;
}

// A hash hit is a CANDIDATE, never a confirmed match -- 64 bits makes
// accidental collisions rare but rolling hashes are more structured than a
// general-purpose hash, so every hit still gets a real memcmp before it's
// trusted enough to become a copy-rect.
bool blocksEqualExact(const uint8_t *a, int strideA, const uint8_t *b, int strideB) {
    constexpr int rowBytes = kBlockSize * 4;
    for (int row = 0; row < kBlockSize; ++row) {
        if (std::memcmp(a + static_cast<size_t>(row) * strideA,
                         b + static_cast<size_t>(row) * strideB, rowBytes) != 0)
            return false;
    }
    return true;
}
