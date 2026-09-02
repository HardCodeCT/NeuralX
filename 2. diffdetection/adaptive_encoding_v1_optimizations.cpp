// ============================================================================
// adaptive_encoding_v1_optimizations.cpp
// Every optimization technique behind the 8x8-diff / 50%-fallback decision,
// excerpted verbatim from diff_detector.cpp and the current sender_client.cpp.
// ============================================================================

// ---------------------------------------------------------------------------
// 1. SIMD (AVX2) -- diff_detector.cpp
// _mm256_sad_epu8 sums 32 bytes of abs-difference in one instruction, with
// an early-out the moment the running sum crosses `budget`.
// ---------------------------------------------------------------------------
static uint32_t rowSadAVX2(const uint8_t *a, const uint8_t *b, int count, uint32_t budget) {
    int x = 0;
    uint32_t sum = 0;
    const int vecCount = count - (count % 32);
    for (; x < vecCount; x += 32) {
        __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + x));
        __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + x));
        __m256i sad = _mm256_sad_epu8(av, bv);

        // ---- 2. Memory alignment: OWN scratch buffer, aligned on purpose ----
        alignas(32) uint64_t lanes[4];
        _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), sad); // ALIGNED store

        sum += static_cast<uint32_t>(lanes[0] + lanes[1] + lanes[2] + lanes[3]);
        if (sum > budget) return sum; // early-out: already know this tile is dirty
    }
    for (; x < count; ++x)
        sum += static_cast<uint32_t>(std::abs(static_cast<int>(a[x]) - static_cast<int>(b[x])));
    return sum;
}
// Note: av/bv above use _mm256_loadu_si256 -- UNALIGNED, deliberately. The
// frame buffer rows they read from carry no 32-byte alignment guarantee
// (they come from a Windows Graphics Capture staging texture), so loadu is
// the correct choice here, not a missed optimization.

// Runtime dispatch, cached once (architecture.cpp/simd_residual.cpp):
//   static std::once_flag g_initFlag;
//   std::call_once(g_initFlag, []() { g_simdLevel = DetectSimdLevel(); });
// -- every call site checks the cached enum, never re-probes CPUID.

// ---------------------------------------------------------------------------
// 3. Cache locality + zero-allocation hot path -- sender_client.cpp
// ---------------------------------------------------------------------------
// Diff detection runs on a downscaled proxy, not the full capture:
//   diffdet::DownscaleBoxAverage(bgra, fullW_, fullH_, strideBytes,
//                                 currFrameDs_.data(), dsW_, dsH_, dsStrideBytes);
// ~9x fewer bytes touched per comparison than native resolution.

// Every hot-path scratch buffer is sized ONCE, at connect time, never resized
// per frame:
//   const size_t maxRectBytes = static_cast<size_t>(fullW_) * fullH_ * 4;
//   rectPackScratch_.assign(maxRectBytes, 0);
//   rectCompressedScratch_.assign(lz4rect::Lz4Bound(maxRectBytes), 0);
// -- encoding a delta rectangle never triggers a heap allocation.

// ---------------------------------------------------------------------------
// 4. Lock-free thread handoff -- TripleBuffer (sender_client.cpp)
// ---------------------------------------------------------------------------
class TripleBuffer {
public:
    uint8_t *writeSlot() { return slots_[writeIdx_.load(std::memory_order_relaxed)]; }

    void publish() {
        int next = (writeIdx_.load(std::memory_order_relaxed) + 1) % 3;
        writeIdx_.store(next, std::memory_order_release);
    }

    const uint8_t *claimLatest(int64_t &outFrameIndex) {
        int idx = writeIdx_.load(std::memory_order_acquire);
        // ... returns slots_[idx] without ever taking a mutex; the capture
        // thread (WGC's FrameArrived callback) and the network/encode
        // thread never block on each other.
        return slots_[idx];
    }
private:
    uint8_t *slots_[3];
    std::atomic<int> writeIdx_{0};
};

// ---------------------------------------------------------------------------
// 5. Smart core pinning -- core_pin_planner.h + sender_client.cpp
// ---------------------------------------------------------------------------
// void networkLoop() {
//     threadutil::PinCurrentThreadToCore(
//         corePinPlanner_.AssignNextCore("sender network/encode thread"),
//         "sender network/encode thread");
//     ...
// CorePinPlanner queries AvailableCoreCount() first, builds a preferred
// list that skips core 0 (a bias carried over from a university lecture's
// claim about default OS interrupt/DPC routing, not a benchmark on this
// specific machine -- see core_pin_planner.h's full header comment), and
// falls back to core 0 explicitly, with a logged exception, on a genuine
// single-core box.

// ---------------------------------------------------------------------------
// 6. TCP_NODELAY -- TcpSender::connectTo (sender_client.cpp)
// ---------------------------------------------------------------------------
// int one = 1;
// setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY,
//            reinterpret_cast<const char *>(&one), sizeof(one));
// Without this, small header-only messages (empty deltas, full-frame
// markers) can sit buffered by Nagle's algorithm for ~40ms waiting to
// coalesce with more outgoing data -- directly undermining 24fps pacing.
