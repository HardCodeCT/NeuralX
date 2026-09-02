// ============================================================================
// jpeg_encode_stage.cpp  (excerpt from codec_common.h, RDTSC-instrumented)
//
// This is just the MJPEG base-layer encode path pulled out of FrameCodec,
// with a cycle-accurate timer wrapped directly around the FFmpeg calls that
// actually do the encode work -- isolated from downscale, residual, and
// FFV1 cost so the JPEG encode itself has its own number.
//
// Why MJPEG for the base layer, not H.264/VP9/AV1:
//   Every frame here is encoded completely independently -- there is no
//   video file, no GOP, no continuous stream. H.264-class codecs get their
//   compression ratio from inter-frame prediction (P/B-frames referencing
//   previous frames), which requires encoder lookahead and reorder buffering
//   -- both add latency and complexity that this pipeline has no use for,
//   since the very next "frame" might be a completely different delta-rect
//   message instead. MJPEG is intra-only by design: one image in, one
//   compressed image out, no cross-frame state, no reorder delay. Matched
//   to a stateless, one-shot-per-frame pipeline, that's the right tool, not
//   a compromise.
//
// SIMD, honestly:
//   This encode call does NOT contain hand-written SIMD *in this file*.
//   The AVX2/SSE-level vectorization for MJPEG's DCT, quantization, and
//   Huffman-adjacent work happens INSIDE FFmpeg's own libavcodec (hand-
//   written x86 assembly the FFmpeg project maintains), selected at
//   runtime via --enable-runtime-cpudetect. What IS this project's own
//   SIMD is the residual math surrounding this call (simd_residual.cpp,
//   AVX2 intrinsics, dispatched once via architecture.cpp) -- worth being
//   precise about which SIMD belongs to which codebase.
//
// Cache locality / alignment in this file specifically:
//   - av_frame_get_buffer(frame, 32) below requests 32-byte-aligned plane
//     buffers -- matching AVX2's 256-bit (32-byte) register width, so
//     FFmpeg's internal vectorized paths can use aligned loads/stores
//     instead of falling back to unaligned access.
//   - Every scratch AVFrame is allocated ONCE in FrameCodec::init() and
//     reused every call (see below) -- no per-frame heap allocation, no
//     allocator churn, and the same memory addresses stay hot across
//     consecutive frames instead of a fresh buffer landing somewhere new
//     (and cold) in the address space every time.
//   - Encoding a downscaled (low-res) frame rather than the native capture
//     means the encoder's working set is a fraction of the full-res data --
//     less to touch means fewer cache misses, independent of any SIMD.
// ============================================================================

#include "rdtsc_timing.h"

namespace scalcodec {

// (unchanged from codec_common.h -- included here for context)
static FramePtr makeYuvFrame(int w, int h) {
    FramePtr f = allocFrame();
    f->format = AV_PIX_FMT_YUV420P;
    f->width = w; f->height = h;
    // 32-byte alignment request -- see "Cache locality" note above.
    SC_CHECK(av_frame_get_buffer(f.get(), 32), "av_frame_get_buffer");
    return f;
}

static void openMjpegEncoder(CtxPtr &ctx, int w, int h, int qscale) {
    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!codec) throw CodecError("encoder not found: mjpeg");
    ctx.reset(avcodec_alloc_context3(codec));
    ctx->width = w; ctx->height = h;
    ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    ctx->time_base = AVRational{1, 1000}; // arbitrary -- frames are independent stills
    ctx->framerate = AVRational{24, 1};
    ctx->flags |= AV_CODEC_FLAG_QSCALE;
    ctx->global_quality = FF_QP2LAMBDA * qscale; // lower qscale = higher quality
    SC_CHECK(avcodec_open2(ctx.get(), codec, nullptr), "avcodec_open2 (mjpeg encoder)");
}

// Timed: this is the actual RDTSC scope around just the FFmpeg encode call
// pair, separate from the sws_scale downscale and FFV1 residual encode that
// surround it in the real pipeline -- isolates the JPEG encode's own cost.
static void encodeMjpegTimed(AVCodecContext *enc, AVFrame *frame,
                              std::vector<uint8_t> &out, timing::FrameBudgetTracker &budget) {
    auto _timingScope = budget.scope(timing::STAGE_ENCODE);

    out.clear();
    SC_CHECK(avcodec_send_frame(enc, frame), "avcodec_send_frame");
    PacketPtr pkt = allocPacket();
    while (true) {
        int rc = avcodec_receive_packet(enc, pkt.get());
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) break;
        SC_CHECK(rc, "avcodec_receive_packet");
        size_t off = out.size();
        out.resize(off + pkt->size);
        std::memcpy(out.data() + off, pkt->data, pkt->size);
        av_packet_unref(pkt.get());
    }
}

} // namespace scalcodec
