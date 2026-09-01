# NeuralX

A real-time screen-streaming pipeline for Windows: capture → change/motion detection → adaptive encoding → TCP transport → AI-upscaled reconstruction.

NeuralX is split into two standalone binaries,  a **sender** that captures and streams the display, and a **receiver** that reconstructs and optionally upscales it, communicating over a small custom protocol on `127.0.0.1:5000`. Built as a personal/local tool, not a multi-viewer server: one sender, one receiver, one connection.

---

## How it works

**Sender**
1. **Capture** — Windows Graphics Capture hands over a live D3D11 texture straight from the compositor, paced to a hard-locked 24 fps (WGC fires far more often than that on a 60Hz+ display; the sender explicitly throttles rather than processing every callback).
2. **Change detection** — an 8×8-block sum-of-absolute-differences diff runs on a downscaled proxy (~9× fewer bytes than native res), AVX2-accelerated.
3. **Motion search** — before a dirty region is treated as "new content," a 2D rolling hash (the same technique rsync uses) checks whether it already existed elsewhere in the previous frame. Matches become 24-byte copy instructions instead of resent pixels; only genuinely new content falls through to the pixel path.
4. **Adaptive encoding** — if more than 50% of blocks are dirty, the sender resets with a full frame (MJPEG base layer + FFV1 lossless residual). Otherwise, only the leftover (unexplained-by-motion) region is packed and LZ4-compressed, at full pixel quality.
5. **Transport** — plain TCP, deliberately not UDP: delta frames are patches applied to existing receiver state, not independent frames, so a dropped packet would silently desync the canvas rather than just skip a frame.

**Receiver**
1. Parses the wire protocol's self-describing framing and branches on message type.
2. **Full frames** decode through FFmpeg and reconstruct via this project's own AVX2 residual-add code.
3. **Delta frames** apply any copy-rects first (two-phase extract-then-write, so overlapping moves in the same frame can't corrupt each other), then LZ4-decompress and paste whatever's left.
4. **AI upscaling** (optional) sharpens the output with an FSRCNN-s neural network — CPU or GPU — before writing frames to disk.

---

## Engineering notes

- **SIMD, precisely, not blanket-applied.** AVX2 is used where it earns its keep (the SAD diff, the residual reconstruction, both this project's own code) and left out where it wouldn't (the rolling hash stays scalar — a correct 64-bit AVX2 multiply there is real engineering effort for little payoff at this data volume).
- **Memory alignment, applied where it's actually free.** Scratch buffers this code owns are `alignas(32)` with aligned SIMD stores. Buffers coming from capture or decode (no alignment guarantee) stay on unaligned loads — deliberately, not as an oversight.
- **Zero-allocation hot paths.** Every per-frame scratch buffer is sized once at connect time; encoding or applying a frame never touches the heap in steady state.
- **Lock-free thread handoff.** A triple buffer connects the capture callback to the encode thread via atomic acquire/release — no mutex, capture never blocks on encode.
- **Smart core pinning (opt-in).** Queries the process's real available core count, avoids core 0 as a default bias, round-robins hot threads across what's left, and falls back safely on a single-core machine. Added as an available tool, not because a scheduling problem was measured — RDTSC already showed comfortable headroom without it.
- **Cycle-accurate timing on both ends.** RDTSC (fenced with LFENCE, calibrated against wall-clock time once at startup) times every major stage on the sender *and* the receiver independently, each held to the same 41.67ms/24fps budget, reported live every 24 frames.
- **Fails safe, consistently.** GPU backend selection, core pinning, and the motion search's region-size cap all degrade gracefully on failure — never a crash, always a logged fallback to the simpler path.

## AI upscaling backends

| Backend | Origin |
|---|---|
| `kCpuEigen` | Vendored, verbatim, from [thinkerleolee/FSRCNN-OpenCV](https://github.com/thinkerleolee/FSRCNN-OpenCV) (LGPL-3.0) — architecture, trained weights, and the closed-source `TensorConv.lib` compute layer, run through `Eigen::ThreadPoolDevice`. |
| `kOpenCLiGpu` | This project's own integration against OpenCV's `dnn_superres` module, running the same FSRCNN-s architecture via `DNN_TARGET_OPENCL` — the CPU's *integrated* graphics (Intel UHD/Iris, AMD APU), not a discrete NVIDIA/CUDA card. |

Device selection (when GPU is available) prefers discrete over integrated GPUs over CPU, checks real-time load via Windows' PDH API before committing to a device, and falls back to CPU automatically on any failure — no OpenCL runtime, device busy, or model load error all degrade gracefully.

---

## Wire protocol

Every message: a fixed 17-byte header (`type`, `frame_index`, `tile_index` [legacy, unused], `payload_size`) followed by a variable-length payload. TCP is a byte stream, not a message stream — `recvMessage()` loops until the full header and payload have actually arrived, using `payload_size` to know exactly how much to wait for.

- **Full frame** — `MSG_FRAME_FULL` (empty marker) + `MSG_BASE` (MJPEG) + `MSG_ENH` (FFV1 residual), same `frame_index`.
- **Delta frame** — `MSG_FRAME_DELTA`: an optional list of `CopyRect{srcX, srcY, dstX, dstY, w, h}` entries, followed by a raw-rect header (`x, y, w, h`) and LZ4 bytes for whatever content wasn't explained by motion. `w == 0` means nothing new; an empty copy-rect list plus `w == 0` means nothing changed at all.

---

## Building

Both sender and receiver are separate MSVC x64 projects (`build.bat` in each package). The sender depends on D3D11/WGC and Winsock; the receiver depends on FFmpeg (`avcodec`, `avutil`, `swscale`), and additionally on OpenCV (`dnn_superres`, `core/ocl`) and the vendored Eigen/TensorConv if AI upscaling is enabled.

FFmpeg notes:
- Confirm `nasm` is available at configure time (`--disable-x86asm` silently drops all hand-written SIMD if missing).
- A trimmed, statically-linked build (`--disable-everything --enable-encoder=mjpeg --enable-decoder=mjpeg --enable-encoder=ffv1 --enable-decoder=ffv1 --enable-swscale`, `-march=native`, LTO) is smaller and CPU-tuned versus the generic prebuilt package — see the build notes in `codec_common.h`.

---

## Known limitations

Stated plainly rather than hidden:

- **Single leftover bounding box.** After motion search explains what it can, whatever's left is still one rectangle, not several. Verified with a synthetic scroll test: 91% of pixels explained by motion, but the leftover box came back unchanged in size because the unexplained ~9% touched an edge. Bandwidth savings are correctness-neutral but genuinely inconsistent as a result. A fix (a short list of leftover rects, or a coarse bitmap) is scoped but not built — it touches both the wire protocol and the receiver's parsing.
- **No wire-protocol version field.** A sender/receiver version mismatch would misparse silently rather than reject cleanly.
- **No automated test suite.** Correctness fixes (a rolling-hash bug caught only by testing a real dragged window; the copy-rect overlap-safety fix) are documented in code comments, not covered by CI.
- **Single connection only.** The receiver exits after one client disconnects — this is a personal tool, not a multi-viewer server.

---

## Attribution

FSRCNN-s network architecture and CPU implementation vendored from [thinkerleolee/FSRCNN-OpenCV](https://github.com/thinkerleolee/FSRCNN-OpenCV), licensed LGPL-3.0. FFmpeg (LGPL/GPL depending on build configuration) provides MJPEG and FFV1 codec support.
