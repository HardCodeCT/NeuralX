#include "ai_upscaler.h"
#include "fsrcnn.h"
#include "fsutils.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/core/ocl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <pdh.h>
#pragma comment(lib, "Pdh.lib")
#endif

namespace aiup {

namespace {
inline uint8_t clampByte(float v) {
    if (v < 0.0f) return 0;
    if (v > 255.0f) return 255;
    return uint8_t(v + 0.5f);
}

// ----------------------------------------------------------------------
// GPU device selection: discrete GPU > integrated GPU > CPU.
//
// OpenCL itself has no load/utilization query, so "is this device busy"
// is answered with an OS-level mechanism (PDH on Windows), entirely
// separate from the OpenCL device enumeration below.
// ----------------------------------------------------------------------

constexpr double kMaxDeviceLoadPercent = 80.0; // skip a tier if busier than this

enum class GpuTier { kDiscrete, kIntegrated };

struct GpuCandidate {
    GpuTier tier;
    std::string name; // used both for logging and for pinning OpenCV to it
};

// Backup signal for integrated-vs-discrete classification: some drivers
// don't report CL_DEVICE_HOST_UNIFIED_MEMORY correctly (or at all) for
// APU-style parts, so cross-check against known integrated-GPU naming.
bool NameSuggestsIntegratedGpu(const std::string &name) {
    static const char *kNeedles[] = {
        "UHD Graphics", "Iris", "HD Graphics", "Radeon(TM) Graphics", "Radeon Graphics",
    };
    for (const char *needle : kNeedles) {
        if (name.find(needle) != std::string::npos) return true;
    }
    return false;
}

// Enumerates OpenCL devices via OpenCV's own cv::ocl wrappers (no raw
// clGetPlatformIDs/clGetDeviceIDs needed, so no extra OpenCL.lib
// dependency beyond what OpenCV already pulls in dynamically), and
// classifies each GPU device as discrete or integrated using
// Device::hostUnifiedMemory() as the primary signal.
std::vector<GpuCandidate> EnumerateGpuCandidates() {
    std::vector<GpuCandidate> discrete, integrated;

    std::vector<cv::ocl::PlatformInfo> platforms;
    cv::ocl::getPlatfomsInfo(platforms);

    for (const cv::ocl::PlatformInfo &platform : platforms) {
        for (int i = 0; i < platform.deviceNumber(); ++i) {
            cv::ocl::Device dev;
            platform.getDevice(dev, i);
            if (dev.type() != cv::ocl::Device::TYPE_GPU) continue;

            const std::string name = dev.name();
            const bool hostUnified = dev.hostUnifiedMemory();
            const bool looksIntegrated = hostUnified || NameSuggestsIntegratedGpu(name);

            if (looksIntegrated) {
                integrated.push_back({GpuTier::kIntegrated, name});
            } else {
                discrete.push_back({GpuTier::kDiscrete, name});
            }
        }
    }

    // Discrete first, integrated after -- caller walks this in order.
    std::vector<GpuCandidate> ordered = std::move(discrete);
    ordered.insert(ordered.end(), integrated.begin(), integrated.end());
    return ordered;
}

#if defined(_WIN32)
// \GPU Engine(*)\Utilization Percentage is a wildcard-instance counter
// (one instance per process/engine/adapter combination). This reports the
// single busiest instance as "the" GPU load -- a deliberate simplification
// that doesn't distinguish per-adapter load when multiple GPUs are
// present (see the caveats in the message to the user).
//
// Like all PDH "rate" counters, this one needs two samples ~1s apart to
// produce a real rate rather than a meaningless first-sample value --
// hence the Sleep(1000) below and the startup delay it costs the caller.
std::optional<double> SampleGpuEnginePercentMax() {
    PDH_HQUERY query = nullptr;
    if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) return std::nullopt;

    PDH_HCOUNTER counter = nullptr;
    if (PdhAddEnglishCounterW(query, L"\\GPU Engine(*)\\Utilization Percentage", 0, &counter) !=
        ERROR_SUCCESS) {
        PdhCloseQuery(query);
        return std::nullopt;
    }

    PdhCollectQueryData(query);
    Sleep(1000);
    if (PdhCollectQueryData(query) != ERROR_SUCCESS) {
        PdhCloseQuery(query);
        return std::nullopt;
    }

    DWORD bufferSize = 0, itemCount = 0;
    PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &bufferSize, &itemCount, nullptr);
    if (bufferSize == 0) {
        PdhCloseQuery(query);
        return std::nullopt; // no GPU Engine instances at all -- treat as unknown
    }

    std::vector<uint8_t> buffer(bufferSize);
    auto *items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W *>(buffer.data());
    if (PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &bufferSize, &itemCount, items) !=
        ERROR_SUCCESS) {
        PdhCloseQuery(query);
        return std::nullopt;
    }

    double maxPercent = 0.0;
    for (DWORD i = 0; i < itemCount; ++i) {
        if (items[i].FmtValue.CStatus == ERROR_SUCCESS) {
            maxPercent = std::max(maxPercent, items[i].FmtValue.doubleValue);
        }
    }

    PdhCloseQuery(query);
    return maxPercent;
}
#endif // _WIN32

} // namespace

struct Upscaler::Impl {
    int scale;
    Backend backend;

    // Exactly one of these is populated, depending on `backend`.
    std::unique_ptr<FSRCNN_FAST> cpuNet;                              // kCpuEigen
    std::unique_ptr<cv::dnn_superres::DnnSuperResImpl> openclNet;      // kOpenCLiGpu

    Impl(int s, Backend b, const std::string &modelDir) : scale(s), backend(b) {
        if (backend == Backend::kOpenCLiGpu) {
            if (!cv::ocl::haveOpenCL()) {
                std::fprintf(stderr,
                    "[ai_upscaler] OpenCL not available on this system -- "
                    "falling back to CPU (kCpuEigen) backend.\n");
                backend = Backend::kCpuEigen;
            } else {
                std::optional<GpuCandidate> chosen = SelectGpuDevice();
                if (!chosen.has_value()) {
                    std::fprintf(stderr,
                        "[ai_upscaler] No usable OpenCL GPU device found (none present, or all "
                        "over the %.0f%% load threshold) -- falling back to CPU (kCpuEigen) backend.\n",
                        kMaxDeviceLoadPercent);
                    backend = Backend::kCpuEigen;
                } else {
                    // Pin OpenCV's OpenCL context to the specific device we picked
                    // (rather than letting it default to whichever device its own
                    // heuristic lands on) via its device-selection env var. This
                    // must happen before setUseOpenCL()/any OpenCL context use --
                    // the platform/device enumeration above (getPlatfomsInfo) does
                    // NOT itself trigger context creation, so this is still safe
                    // to set at this point.
                    const std::string deviceConfig = ":GPU:" + chosen->name;
#if defined(_WIN32)
                    _putenv_s("OPENCV_OPENCL_DEVICE", deviceConfig.c_str());
#else
                    setenv("OPENCV_OPENCL_DEVICE", deviceConfig.c_str(), 1);
#endif

                    cv::ocl::setUseOpenCL(true);
                    if (!cv::ocl::useOpenCL()) {
                        std::fprintf(stderr,
                            "[ai_upscaler] OpenCL reported available but could not be "
                            "enabled -- falling back to CPU (kCpuEigen) backend.\n");
                        backend = Backend::kCpuEigen;
                    } else {
                        const std::string modelPath = modelDir + "/fsrcnn_s_x" + std::to_string(s) + ".pb";
                        auto net = std::make_unique<cv::dnn_superres::DnnSuperResImpl>();
                        try {
                            net->readModel(modelPath);
                            net->setModel("fsrcnn", s);
                            net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                            net->setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
                            openclNet = std::move(net);
                            cv::ocl::Device dev = cv::ocl::Device::getDefault();
                            std::fprintf(stderr,
                                "[ai_upscaler] Using OpenCL (%s GPU) backend: %s\n",
                                chosen->tier == GpuTier::kDiscrete ? "discrete" : "integrated",
                                dev.name().c_str());
                        } catch (const cv::Exception &ex) {
                            std::fprintf(stderr,
                                "[ai_upscaler] Failed to load '%s' for the OpenCL backend (%s) -- "
                                "falling back to CPU (kCpuEigen) backend.\n", modelPath.c_str(), ex.what());
                            backend = Backend::kCpuEigen;
                        }
                    }
                }
            }
        }

        if (backend == Backend::kCpuEigen) {
            cpuNet = std::make_unique<FSRCNN_FAST>(s);
        }
    }

    // Priority: discrete GPU > integrated GPU > CPU (kCpuEigen, handled by
    // the caller as the unconditional final fallback -- there's no separate
    // "OpenCL CPU device" tier here, since it would be functionally
    // redundant with kCpuEigen). Skips any GPU tier whose only device(s)
    // are over kMaxDeviceLoadPercent.
    static std::optional<GpuCandidate> SelectGpuDevice() {
        std::vector<GpuCandidate> candidates = EnumerateGpuCandidates();
        if (candidates.empty()) {
            std::fprintf(stderr, "[ai_upscaler] No OpenCL GPU devices found.\n");
            return std::nullopt;
        }

#if defined(_WIN32)
        // One load sample covers both tiers below -- OS-level GPU engine
        // utilization isn't broken down per discrete-vs-integrated adapter
        // by this check (see the caveats in the accompanying explanation).
        std::optional<double> gpuLoad = SampleGpuEnginePercentMax();
        if (!gpuLoad.has_value()) {
            std::fprintf(stderr,
                "[ai_upscaler] Could not read GPU load via PDH -- proceeding without a "
                "load check.\n");
        }
#else
        std::optional<double> gpuLoad = std::nullopt; // PDH is Windows-only
#endif

        for (GpuTier tier : {GpuTier::kDiscrete, GpuTier::kIntegrated}) {
            for (const GpuCandidate &cand : candidates) {
                if (cand.tier != tier) continue;
                if (gpuLoad.has_value() && *gpuLoad > kMaxDeviceLoadPercent) {
                    std::fprintf(stderr,
                        "[ai_upscaler] %s GPU '%s' at %.0f%% utilization, skipping.\n",
                        tier == GpuTier::kDiscrete ? "Discrete" : "Integrated",
                        cand.name.c_str(), *gpuLoad);
                    continue;
                }
                return cand;
            }
        }
        return std::nullopt;
    }
};

Upscaler::Upscaler(int scale, Backend backend, const std::string &modelDir)
    : impl_(new Impl(scale, backend, modelDir)) {}
Upscaler::~Upscaler() { delete impl_; }
int Upscaler::scale() const { return impl_->scale; }
