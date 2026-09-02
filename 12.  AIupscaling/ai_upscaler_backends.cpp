// ============================================================================
// ai_upscaler_backends.cpp  (excerpt from ai_upscaler.cpp / ai_upscaler.h)
//
// Two backends behind FullFrameUpscale(), unmodified from the real code:
//
//   kCpuEigen   -- thinkerleolee/FSRCNN-OpenCV's own fsrcnn.h/fsrcnn.cc,
//                  vendored verbatim, calling the closed-source
//                  TensorConv.lib through Eigen::ThreadPoolDevice(8).
//                  This is copied code, not written by me.
//
//   kOpenCLiGpu -- my own integration: OpenCV's dnn_superres module
//                  loading the same FSRCNN-s architecture from a portable
//                  .pb graph, run through cv::dnn with DNN_TARGET_OPENCL
//                  (the CPU's *integrated* GPU -- Intel UHD/Iris, AMD APU
//                  graphics -- not a discrete NVIDIA/CUDA card). This is
//                  code I wrote against OpenCV's dnn module, not vendored
//                  from the FSRCNN-OpenCV repo.
//
// Every GPU failure mode is caught and falls back to kCpuEigen automatically:
//   1. cv::ocl::haveOpenCL()   -- no OpenCL runtime present at all
//   2. cv::ocl::useOpenCL()    -- reported available but couldn't enable
//   3. net->readModel() throw  -- .pb missing/corrupt/incompatible
// Each fallback is logged, not silent -- you can always see which backend
// actually ran for a given session.
// ============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/core/ocl.hpp>
#include <memory>
#include <string>
#include <cstdio>

namespace aiup {

enum class Backend {
    kCpuEigen,    // thinkerleolee/FSRCNN-OpenCV + TensorConv.lib (CPU threads)
    kOpenCLiGpu,  // my own: OpenCV dnn_superres + DNN_TARGET_OPENCL (integrated GPU)
};

struct Upscaler::Impl {
    int scale;
    Backend backend;

    // Exactly one of these is populated, depending on `backend`.
    std::unique_ptr<FSRCNN_FAST> cpuNet;                              // kCpuEigen (vendored class)
    std::unique_ptr<cv::dnn_superres::DnnSuperResImpl> openclNet;     // kOpenCLiGpu (my integration)

    Impl(int s, Backend b, const std::string &modelDir) : scale(s), backend(b) {
        if (backend == Backend::kOpenCLiGpu) {
            // ---- Failure mode 1: no OpenCL runtime on this machine at all ----
            if (!cv::ocl::haveOpenCL()) {
                std::fprintf(stderr,
                    "[ai_upscaler] OpenCL not available on this system -- "
                    "falling back to CPU (kCpuEigen) backend.\n");
                backend = Backend::kCpuEigen;
            } else {
                cv::ocl::setUseOpenCL(true);
                // ---- Failure mode 2: OpenCL reported present but wouldn't enable ----
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
                        std::fprintf(stderr, "[ai_upscaler] Using OpenCL (integrated GPU) backend: %s\n",
                                     dev.name().c_str());
                    } catch (const cv::Exception &ex) {
                        // ---- Failure mode 3: model load/incompatibility ----
                        std::fprintf(stderr,
                            "[ai_upscaler] Failed to load '%s' for the OpenCL backend (%s) -- "
                            "falling back to CPU (kCpuEigen) backend.\n", modelPath.c_str(), ex.what());
                        backend = Backend::kCpuEigen;
                    }
                }
            }
        }

        if (backend == Backend::kCpuEigen) {
            cpuNet = std::make_unique<FSRCNN_FAST>(s); // vendored class, unmodified
        }
    }
};

// The actual inference dispatch -- same input/output contract regardless
// of which backend ended up active.
void Upscaler::FullFrameUpscale(const uint8_t *srcBGRA, int srcW, int srcH, uint8_t *dstBGRA) const {
    const int scale = impl_->scale;
    const int dstW = srcW * scale, dstH = srcH * scale;

    cv::Mat srcBGRAMat(srcH, srcW, CV_8UC4, const_cast<uint8_t *>(srcBGRA));
    cv::Mat srcBGR;
    cv::cvtColor(srcBGRAMat, srcBGR, cv::COLOR_BGRA2BGR);

    cv::Mat resultBGR;
    if (impl_->backend == Backend::kOpenCLiGpu) {
        // DNN_TARGET_OPENCL routes the conv layers through the integrated
        // GPU via OpenCV's T-API; dnn_superres handles the YCrCb
        // split/recombine internally.
        impl_->openclNet->upsample(srcBGR, resultBGR);
    } else {
        // fsutils::SR() is the original repo's driver: Y channel through
        // FSRCNN_FAST::SrOp (TensorConv.lib on CPU threads), Cb/Cr
        // bicubic, recombined back to BGR.
        resultBGR = fsutils::SR(srcBGR, *impl_->cpuNet, scale);
    }

    cv::Mat dstMat(dstH, dstW, CV_8UC4, dstBGRA);
    cv::cvtColor(resultBGR, dstMat, cv::COLOR_BGR2BGRA);
}

} // namespace aiup
