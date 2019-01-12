#include "Algorithm.h"

#include <opencv2/imgproc.hpp>

Algorithm::Algorithm(cv::Mat const &image, int preferred_size)
    : image_(image.clone())
    , preferred_size_(preferred_size)
    , debug_(false)
    , executed_(false)
{
    if (image_.channels() != 1) {
        throw std::runtime_error("Algorithm accepts only grayscale images!");
    }
}

Algorithm::~Algorithm() {
}

float Algorithm::resize(cv::Mat &image) {
    if (image.rows > preferred_size_ && image.cols > preferred_size_) {
        float factor = static_cast<float>(preferred_size_) / static_cast<float>(cv::max(image.rows, image.cols));
        cv::resize(image, image, {}, factor, factor);
        return factor;
    }
    return 1.0f;
}

void Algorithm::setDebug(bool enable) {
    debug_ = enable;
}

void Algorithm::debug(cv::Mat const &image) {
    if (debug_) {
        debug_steps_.push_back(image.clone());
    }
}

void Algorithm::debugNormalized(cv::Mat const &image) {
    if (debug_) {
        cv::Mat normalized;
        switch (image.type()) {
            case CV_32F:
            case CV_64F:
                cv::normalize(image, normalized, 0, 1, CV_MINMAX);
                break;
            case CV_8U:
                cv::normalize(image, normalized, 0, 255, CV_MINMAX);
                break;
            case CV_8S:
                cv::normalize(image, normalized, -128, 127, CV_MINMAX);
                break;
            case CV_16U:
                cv::normalize(image, normalized, 0, 65535, CV_MINMAX);
                break;
            case CV_16S:
                cv::normalize(image, normalized, -32768, 32767, CV_MINMAX);
                break;
            default:
                throw std::runtime_error("unsupported type");
        }
        debug_steps_.push_back(normalized);
    }
}
