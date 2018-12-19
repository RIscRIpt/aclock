#include "Gradient.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>

Gradient::Gradient(cv::Mat const &image, int ddepth)
    : ddepth_(ddepth)
    , x_(image.size(), ddepth)
    , y_(image.size(), ddepth)
{
    cv::Scharr(image, x_, ddepth, 1, 0);
    cv::Scharr(image, y_, ddepth, 0, 1);
}

cv::Mat Gradient::directionInRadians() {
    if (directionInRadians_.empty())
        directionInRadians_ = direction(false);
    return directionInRadians_;
}

cv::Mat Gradient::directionInDegrees() {
    if (directionInDegrees_.empty())
        directionInDegrees_ = direction(true);
    return directionInDegrees_;
}

cv::Mat Gradient::magnitude() {
    if (magnitude_.empty()) {
        magnitude_ = cv::Mat(x_.size(), ddepth_);
        switch (ddepth_) {
            case CV_32F:
                cv::hal::magnitude32f(x_.ptr<float>(), y_.ptr<float>(), magnitude_.ptr<float>(), x_.rows * x_.cols);
                break;
            case CV_64F:
                cv::hal::magnitude64f(x_.ptr<double>(), y_.ptr<double>(), magnitude_.ptr<double>(), x_.rows * x_.cols);
                break;
            default:
                throw std::runtime_error("unsupported ddepth");
        }
    }
    return magnitude_;
}

cv::Mat Gradient::visualize() {
    if (visualization_.empty()) {
        auto dir = directionInRadians();
        auto mag = magnitude();

        dir = dir / (2 * CV_PI) * 180;
        cv::normalize(mag, mag, 0, 255, CV_MINMAX);

        auto all255 = cv::Mat::ones(dir.size(), CV_32F) * 255;

        cv::Mat channels[] = { dir, all255, mag };
        cv::merge(channels, 3, visualization_);

        visualization_.convertTo(visualization_, CV_8UC3);
        cv::cvtColor(visualization_, visualization_, cv::COLOR_HSV2BGR);
    }
    return visualization_;
}

cv::Mat Gradient::direction(bool angleInDegrees) const {
    cv::Mat result(x_.size(), ddepth_);
    switch (ddepth_) {
        case CV_32F:
            cv::hal::fastAtan32f(y_.ptr<float>(), x_.ptr<float>(), result.ptr<float>(), x_.rows * x_.cols, false);
            break;
        case CV_64F:
            cv::hal::fastAtan64f(y_.ptr<double>(), x_.ptr<double>(), result.ptr<double>(), x_.rows * x_.cols, false);
            break;
        default:
            throw std::runtime_error("unsupported ddepth");
    }
    return result;
}
