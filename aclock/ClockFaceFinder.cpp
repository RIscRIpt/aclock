#include "ClockFaceFinder.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Gradient.h"

ClockFaceFinder::ClockFaceFinder(cv::Mat const &image)
    : executed_(false)
    , debug_(false)
{
    image_ = image.clone();
    if (image_.channels() != 1) {
        throw std::runtime_error("ClockFaceFinder accepts only grayscale images!");
    }
}

std::vector<ClockFace> const& ClockFaceFinder::find() {
    execute();
    return found_;
}

std::vector<cv::Mat> const& ClockFaceFinder::get_steps() {
    execute();
    return debug_steps_;
}

void ClockFaceFinder::setDebug(bool enable) {
    debug_ = enable;
}

void ClockFaceFinder::execute() {
    if (executed_) {
        return;
    }
    executed_ = true;
    debug(image_);

    cv::medianBlur(image_, image_, 5);
    debug(image_);

    cv::blur(image_, image_, { 5, 5 });
    debug(image_);

    cv::normalize(Gradient(image_).magnitude(), image_, 0, 255, CV_MINMAX, CV_8U);
    debug(image_);

    cv::dilate(image_, image_, makeStructElemSquare(7));
    debug(image_);

    cv::resize(image_, image_, image_.size() / 2);
    debug(image_);

    for (float i = 10.0f; i < 100.0f; i += 1) {
        cv::Mat test;
        auto se = makeSturctElemClock(i);
        debug(se);
        cv::dilate(image_, test, se);
        debug(test);
    }
}

void ClockFaceFinder::debug(cv::Mat const &image) {
    if (debug_) {
        debug_steps_.push_back(image.clone());
    }
}

cv::Mat ClockFaceFinder::makeStructElemSquare(int size) const {
    return cv::getStructuringElement(cv::MORPH_RECT, { size, size });
}

cv::Mat ClockFaceFinder::makeSturctElemClock(float radius) const {
    float const size = radius * 2;
    float const center = radius;
    float const max_radius = radius;
    float const min_radius = radius * 2.0f / 3.0f;

    cv::Mat elem(cv::Size(size + 1, size + 1), CV_8U, cv::Scalar::all(0));

    for (radius = min_radius; radius <= max_radius; radius += 0.5f) {
        float const step = 1.0f / radius; // 2 * PI / (2 * PI * R)
        for (float i = 0.0f; i < 2 * CV_PI; i += step) {
            cv::Point point;
            point.x = center + radius * std::cos(i);
            point.y = center + radius * std::sin(i);
            elem.at<uchar>(point) = 255 * (std::cos(i * 12.0f) + 1.0f) / 2.0f;
        }
    }

    cv::threshold(elem, elem, 240, 255, CV_MINMAX);

    return elem;
}
