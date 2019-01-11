#include "ClockFaceFinder.h"

#include <iostream>
#include <iomanip>
#include <cstdarg>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Gradient.h"

ClockFaceFinder::ClockFaceFinder(cv::Mat const &image)
    : executed_(false)
    , debug_(false)
    , max_progress_(1.0f)
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

std::vector<cv::Mat> const& ClockFaceFinder::getSteps() {
    execute();
    return debug_steps_;
}

cv::Mat ClockFaceFinder::getMaskedImage() {
    execute();
    auto result = image_.clone();
    for (auto const &found : found_) {
        cv::Mat mask = cv::Mat::zeros(image_.size(), image_.type());
        cv::circle(mask, found.location, found.radius, cv::Scalar::all(1), -1);
        std::cout << found.location << '\n' << found.radius << '\n';
        result = result.mul(mask);
    }
    return result;
}

void ClockFaceFinder::setDebug(bool enable) {
    debug_ = enable;
}

void ClockFaceFinder::execute() {
    if (executed_) {
        return;
    }
    executed_ = true;

    std::vector<cv::Mat> detections, final_detections;

    struct {
        size_t index;
        double max_value;
        cv::Point location;
    } detection_candidate{ 0 }, detection_peak{ 0 };

    float const min_radius = 10;
    float const max_radius = std::min(image_.rows, image_.cols);
    float const radius_step = 1.0f;

    initializeProgress(1.0f / (5.0f + (max_radius - min_radius) * 2.0f));

    reportProgress("preprocessing");

    auto image = image_.clone();

    debug(image);

    cv::medianBlur(image, image, 5);
    debug(image);

    cv::normalize(Gradient(image).magnitude(), image, 0, 1, CV_MINMAX, CV_32F);
    debug(image);

    cv::dilate(image, image, makeMatSquare(7));
    debug(image);

    cv::GaussianBlur(image, image, { 7, 7 }, 1);
    debug(image);

    reportProgress("preprocessing done");

    for (float radius = min_radius; radius < max_radius; radius += radius_step) {
        reportProgress("detecting clock with radius %f/%f", radius, max_radius);

        cv::Mat test;
        cv::filter2D(image, test, CV_32F, makeMatClock(radius), { -1, -1 }, 0.0, cv::BORDER_CONSTANT);
        test = test.mul(test); // square test matrix
        debugNormalized(test);
        detections.emplace_back(std::move(test));
    }

    reportProgress("detection done");

    // Smooth over 3 neighbour detections
    for (int i = 0; i < detections.size(); i++) {
        reportProgress("smooting detection layer %i/%li", i + 1, detections.size());

        auto first = std::max(i - 2, 0);
        auto last = std::min(i + 2, static_cast<int>(detections.size()) - 1);

        auto current = detections[i].clone();
        
        for (int i = first; i <= last; i++) {
            current += detections[i];
        }
        cv::GaussianBlur(current, current, { 5, 5 }, 1);

        detection_candidate.index = final_detections.size();
        cv::minMaxLoc(current, nullptr, &detection_candidate.max_value, nullptr, &detection_candidate.location);
        if (detection_candidate.max_value > detection_peak.max_value)
            detection_peak = detection_candidate;

        debugNormalized(current);
        final_detections.emplace_back(std::move(current));
    }

    reportProgress("masking input image");

    int radius = min_radius + static_cast<float>(detection_peak.index) * radius_step;
    std::cout << detection_peak.index << '\n' << radius << '\n';
    found_.push_back({ detection_peak.location, radius });

    reportProgress("done");
}

void ClockFaceFinder::debug(cv::Mat const &image) {
    if (debug_) {
        debug_steps_.push_back(image.clone());
    }
}

void ClockFaceFinder::debugNormalized(cv::Mat const &image) {
    if (debug_) {
        cv::Mat normalized;
        cv::normalize(image, normalized, 0, 1, CV_MINMAX);
        debug_steps_.push_back(normalized);
    }
}

void ClockFaceFinder::initializeProgress(float step) {
    progress_step_ = step;
}

void ClockFaceFinder::reportProgress(char const *fmt, ...) {
    total_progress_ += progress_step_;

    std::vector<char> buffer(1024);

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buffer.data(), buffer.size(), fmt, ap);
    va_end(ap);

    int flags = std::cout.flags();
    std::cout << std::fixed << std::setprecision(6) << total_progress_ << '\t';
    std::cout.flags(flags);
    std::cout << buffer.data() << '\n';
}

cv::Mat ClockFaceFinder::makeMatSquare(int size) const {
    return cv::getStructuringElement(cv::MORPH_RECT, { size, size });
}

cv::Mat ClockFaceFinder::makeMatClock(float radius) const {
    float const size = radius * 2;
    float const center = radius;
    float const max_radius = radius;
    float const min_radius = radius - 3;

    cv::Mat elem(cv::Size(size + 1, size + 1), CV_32F, cv::Scalar::all(0));

    for (radius = min_radius; radius <= max_radius; radius += 1.0f) {
        float const step = 1.0f / radius; // 2 * PI / (2 * PI * R)
        for (float i = 0.0f; i < 2 * CV_PI; i += step) {
            cv::Point point;
            point.x = center + radius * std::cos(i);
            point.y = center + radius * std::sin(i);
            elem.at<float>(point) = std::pow(std::cos(i * 12.0f), 17.0f);
        }
    }

    return elem;
}
