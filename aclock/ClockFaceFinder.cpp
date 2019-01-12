#include "ClockFaceFinder.h"

#include <iostream>
#include <iomanip>

#include <opencv2/imgproc.hpp>

#include "Gradient.h"

ClockFaceFinder::ClockFaceFinder(cv::Mat const &image, float min_clock_alignment, int preferred_size)
    : Algorithm(image, preferred_size)
    , min_clock_alignment_(min_clock_alignment)
    , found_(false)
    , found_clock_({ {0, 0}, -1 })
{
}

std::optional<ClockFace> const& ClockFaceFinder::find() {
    execute();
    if (!found_)
        return std::nullopt;
    return found_clock_;
}

std::vector<cv::Mat> const& ClockFaceFinder::getSteps() {
    execute();
    return debug_steps_;
}

std::optional<cv::Mat> ClockFaceFinder::getMaskedImage() {
    execute();
    if (!found_)
        return std::nullopt;
    auto result = image_.clone();
    cv::Mat mask = cv::Mat::zeros(image_.size(), image_.type());
    cv::circle(mask, found_clock_.location, found_clock_.radius, cv::Scalar::all(1), -1);
    result = result.mul(mask);
    std::vector<cv::Point> non_zeros;
    cv::findNonZero(result, non_zeros);
    return result(cv::boundingRect(non_zeros));
}

void ClockFaceFinder::execute() {
    if (executed_) {
        return;
    }
    executed_ = true;

    auto image = image_.clone();
    auto resize_factor = resize(image);
    preprocess(image);
    find_clock(image, resize_factor, 10.0f, std::min(image.rows, image.cols), 1.0f);
}

void ClockFaceFinder::preprocess(cv::Mat &image) {
    if (debug_) {
        std::cout << "preprocessing\n";
    }

    debug(image);

    cv::medianBlur(image, image, 3);
    debug(image);

    cv::normalize(Gradient(image).magnitude(), image, 0, 1, CV_MINMAX, CV_32F);
    debug(image);

    cv::dilate(image, image, makeMatSquare(3));
    debug(image);

    cv::GaussianBlur(image, image, { 5, 5 }, 1);
    debug(image);

    if (debug_) {
        std::cout << "preprocessing done\n";
    }
}

void ClockFaceFinder::find_clock(cv::Mat const &image, float resize_factor, float min_radius, float max_radius, float radius_step) {
    struct {
        double max_value;
        float radius;
        cv::Point location;
    } detection_candidate{ 0 }, detection_peak{ 0 };

    float const progress_step = 1.0f / (max_radius - min_radius);
    float total_progress = 0.0f;

    for (float radius = min_radius; radius < max_radius; radius += radius_step) {
        total_progress += progress_step;
        if (debug_) {
            std::cout << total_progress << " | detecting clock with radius " << radius << '/' << max_radius << '\t';
        }

        cv::Mat clock = makeMatClock(radius);
        cv::threshold(clock, clock, 0, 0, CV_THRESH_TOZERO);
        float threshold = cv::sum(clock)[0] * min_clock_alignment_;

        cv::Mat test;
        cv::filter2D(image, test, CV_32F, clock, { -1, -1 }, 0.0, cv::BORDER_CONSTANT);
        cv::threshold(test, test, threshold, 0, CV_THRESH_TOZERO);

        if (cv::countNonZero(test) > 0) {
            cv::GaussianBlur(test, test, { 5, 5 }, 1);

            debugNormalized(test);

            detection_candidate.radius = radius;
            cv::minMaxLoc(test, nullptr, &detection_candidate.max_value, nullptr, &detection_candidate.location);
            if (detection_candidate.max_value > detection_peak.max_value) {
                found_ = true;
                detection_peak = detection_candidate;
            }

            if (debug_) {
                std::cout << "found candidate with value " << detection_peak.max_value << '\n';
            }
        } else {
            if (debug_) {
                std::cout << "nothing found\n";
            }
        }
    }

    if (found_) {
        found_clock_ = {
            detection_peak.location / resize_factor,
            static_cast<int>(detection_peak.radius / resize_factor)
        };

        if (debug_) {
            std::cout << "clock found\n"
                << "location: " << found_clock_.location << '\n'
                << "radius:   " << found_clock_.radius << '\n';
        }
    } else {
        if (debug_) {
            std::cout << "no clock found\n";
        }
    }

    if (debug_) {
        std::cout << "done\n";
    }
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
