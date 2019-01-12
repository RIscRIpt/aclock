#include "ClockFaceReader.h"

#include <iostream>
#include <iomanip>
#include <deque>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Gradient.h"

ClockFaceReader::ClockFaceReader(cv::Mat const &image, int preferred_size)
    : Algorithm(image, preferred_size)
    , time_({ -1, -1 })
{
}

std::vector<cv::Mat> const& ClockFaceReader::getSteps() {
    execute();
    return debug_steps_;
}

std::pair<int, int> ClockFaceReader::getTime() {
    execute();
    return time_;
}

void ClockFaceReader::execute() {
    if (executed_)
        return;
    executed_ = true;

    auto image = image_.clone();
    auto resize_factor = resize(image);
    preprocess(image);

    if (image.type() != CV_32F) {
        throw std::runtime_error("image.type() must be CV_32F");
    }

    int const radius = cv::min(image.rows, image.cols) / 2;
    float const rotation_step = 2 * CV_PI / 60.0;
    auto inner_radius = determineInnerRadius(image) * 1.33;

    if (inner_radius <= 0)
        return;

    std::deque<std::pair<float, int>> extrema;

    for (float inner_angle = 0.0f; inner_angle < 2 * CV_PI; inner_angle += rotation_step) {
        cv::Point center = image.size() / 2;
        center.x += std::sin(inner_angle) * inner_radius;
        center.y -= std::cos(inner_angle) * inner_radius;

        for (float angle = 0.0f; angle < 2 * CV_PI; angle += rotation_step) {
            int line_start = 0;
            int line_end = 0;

            for (int line_length = 1; line_length <= radius / 3; line_length++) {
                auto non_zero = countNonZeroOnRadius(image, center, angle, line_length);
                if (non_zero.first == 0)
                    continue;
                line_start = line_length;
                break;
            }

            if (line_start == 0)
                continue;

            int max_non_zero = -1;
            for (int line_length = line_start + 1; line_length <= radius; line_length++) {
                auto non_zero = countNonZeroOnRadius(image, center, angle, line_length, line_start);
                if (non_zero.first < non_zero.second) {
                    line_end = line_length - 1;
                    break;
                }
                max_non_zero = non_zero.first;
            }

            if (line_end <= line_start)
                continue;

            if (extrema.empty() || extrema.back().second < max_non_zero) {
                if (extrema.size() > 2)
                    extrema.pop_front();
                extrema.push_back({ angle, max_non_zero });
            }
        }
    }

    time_ = { angleToHours(extrema.front().first), angleToMinutes(extrema.back().first) };
}

void ClockFaceReader::preprocess(cv::Mat &image) {
    if (debug_) {
        std::cout << "preprocessing\n";
    }

    debug(image);

    cv::medianBlur(image, image, 5);
    debug(image);

    cv::normalize(Gradient(image).magnitude(), image, 0, 1, CV_MINMAX, CV_32F);
    debug(image);

    cv::dilate(image, image, makeMatSquare(7));
    debug(image);

    cv::erode(image, image, makeMatSquare(9));
    debug(image);

    cv::GaussianBlur(image, image, { 7, 7 }, 1);
    debug(image);

    cv::threshold(image, image, 0.4, 0, CV_THRESH_TOZERO);
    debug(image);

    if (debug_) {
        std::cout << "preprocessing done\n";
    }
}

int ClockFaceReader::determineInnerRadius(cv::Mat const &image) {
    cv::Point center = image.size() / 2;
    int max_radius = cv::min(image.cols, image.rows) / 3;
    for (int radius = 1; radius < max_radius; radius++) {
        cv::Mat mask = cv::Mat::zeros(image.size(), image.type());
        cv::circle(mask, center, radius, cv::Scalar::all(1), -1);
        if (cv::countNonZero(image.mul(mask)) > 0)
            return radius;
    }
    return -1;
}

std::pair<int, int> ClockFaceReader::countNonZeroOnRadius(cv::Mat const &image, cv::Point const &center, float angle, int radius, int offset) const {
    cv::Mat mask = cv::Mat::zeros(image.size(), image.type());
    cv::Point start_point, end_point;
    start_point.x = center.x + std::sin(angle) * offset;
    start_point.y = center.y - std::cos(angle) * offset;
    end_point.x = center.x + std::sin(angle) * radius;
    end_point.y = center.y - std::cos(angle) * radius;
    cv::line(mask, center, end_point, cv::Scalar::all(1));
    auto count = cv::countNonZero(image.mul(mask));
    auto max_count = cv::countNonZero(mask);
    return { count, max_count };
}

int ClockFaceReader::angleToHours(float angle) const {
    int hours = std::floor(angle / (2 * CV_PI) * 12);
    return hours != 0 ? hours : 12;
}

int ClockFaceReader::angleToMinutes(float angle) const {
    return std::round(angle / (2 * CV_PI) * 60);
}

cv::Mat ClockFaceReader::makeMatSquare(int size) const {
    return cv::getStructuringElement(cv::MORPH_RECT, { size, size });
}
