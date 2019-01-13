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

std::optional<std::pair<int, int>> ClockFaceReader::getTime() {
    execute();
    if (time_.first == -1 && time_.second == -1)
        return std::nullopt;
    return time_;
}

void ClockFaceReader::execute() {
    if (executed_)
        return;
    executed_ = true;

    auto image = image_.clone();
    auto resize_factor = resize(image);
    preprocess(image);

    std::vector<cv::Vec4i> all_lines;
    cv::HoughLinesP(image, all_lines, 1, CV_PI / 360.0, 10, std::min(image.cols, image.rows) / 8, 1);

    auto lines = filterLines(image.size() / 2, std::min(image.cols, image.rows) / 8, all_lines);

    if (debug_) {
        cv::Mat debug_image;
        cv::cvtColor(image, debug_image, cv::COLOR_GRAY2BGR);
        for (auto line : lines) {
            cv::line(debug_image, line.first, line.second, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        }
        debug(debug_image);
    }

    if (lines.size() == 2) {
        if (lineLength(lines[0]) > lineLength(lines[1])) {
            std::swap(lines[0], lines[1]);
        }

        time_.first = angleToHours(lineAngle(lines[0]));
        time_.second = angleToMinutes(lineAngle(lines[1]));
    }
}

void ClockFaceReader::preprocess(cv::Mat &image) {
    if (debug_) {
        std::cout << "preprocessing\n";
    }

    debug(image);

    cv::medianBlur(image, image, 5);
    debug(image);

    cv::Mat mask = cv::Mat::zeros(image.size(), image.type());
    cv::circle(mask, image.size() / 2, std::min(image.rows, image.cols) / 4, cv::Scalar::all(1));
    auto total_non_zero = cv::countNonZero(mask);
    mask = mask.mul(image);
    cv::threshold(mask, mask, 0.5, 1, CV_THRESH_BINARY);
    auto masked_non_zero = cv::countNonZero(mask);
    if (masked_non_zero >= total_non_zero / 2) {
        image = 255 - image;
        // reapply circular clock face mask
        mask = cv::Mat::zeros(image.size(), image.type());
        cv::circle(mask, image.size() / 2, std::min(image.rows, image.cols) / 2, cv::Scalar::all(1), -1);
        image = image.mul(mask);
    }
    debug(image);

    cv::threshold(image, image, 127, 255, CV_THRESH_BINARY);
    debug(image);

    cv::Mat skeleton(image.size(), image.type(), cv::Scalar(0));
    cv::Mat tmp, eroded;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, { 3, 3 });
    do {
        cv::erode(image, eroded, element);
        cv::dilate(eroded, tmp, element);
        cv::subtract(image, tmp, tmp);
        cv::bitwise_or(skeleton, tmp, skeleton);
        image = std::move(eroded);
    } while (cv::countNonZero(image) > 0);
    image = std::move(skeleton);

    debug(image);

    if (debug_) {
        std::cout << "preprocessing done\n";
    }
}

bool ClockFaceReader::pointBelongsToCenter(cv::Point const &center, int radius, cv::Point const &p) {
    int dx = center.x - p.x;
    int dy = center.y - p.y;
    return std::sqrt(dx * dx + dy * dy) <= radius;
}

std::vector<std::pair<cv::Point, cv::Point>> ClockFaceReader::filterLines(cv::Point const &center, int radius, std::vector<cv::Vec4i> const &points) {
    std::vector<std::pair<cv::Point, cv::Point>> result;
    for (auto const &p : points) {
        auto pp = vec4iToPointPair(p);
        if (pointBelongsToCenter(center, radius, pp.first)) {
            result.push_back({ pp.first, pp.second });
        } else if (pointBelongsToCenter(center, radius, pp.second)) {
            result.push_back({ pp.second, pp.first });
        }
    }
    return result;
}

int ClockFaceReader::lineLength(std::pair<cv::Point, cv::Point> const &line) {
    int dx = line.first.x - line.second.x;
    int dy = line.first.y - line.second.y;
    return std::sqrt(dx * dx + dy * dy);
}

float ClockFaceReader::lineAngle(std::pair<cv::Point, cv::Point> const &line) {
    int dx = line.first.x - line.second.x;
    int dy = line.first.y - line.second.y;
    float angle = std::atan2(dy, dx) - CV_PI / 2.0;
    if (angle < 0)
        angle += CV_2PI;
    return angle;
}

int ClockFaceReader::angleToHours(float angle) {
    int hours = std::floor(angle / CV_2PI * 12);
    return hours != 0 ? hours : 12;
}

int ClockFaceReader::angleToMinutes(float angle) {
    int minutes = std::round(angle / CV_2PI * 60);
    return minutes != 60 ? minutes : 0;
}

cv::Mat ClockFaceReader::makeMatSquare(int size) {
    return cv::getStructuringElement(cv::MORPH_RECT, { size, size });
}

std::pair<cv::Point, cv::Point> ClockFaceReader::vec4iToPointPair(cv::Vec4i const & vec) {
    return { cv::Point(vec[0], vec[1]), cv::Point(vec[2], vec[3]) };
}
