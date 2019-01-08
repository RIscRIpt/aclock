#pragma once

#include <opencv2/core.hpp>

#include "ClockFace.h"

class ClockFaceFinder {
public:
    ClockFaceFinder(cv::Mat const &image);

    void execute();
    std::vector<ClockFace> const& find();
    std::vector<cv::Mat> const& get_steps();

    void setDebug(bool enable);

private:
    void debug(cv::Mat const &image);

    cv::Mat makeStructElemSquare(int size) const;
    cv::Mat makeSturctElemClock(float radius) const;

    bool executed_;
    bool debug_;

    cv::Mat image_;
    std::vector<cv::Mat> debug_steps_;
    std::vector<ClockFace> found_;
};
