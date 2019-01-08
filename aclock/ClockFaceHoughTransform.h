#pragma once

#include <opencv2/core.hpp>

class ClockFaceHoughTransform {
public:

private:
    cv::Mat accumulator_;
    std::vector<std::vector<cv::Point>> rTable_;
};
