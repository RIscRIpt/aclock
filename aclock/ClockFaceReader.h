#pragma once

#include <opencv2/core.hpp>

#include "Algorithm.h"

class ClockFaceReader : public Algorithm {
public:
    ClockFaceReader(cv::Mat const &image, int preferred_size);

    virtual void execute();
    virtual std::vector<cv::Mat> const& getSteps();
    std::pair<int, int> getTime();

private:
    void preprocess(cv::Mat &image);

    int determineInnerRadius(cv::Mat const &image);
    std::pair<int, int> countNonZeroOnRadius(cv::Mat const &image, cv::Point const &center, float angle, int radius, int offset = 0) const;

    int angleToHours(float angle) const;
    int angleToMinutes(float angle) const;

    cv::Mat makeMatSquare(int size) const;

    std::pair<int, int> time_;
};
