#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "ClockFace.h"

class ClockFaceFinder {
public:
    ClockFaceFinder(cv::Mat const &image);

    void execute();
    std::vector<ClockFace> const& find();
    std::vector<cv::Mat> const& getSteps();
    cv::Mat getMaskedImage();

    void setDebug(bool enable);

private:
    void debug(cv::Mat const &image);
    void debugNormalized(cv::Mat const &image);

    void initializeProgress(float step);
    void reportProgress(char const *fmt, ...);

    cv::Mat makeMatSquare(int size) const;
    cv::Mat makeMatClock(float radius) const;

    bool executed_;
    bool debug_;

    cv::Mat image_;
    std::vector<cv::Mat> debug_steps_;
    std::vector<ClockFace> found_;

    float total_progress_;
    float progress_step_;
    float const max_progress_;
};
