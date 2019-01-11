#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "ClockFace.h"

class ClockFaceFinder {
public:
    ClockFaceFinder(cv::Mat const &image, float min_clock_alignment = 0.5f, int preferred_size = 128);

    void execute();
    ClockFace const& find();
    std::vector<cv::Mat> const& getSteps();
    cv::Mat getMaskedImage();

    void setDebug(bool enable);

private:
    void preprocess(cv::Mat &image);
    float resize(cv::Mat &image);
    void find_clock(cv::Mat const &image, float resize_factor, float min_radius, float max_radius, float radius_step);

    void debug(cv::Mat const &image);
    void debugNormalized(cv::Mat const &image);

    void initializeProgress(float step);
    void stepProgress();

    cv::Mat makeMatSquare(int size) const;
    cv::Mat makeMatClock(float radius) const;

    float const min_clock_alignment_;
    int const preferred_size_;

    bool executed_;
    bool debug_;

    cv::Mat image_;
    std::vector<cv::Mat> debug_steps_;
    ClockFace found_clock_;

    float total_progress_;
    float progress_step_;
    float const max_progress_;
};
