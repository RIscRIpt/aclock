#pragma once

#include <string>
#include <optional>

#include <opencv2/core.hpp>

#include "Algorithm.h"
#include "ClockFace.h"

class ClockFaceFinder : public Algorithm {
public:
    ClockFaceFinder(cv::Mat const &image, float min_clock_alignment = 0.5f, int preferred_size = 128);

    virtual void execute();
    virtual std::vector<cv::Mat> const& getSteps();
    std::optional<ClockFace> const& find();
    std::optional<cv::Mat> getMaskedImage();

private:
    void preprocess(cv::Mat &image);
    void find_clock(cv::Mat const &image, float resize_factor, float min_radius, float max_radius, float radius_step);

    cv::Mat makeMatSquare(int size) const;
    cv::Mat makeMatClock(float radius) const;

    float const min_clock_alignment_;

    bool found_;
    ClockFace found_clock_;
};
