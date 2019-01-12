#pragma once

#include <opencv2/core.hpp>

class Algorithm {
public:
    Algorithm(cv::Mat const &image, int preferred_size);
    virtual ~Algorithm();

    virtual void execute() = 0;
    virtual std::vector<cv::Mat> const& getSteps() = 0;

    void setDebug(bool enable);

protected:
    float resize(cv::Mat &image);

    void debug(cv::Mat const &image);
    void debugNormalized(cv::Mat const &image);

    cv::Mat const image_;
    int const preferred_size_;

    bool debug_;
    bool executed_;
    std::vector<cv::Mat> debug_steps_;
};
