#pragma once

#include <opencv2/core.hpp>

class Gradient {
public:
    Gradient(cv::Mat const &image, int ddepth = CV_32F);

    inline cv::Mat const& x() const noexcept { return x_; }
    inline cv::Mat const& y() const noexcept { return y_; }

    cv::Mat directionInRadians();
    cv::Mat directionInDegrees();
    cv::Mat magnitude();
    cv::Mat visualize();

private:
    cv::Mat direction(bool angleInDegrees = false) const;

    int ddepth_;
    cv::Mat x_, y_;
    cv::Mat directionInRadians_;
    cv::Mat directionInDegrees_;
    cv::Mat magnitude_;
    cv::Mat visualization_;

};
