#pragma once

#include <optional>

#include <opencv2/core.hpp>

#include "Algorithm.h"

class ClockFaceReader : public Algorithm {
public:
    ClockFaceReader(cv::Mat const &image, int preferred_size);

    virtual void execute();
    virtual std::vector<cv::Mat> const& getSteps();
    std::optional<std::pair<int, int>> getTime();
    std::optional<cv::Mat> getDetectedTimeImage();

private:
    void preprocess(cv::Mat &image);

    static bool pointBelongsToCenter(cv::Point const &center, int radius, cv::Point const &p);
    static std::vector<std::pair<cv::Point, cv::Point>> filterLines(cv::Point const &center, int radius, std::vector<cv::Vec4i> const &points);
    static void scaleClockHands(std::vector<std::pair<cv::Point, cv::Point>> &clock_hands, float factor);
    static int lineLength(std::pair<cv::Point, cv::Point> const &line);
    static float lineAngle(std::pair<cv::Point, cv::Point> const &line);

    static int angleToHours(float angle);
    static int angleToMinutes(float angle);

    static cv::Mat makeMatSquare(int size);

    static std::pair<cv::Point, cv::Point> vec4iToPointPair(cv::Vec4i const &vec);

    std::pair<int, int> time_;
    std::vector<std::pair<cv::Point, cv::Point>> clock_hands_;
};
