#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <conio.h>

#include "ClockFaceFinder.h"
#include "ClockFaceReader.h"

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        char const *program = argc > 0 ? argv[0] : "aclock";
        std::cout << "Usage: " << argv[0] << " image.png\n"
            << "Image not specified, trying to open `image.png`\n";
    }
    auto image = cv::imread("image.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Image not found.\n";
        return 1;
    }

    ClockFaceFinder cff(image, 0.75f, 128);
    cff.setDebug(true);
    auto steps = cff.getSteps();
    for (auto const &step : steps) {
        cv::imshow("Debug", step);
        cv::waitKey();
    }
    cv::destroyAllWindows();
    if (auto clock = cff.getMaskedImage(); clock.has_value()) {
        cv::imshow("Result", *clock);
        while (cv::waitKey() != 27)
            ;

        cv::destroyAllWindows();

        ClockFaceReader cfr(*clock, 128);
        cfr.setDebug(true);
        auto steps = cfr.getSteps();
        for (auto const &step : steps) {
            cv::imshow("Debug", step);
            cv::waitKey();
        }
        cv::destroyAllWindows();

        auto time = cfr.getTime();
        std::cout << time.first << ':' << time.second << '\n';
        getch();
    }

	return 0;
}
