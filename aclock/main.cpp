#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ClockFaceFinder.h"

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        char const *program = argc > 0 ? argv[0] : "aclock";
        std::cout << "Usage: " << argv[0] << " image.png\n"
            << "Image not specified, trying to open `image.png`\n";
    }
    auto image = cv::imread("image3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Image not found.\n";
        return 1;
    }

    ClockFaceFinder cff(image, 0.5f, 128);
    cff.setDebug(true);
    auto steps = cff.getSteps();
    for (auto const &step : steps) {
        cv::imshow("Debug", step);
        cv::waitKey();
    }
    cv::destroyAllWindows();
    cv::imshow("Result", cff.getMaskedImage());
    while (cv::waitKey() != 27)
        ;

	return 0;
}
