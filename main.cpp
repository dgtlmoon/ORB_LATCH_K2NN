#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "LATCH.h"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>


using namespace std;


bool write_file_binary(std::string const &filename, char const *data, size_t const bytes) {
    std::cout << "Saving bytes: " << bytes << std::endl;

    std::ofstream b_stream(filename.c_str(),  std::fstream::out | std::fstream::binary | std::fstream::app);
    if (b_stream) {
        b_stream.write(data, bytes);
        return (b_stream.good());
    }
    return false;
}


int main(int argc, char *argv[]) {
    boost::filesystem::path p(argc > 1 ? argv[1] : "./images");

    // ------------- Configuration ------------
    // detector
    constexpr int numkps = 5000;
    constexpr bool multithread = true;
    // --------------------------------


    if (boost::filesystem::is_directory(p)) {
        std::vector<cv::KeyPoint> keypoints;
        bool got_first = false;
        uint64_t *desc;
        std::vector<KeyPoint> kps;

        std::cout << p << " is a directory containing:\n";

        for (auto &entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {})) {
            std::cout << entry.path().string() << "\n";

            // ------------- Image Read ------------
            cv::Mat image = cv::imread(entry.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
            if (!image.data) {
                std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
                return EXIT_FAILURE;
            }

            // ------------- Detection of key points ------------
            std::cout << std::endl << "Detecting..." << std::endl;
            cv::Ptr<cv::ORB> orb = cv::ORB::create(numkps, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
            orb->detect(image, keypoints);
            // ---------------------------------------------------

            // -------------- Building LATCH ---------------------
            std::cout << std::endl << "LATCHing..." << std::endl;

            desc = new uint64_t[8 * keypoints.size()];
            LATCH<multithread>(image.data, image.cols, image.rows, static_cast<int>(image.step), kps, desc);

            // --------------- Write binary keypoints to file ----
            std::cout << std::endl << "SAVING..... keypoints found " << keypoints.size() << std::endl;

            // Write all keypoint descriptors to a file

            write_file_binary("training.dat",
                              reinterpret_cast<char const *>(desc),
                              (sizeof(uint64_t)*8) * keypoints.size());
            delete[] desc;

            // The first one will be selected as the query/search one
            if (!got_first) {
                got_first = true;
                write_file_binary("query.dat",
                                  reinterpret_cast<char const *>(desc),
                                  (sizeof(uint64_t)*8) * keypoints.size());
            }

        }
    }
}

