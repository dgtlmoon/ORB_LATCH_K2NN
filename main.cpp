#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <bitset>
#include "LATCH.h"
#include "K2NN.h"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/program_options.hpp>

constexpr int numkps = 1000;


using namespace std;
namespace po = boost::program_options;

bool write_file_binary(std::string const &filename, char const *data, size_t const bytes) {
    std::cout << "Saving bytes: " << bytes << std::endl;

    std::ofstream b_stream(filename.c_str(), std::fstream::out | std::fstream::binary | std::fstream::app);
    if (b_stream) {
        b_stream.write(data, bytes);
        return (b_stream.good());
    }
    return false;
}


void index_from_path(const char *path) {

    std::cout << "Scanning...\n";

    boost::filesystem::path p(path);

    // ------------- Configuration ------------
    // detector

    constexpr bool multithread = true;
    // --------------------------------

    int n = 0;

    if (boost::filesystem::is_directory(p)) {
        std::vector<cv::KeyPoint> keypoints;
        uint64_t *descriptor;
        std::vector<KeyPoint> kps;

        std::cout << p << " is a directory containing:\n";

        for (auto &entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {})) {
            n++;
            kps.clear();
            std::cout << entry.path().string() << "\n";

            // ------------- Image Read ------------
            cv::Mat image = cv::imread(entry.path().string(), CV_LOAD_IMAGE_GRAYSCALE);
            if (!image.data) {
                std::cerr << "ERROR: failed to open image. Skipping." << std::endl;
                continue;
            }

            // ------------- Detection of key points ------------
            std::cout << std::endl << "Detecting..." << std::endl;
            cv::Ptr<cv::ORB> orb = cv::ORB::create(numkps, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
            orb->detect(image, keypoints);
            // ---------------------------------------------------

            // -------------- Building LATCH ---------------------
            std::cout << std::endl << "LATCHing..." << std::endl;

            descriptor = new uint64_t[8 * keypoints.size()];
            // OpenCV requires radians not degrees
            for (auto &&kp : keypoints) kps.emplace_back(kp.pt.x, kp.pt.y, kp.size, kp.angle * 3.14159265f / 180.0f);

            LATCH<multithread>(image.data, image.cols, image.rows, static_cast<int>(image.step), kps, descriptor);


            // --------------- Write binary keypoints to file ----
            std::cout << std::endl << "SAVING..... keypoints found " << keypoints.size() << std::endl;


            // The first one will be selected as the query/search one
            if (n == 3) {
                for (int i = 0; i < 8; ++i) std::cout << std::bitset<64>(descriptor[i]) << std::endl;
                std::cout << std::endl << " ************* SAVING QUERY ********" << keypoints.size() << std::endl;
                write_file_binary("query.dat",
                                  reinterpret_cast<char const *>(descriptor),
                                  (sizeof(uint64_t) * 8) * keypoints.size());
            }


            // Write all keypoint descriptors to a file

            write_file_binary("training.dat",
                              reinterpret_cast<char const *>(descriptor),
                              (sizeof(uint64_t) * 8) * keypoints.size());

            delete[] descriptor;

        }
    }
}

void search() {
    constexpr int threshold = 5;
    constexpr int max_twiddles = 200;
    constexpr int warmups = 5;

    int start;
    int i;

    // --------------------------------
//        std::ofstream tout("tvecs.dat", std::ios::out | std::ios::binary);
//        tout.write((char*)&tvecs[0], tvecs.size() * sizeof(uint8_t));
//        tout.close();

//        std::ofstream qout("qvecs.dat", std::ios::out | std::ios::binary);
//        qout.write((char*)&qvecs[0], qvecs.size() * sizeof(uint8_t));
//        qout.close();


    // Loading training data
    std::ifstream tin("training.dat", std::ios::in | std::ios::binary | std::ios::ate);
    int tsize = tin.tellg();
    tin.seekg(0);
    std::vector<uint8_t> tvecs(tsize);
    tin.read(reinterpret_cast<char *>(&tvecs[0]), tsize);
    tin.close();

    // Load query data
    std::ifstream qin("query.dat", std::ios::in | std::ios::binary | std::ios::ate);
    int qsize = qin.tellg();
    qin.seekg(0);
    std::vector<uint8_t> qvecs(qsize);
    qin.read(reinterpret_cast<char *>(&qvecs[0]), qsize);
    qin.close();


    // --------------------------------


    // Initialization of Matcher class
    // number for vector size size is /64, this is because it needs to know how many descriptors
    Matcher<false> m(tvecs.data(), tsize / 64, qvecs.data(), qsize / 64, threshold, max_twiddles);

    // -----------------------------------------------------------------------------------
    std::cout << "---- fastApproxMatch results ----" << std::endl;
    std::cout << "Warming up..." << std::endl;
    for (i = 0; i < warmups; ++i) {
        m.fastApproxMatch();
        std::cout << ".";
    }
    std::cout << std::endl;
    m.fastApproxMatch();

    i = 0;
    start = m.matches[0].t;
    for (auto &&match : m.matches) {
        if (start == match.t) {
            i++;
        } else {
            std::cout << "Non consecutive reference found at " << match.t << " -> " << start << std::endl;
        }
        start++;
    }
    if ((unsigned) i == m.matches.size()) {
        std::cout << i << " consecutive results found" << std::endl;
    }

    // -----------------------------------------------------------------------------------
    std::cout << "---- bruteMatch results ----" << std::endl;
    std::cout << "Warming up..." << std::endl;
    for (i = 0; i < warmups; ++i) {
        m.bruteMatch();
        std::cout << ".";
    }
    std::cout << std::endl;

    m.bruteMatch();

    i = 0;
    start = m.matches[0].t;
    for (auto &&match : m.matches) {
        if (start == match.t) {
            i++;
        } else {
            std::cout << "Non consecutive reference found at " << match.t << " -> " << start << std::endl;
        }
        start++;
    }
    if ((unsigned) i == m.matches.size()) {
        std::cout << i << " consecutive results found" << std::endl;
    }

    // -----------------------------------------------------------------------------------
    std::cout << "---- exactMatch results ----" << std::endl;
    std::cout << "Warming up..." << std::endl;
    for (i = 0; i < warmups; ++i) {
        m.exactMatch();
        std::cout << ".";
    }
    std::cout << std::endl;
    m.exactMatch();
    i = 0;
    start = m.matches[0].t;
    for (auto &&match : m.matches) {
        if (start == match.t) {
            i++;
        } else {
            std::cout << "Non consecutive reference found at " << match.t << " -> " << start << std::endl;
        }
        start++;
    }
    if ((unsigned) i == m.matches.size()) {
        std::cout << i << " consecutive results found" << std::endl;
    }


}


int main(int argc, char **argv) {

    std::string index_path;
    std::string search_type;
    std::cout.setf(std::ios::unitbuf);

    try {
        /** Define and parse the program options
         */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
                ("help", "Print help messages")
                ("index-path",  po::value<std::string>(&index_path), "</path/to>; Create an index/training.dat file from images and a simple file with a single image query.dat set")
                ("search", "Perform various searches");

        po::variables_map vm;
        try {
            po::store(po::parse_command_line(argc, argv, desc),
                      vm); // can throw

            /** --help option
             */
            if (vm.count("help")) {
                std::cout << "Basic Command Line Parameter App" << std::endl
                          << desc << std::endl;
                return 1;
            }


            if (vm.count("index-path")) {
                std::cout << "Indexing images in " << vm["index-path"].as<std::string>()<< std::endl;
                const char *cstr = vm["index-path"].as<std::string>().c_str();
                index_from_path(cstr);
                return 1;
            }


            if (vm.count("search")) {
                search();
                return 1;
            }


            po::notify(vm); // throws on error, so do after help in case
            // there are any problems
        }
        catch (po::error &e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << desc << std::endl;
            return 0;
        }

        // application code here //

    }
    catch (std::exception &e) {
        std::cerr << "Unhandled Exception reached the top of main: "
                  << e.what() << ", application will now exit" << std::endl;
        return 0;

    }

    return 1;

}