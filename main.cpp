#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <bitset>
#include "LATCH.h"
#include "K2NN.h"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/program_options.hpp>


using namespace std::chrono;


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


void index(const char *path) {

    std::cout << "Scanning...\n";

    boost::filesystem::path p(path);

    // ------------- Configuration ------------
    // detector
    constexpr int numkps = 200;
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
                std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
                return;
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

void search(std::string search_type) {
    constexpr int warmups = 10;
    constexpr int runs = 25;
    constexpr int size = 10000;
    constexpr int threshold = 20;
    constexpr int max_twiddles = 20;
    // --------------------------------


    std::vector<uint8_t> qvecs(64 * size);
    std::vector<uint8_t> tvecs(64 * size);


//        std::ofstream tout("tvecs.dat", std::ios::out | std::ios::binary);
//        tout.write((char*)&tvecs[0], tvecs.size() * sizeof(uint8_t));
//        tout.close();

//        std::ofstream qout("qvecs.dat", std::ios::out | std::ios::binary);
//        qout.write((char*)&qvecs[0], qvecs.size() * sizeof(uint8_t));
//        qout.close();


    std::ifstream tin("training.dat", std::ios::in | std::ios::binary);
    tin.read(reinterpret_cast<char *>(&tvecs[0]), sizeof(uint8_t) * 64 * size);
    tin.close();

    std::ifstream qin("query.dat", std::ios::in | std::ios::binary);
    qin.read(reinterpret_cast<char *>(&qvecs[0]), sizeof(uint8_t) * 64 * size);
    qin.close();


    printf("%.2X", tvecs[0]);
    printf("%.2X", qvecs[0]);

    // --------------------------------

    // Initialization of Matcher class
    Matcher<false> m(tvecs.data(), size, qvecs.data(), size, threshold, max_twiddles);

    // @todo run in different thread so its already 'warmed up'
    std::cout << std::endl << "Warming up..." << std::endl;
    high_resolution_clock::time_point start;

    if(!search_type.compare("fast-approx")) {
        for (int i = 0; i < warmups; ++i) m.fastApproxMatch();
        std::cout << "Testing fast approximate matches..." << std::endl;
        start = high_resolution_clock::now();
        m.fastApproxMatch();
    } else {
        for (int i = 0; i < warmups; ++i) m.bruteMatch();
        std::cout << "Testing brute-force..." << std::endl;
        start = high_resolution_clock::now();
        m.bruteMatch();
    }

    high_resolution_clock::time_point end = high_resolution_clock::now();

    const double sec =
            static_cast<double>(duration_cast<nanoseconds>(end - start).count()) * 1e-9 / static_cast<double>(runs);
    std::cout << std::endl << search_type << " K2NN found " << m.matches.size() << " matches in " << sec * 1e3 << " ms"
              << std::endl;
    std::cout << "Throughput: " << static_cast<double>(size) * static_cast<double>(size) / sec * 1e-9
              << " billion comparisons/second." << std::endl << std::endl;

    for (auto &&match : m.matches) {
        std::cout << match.t << std::endl;

    }

}


int main(int argc, char **argv) {
    std::string index_path;
    std::string search_type;

    try {
        /** Define and parse the program options
         */
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
                ("help", "Print help messages")
                ("index-path",  po::value<std::string>(&index_path), "/path/to; Create an index/training.dat file from images and a simple file with a single image query.dat set")
                ("search",  po::value<std::string>(&search_type), "fast-approx or brute-force; Search training.dat for what is present in query.dat");

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
                index(cstr);
                return 1;
            }


            if (vm.count("search")) {
                search(vm["search"].as<std::string>());
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