/** \mainpage HOG_SVM-detection
 * 
 * 
 *   \section splitting_images
 *      1.) Resizes images and crops them into NxM images.\n
 *      2.) Overlap of cropped images is possible. Cropped images are manually reviewed and evaluated if dooor knob is in picture or not.\n
 *      3.) This evaluated cropped pictures are saved into to different folders:\n
 *          "../Data/images/split_images/positiv"\n
 *          "../Data/images/split_images/negativ"
 *   \section HOG
 *      1.) Pushes the positive and negative cropped images in acording image vectors.\n
 *      2.) Computes the HOG descriptor of positive and negative images an writes them in two CVS files:\n
 *      "../Data/positiv"\n
 *      "../Data/negativ"
 * 
 *   \section SVM
 *      1.) The main function opens the CSV files with the hog descriptors for cropped images, split up in two files: negative and positive detection.\n
 *      2.) The data from the csv files is rearranged in a way, the SVM function can compute it.\n
 *      3.) The svm is trained with the data from the csv files.\n
 *      4.) The support vector machine is tested with a test image and a test video.\n
 *      5.) Predicts door handle in the cropped image and draws a blue square over it.\n
 *      6.) Calculates a red rectangle from the minimum and maximum coordinates of the blue rectangles, which includes all recognized cropped images.\n
 *      7.) Red rectangle is the result of the SVM detection.
*/

#include <opencv2/ml.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <vector>
#include <fstream>

/** @brief computes HOG features.

gets HOG descriptors from cropped image.

@param image input image.
@param hog Descriptor
@param cellSize size of cell
@param nbins 
@param blockSize 
*/
std::vector<float> get_HOG_features(cv::Mat image, cv::HOGDescriptor hog, cv::Size cellSize, int nbins, cv::Size blockSize)
{
    cv::Mat img_resized;
    std::vector<float> output;
    hog.compute(image, output, cv::Size(cellSize.height, cellSize.width), cv::Size(0, 0));

    return output;
}

/** @brief 

Predicts door handle in the cropped image and draws a blue square over it.
Calculates a red rectangle from the minimum and maximum coordinates of the blue rectangles, which includes all recognized cropped images.
Red rectangle is the result of the SVM modell.

@param test_image image the SVM ist tested on.
@param svm SVM
@param resize_factor factor to resize image (same as splitting_images.cpp)
*/
void detect(cv::Mat test_image, cv::Ptr<cv::ml::SVM> svm, float resize_factor)
{
    bool detected_something = false;
    cv::namedWindow("Prediction", 0);
    int first_x = 10000;
    int first_y = 10000;
    int last_x = 0;
    int last_y = 0;

    cv::Mat test_image_resized;
    cv::Mat test_image_resized_gray;
    cv::resize(test_image, test_image_resized, cv::Size(), resize_factor, resize_factor);
    cv::cvtColor(test_image_resized, test_image_resized_gray, CV_BGR2GRAY);

    // Split images
    int cropsplit_row = 3;      //defines 3x3 Schnisl
    int cropsplit_col = 2;      //defines 3x3 Schnisl
    int crop_itterator_row = 2; // wie sehr die schnippsö überlappen // ITTERATOR MUSS KLEINER/GLEICH SEIN ALS KLEINSTER CROSSPLIT
    int crop_itterator_col = 2; // wie sehr die schnippsö überlappen

    int rows_crop = int(test_image_resized_gray.rows / cropsplit_row);
    int cols_crop = int(test_image_resized_gray.cols / cropsplit_col);

    int detection_map[cropsplit_row * crop_itterator_row][cropsplit_col * crop_itterator_col];

    for (int i = 0; i < cropsplit_row * crop_itterator_row - crop_itterator_row + 1; i++)
    {
        for (int j = 0; j < cropsplit_col * crop_itterator_col - crop_itterator_col + 1; j++)
        {

            cv::Mat ROI(test_image_resized_gray, cv::Rect(j * cols_crop / crop_itterator_col, i * rows_crop / crop_itterator_row, cols_crop, rows_crop));
            cv::Mat croppedImage;
            ROI.copyTo(croppedImage); // ROI = Region of interesst

            //https://stackoverflow.com/questions/55397301/how-to-detect-an-object-in-an-image-using-hog-descriptors
            //credits: emlot77 / mehr informationen in HOGmain.cpp
            //params
            cv::Size cellSize(8, 8);
            int nbins = 9;
            cv::Size blockSize(2, 2);
            //HOG Struct
            cv::HOGDescriptor hog(cv::Size(croppedImage.cols / cellSize.width * cellSize.width, croppedImage.rows / cellSize.height * cellSize.height),
                                  cv::Size(blockSize.height * cellSize.height, blockSize.width * cellSize.width),
                                  cv::Size(cellSize.height, cellSize.width),
                                  cellSize,
                                  nbins);
            // stackoverflow implementation ende

            std::vector<float> test_HOG_features = get_HOG_features(croppedImage, hog, cellSize, nbins, blockSize);
            float response = svm->predict(test_HOG_features); //predict fuer das bild schnippsl -> response -1 keine türschnalle & +1 tuerschnalle
            if (response == -1)
            {
                std::cout << "NEGATIV | Auf dem Bild ist keine Türschnalle" << std::endl;
            }
            else if (response == 1)
            {
                detected_something = true;
                std::cout << "POSITIV | Auf dem Bild ist eine Türschnalle" << std::endl;

                // DRAW BLUE DETECTION
                cv::Rect r = cv::Rect(j * cols_crop / crop_itterator_col, i * rows_crop / crop_itterator_row, cols_crop, rows_crop);
                cv::rectangle(test_image_resized, r, cv::Scalar(255, 0, 0), 4, 8, 0);
            }
            else
            {
                std::cout << "Something went wrong" << std::endl;
            }

            detection_map[i][j] = response; //remember which schnippsl has a tuerschnalle 
        }
    }

    // DETECTION VISU -> works better if no overlapp in schnippsl 
    for (int i = 0; i < cropsplit_row * crop_itterator_row; i++)
    {
        for (int j = 0; j < cropsplit_col * crop_itterator_col; j++)
        {
            if (detection_map[i][j] < 0)
            {
                std::cout << "-";
            }
            else
            {
                std::cout << "x";
            }
        }
        std::cout << std::endl;
    }

    // DRAW DETECTION
    // min max algorithm for our problem
    // detectionmap contains all schnippsl with a (partial) doorknob
    // we want to finde a rectangle which contains all (partital) doorknobs 
    // we wannt the biggest and smallest x&y koordinate of all blue/partial detections (schnippsl) 
    for (int i = 0; i < cropsplit_row * crop_itterator_row; i++)
    {
        for (int j = 0; j < cropsplit_col * crop_itterator_col; j++)
        {

            int x = j * cols_crop / crop_itterator_col;
            int y = i * rows_crop / crop_itterator_row;
            if (detection_map[i][j] == 1)
            {   
            //get minimum
                if (x < first_x)
                {
                    first_x = x;
                }

                if (y < first_y)
                {
                    first_y = y;
                }
                
            //get max. 
                if (y > last_y)
                {
                    last_y = y;
                }

                if (x > last_x)
                {
                    last_x = x;
                }
            }
        }
    }
    // only draw rec if at least one detection was positiv
    if (detected_something)
    {
        cv::Rect r = cv::Rect(first_x, first_y, last_x - first_x + cols_crop, last_y - first_y + rows_crop);
        cv::rectangle(test_image_resized, r, cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    cv::imshow("Prediction", test_image_resized);
    cv::waitKey(0); //change to 0 for stepping through / 25-30 for "video" 
}

/** @brief 
 *      the main function opens the CSV files with the hog descriptors for cropped images, split up in two files: negative and positive detection.\n
 *      <https://stackoverflow.com/questions/18777267/c-program-for-reading-csv-writing-into-array-then-manipulating-and-printing-i>\n\n
 * 
 *      the data from the csv files is rearranged in a way, the SVM function can compute it.\n
 *      <https://docs.opencv.org/4.5.3/d1/d73/tutorial_introduction_to_svm.html>\n
 *      <https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/ml/introduction_to_svm/introduction_to_svm.cpp>\n
 *      <https://docs.opencv.org/4.5.3/d1/d2d/classcv_1_1ml_1_1SVM.html>\n\n
 * 
 *      the svm is trained with the data from the csv files.\n
 *      <https://docs.opencv.org/4.5.3/db/d7d/classcv_1_1ml_1_1StatModel.html#af96a0e04f1677a835cc25263c7db3c0c>\n\n
 * 
 *      the support vector machine is tested with a test image and a test video.\n
*/
int main()
{   
    float resize_factor = 0.25; // ACHTUNG: Has to be the same as resize_factor in splitting images programm
    // LOAD THE DATA
    // https://stackoverflow.com/questions/18777267/c-program-for-reading-csv-writing-into-array-then-manipulating-and-printing-i
    // https://en.cppreference.com/w/cpp/string/basic_string/stol

    // Get trainingsdata for SVM 
    // Positiv
    std::string line, val;                      /* string for line & value */
    std::vector<std::vector<float>> data_array; /* vector of vector<int>  */
    std::ifstream f_pos;
    std::ifstream f_neg;
    f_pos.open("../Data/positiv.csv");
    f_neg.open("../Data/negativ.csv");

    while (std::getline(f_pos, line))
    {                                    /* read each line */
        std::vector<float> v;            /* row vector v */
        std::stringstream s(line);       /* stringstream line */
        while (getline(s, val, ','))     /* get each value (',' delimited) */
            v.push_back(std::stof(val)); /* add to row vector */
        data_array.push_back(v);         /* add row vector to array */
    }
    int size_positiv = data_array.size();

    while (std::getline(f_neg, line))
    {                                    /* read each line */
        std::vector<float> v;            /* row vector v */
        std::stringstream s(line);       /* stringstream line */
        while (getline(s, val, ','))     /* get each value (',' delimited) */
            v.push_back(std::stof(val)); /* add to row vector */
        data_array.push_back(v);         /* add row vector to array */
    }
    int size_negativ = data_array.size() - size_positiv;

    std::cout << "Size of Array: [" << data_array.size() << "] [" << data_array[0].size() << "]" << std::endl;
    int labels[data_array.size()] = {};
    //fill labels
    for (int i = 0; i < size_positiv; i++)
    {
        labels[i] = 1;
    }
    for (int j = size_positiv; j < (size_negativ + size_positiv); j++)
    { // Continue after positiv entrys
        labels[j] = -1;
    }

    /////////////////////// EINLESEN ENDE //////////////////////////

    // CONVERT TO SVM Compatibility
    // Set up training data
    // https://docs.opencv.org/4.5.3/d1/d73/tutorial_introduction_to_svm.html

    cv::Mat trainingDataMat(data_array.size(), data_array[0].size(), CV_32F); //Change to our data
    for (int i = 0; i < data_array.size(); i++)
    {
        for (int j = 0; j < data_array[0].size(); j++)
        {
            trainingDataMat.at<float>(i, j) = data_array[i][j];
        }
    }

    cv::Mat labelsMat(data_array.size(), 1, CV_32SC1, labels); //Change to our data //look up datatype

    // SVM //
    // https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/ml/introduction_to_svm/introduction_to_svm.cpp
    // https://docs.opencv.org/4.5.3/d1/d2d/classcv_1_1ml_1_1SVM.html //SVM erklärt

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);    // change type for different outcomes: https://docs.opencv.org/4.5.3/d1/d2d/classcv_1_1ml_1_1SVM.html#ab4b93a4c42bbe213ffd9fb3832c6c44f
    svm->setKernel(cv::ml::SVM::LINEAR); //Kernel
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

    //Train
    // https://docs.opencv.org/4.5.3/db/d7d/classcv_1_1ml_1_1StatModel.html#af96a0e04f1677a835cc25263c7db3c0c // trainieren erklärt

    svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat); // TRAIN SVM 
    //Evaluate

    


    ///////////////////////////// TESTING ////////////////////////////

    //Test on new image

    cv::Mat test_image = cv::imread("../Data/test/test.png");
    detect(test_image, svm, resize_factor);

    //video detection

    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    cv::VideoCapture cap("../Data/test/video.mp4");

    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    while (1)
    {

        cv::Mat frame;
        // Capture frame-by-frame
        cap >> frame;
        std::cout << frame.size() << std::endl;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Display the resulting frame
        detect(frame, svm, resize_factor);
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}
