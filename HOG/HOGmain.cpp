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


/** @brief creates CVS file with all HOG descriptos of cropped images.

@param name file name of CVS
@param count count amount of descriptors
@param images vector of cropped images
@param hog HOG
@param cellSize
@param blockSize 
*/
void create_HOG_CSV(std::string name, size_t count, std::vector<cv::Mat> images, cv::HOGDescriptor hog, cv::Size cellSize, int nbins, cv::Size blockSize){

    std::ofstream myfile;
    std::string file_name = name + ".csv";
    myfile.open(file_name);
    for (size_t i=0; i<count; i++){ //Loop over all images
        std::vector< float > output;
        
        //cv::imshow("test", images[i]);
        //cv::waitKey(0);
        hog.compute(images[i],output, cv::Size(cellSize.height,cellSize.width), cv::Size( 0, 0 )); //compute HOG for every image 

        //WRITE TO CSV
            if ( output.size() != 0){
            std::cout << "HOG FEATURES: for image #" << i << ": "<< output.size() << std::endl;

                for (int i = 0; i < output.size(); i++) {
                    myfile << output.at(i) << ',';
                }
                myfile << "\n";
                

            }else{
                std::cout << "no hog features" << std::endl; 
                break;
            }

    }
    myfile.close(); 

 // Done with csv feature table
}

/** @brief pushes the positive and negative cropped images in acording image vectors.
 * 
 * Computes the HOG descriptor of positive and negative images an writes them in two CVS files:\n
 * "../Data/positiv"\n
 * "../Data/negative"\n\n
 * 
 *  https://stackoverflow.com/questions/55397301/how-to-detect-an-object-in-an-image-using-hog-descriptors\n
 *  https://stackoverflow.com/questions/31346132/how-to-get-all-images-in-folder-using-c\n
*/
int main(){
float resize_factor = 1.0; //Has to be 1.0 or change resize in SVM to resize_hog*resize_splittingImages

// get all images via GLOB
// https://stackoverflow.com/questions/31346132/how-to-get-all-images-in-folder-using-c


//Positiv Images
    std::vector<cv::String> fn_pos;
    cv::glob("../Data/images/split_images/positiv/*.JPG", fn_pos, false); //data format is defined by output of splitting_images
    std::vector<cv::Mat> images_pos; //Positive Bilder 
    size_t count_pos = fn_pos.size(); //number of files in images folder

    std::cout << "We got " << count_pos << " images in folder" << std::endl;

    for (size_t i=0; i<count_pos; i++){ // load each image and scale
        cv::Mat tmp_img; 
        cv::Mat tmp_img_resized;
        tmp_img = imread(fn_pos[i], CV_LOAD_IMAGE_GRAYSCALE); //Achtung als Grayscale einlesen
        cv::resize(tmp_img, tmp_img_resized, cv::Size(), resize_factor, resize_factor);
        images_pos.push_back(tmp_img_resized);
    } 


// Negative IMages

    std::vector<cv::String> fn_neg;
    cv::glob("../Data/images/split_images/negativ/*.JPG", fn_neg, false); //data format is defined by output of splitting_images
    std::vector<cv::Mat> images_neg; //Negative Bilder 
    size_t count_neg = fn_neg.size(); //number of files in images folder

    std::cout << "We got " << count_neg << " images in folder" << std::endl;

    for (size_t i=0; i<count_neg; i++){ // load each image and scale
        cv::Mat tmp_img; 
        cv::Mat tmp_img_resized;
        tmp_img = imread(fn_neg[i], CV_LOAD_IMAGE_GRAYSCALE); //Achtung als Grayscale einlesen
        cv::resize(tmp_img, tmp_img_resized, cv::Size(), resize_factor, resize_factor);
        images_neg.push_back(tmp_img_resized);
    } 

// Aufbereitung ende



//HOG Implementation for small sized images
//https://stackoverflow.com/questions/55397301/how-to-detect-an-object-in-an-image-using-hog-descriptors
/* Man fragt sich hier vielleicht wieso diese Merkwuerdige HOG implementierung gewaehlt wurde? 
* -> da das Projekt mit sich gewachsen ist und HOG eine gewisse "Sliding-Window" groesse in Abhaengigkeit zum input Bild
* benoetigt haben wir uns fuer die dynamische implementierung entschieden. Natuerlich koennte man einfach sich auf eine Bildergroese einigen aber da
* unser Datensatz sich immer geaendert hat war das die beste loesung.
* Credits gehen raus an emlot77 von Stackoverflow :D 
*/

// emlot77 implementation angepasst auf unser problem: 
//params
    cv::Size cellSize(8,8);
    int nbins= 9;
    cv::Size blockSize(2,2);
//HOG Struct 
    cv::HOGDescriptor hog(cv::Size(images_pos[0].cols/cellSize.width*cellSize.width,images_pos[0].rows/cellSize.height*cellSize.height),
                    cv::Size(blockSize.height*cellSize.height,blockSize.width*cellSize.width),
                    cv::Size(cellSize.height,cellSize.width),
                    cellSize,
                    nbins);
// stackoverflow implementation ende



 create_HOG_CSV("../Data/positiv", count_pos,images_pos, hog, cellSize, nbins, blockSize); //Computes the actuall hog features and safes them for +1
 create_HOG_CSV("../Data/negativ", count_neg,images_neg, hog, cellSize, nbins, blockSize); //Computes the actuall hog features and safes them for -1


//All good - hog done
return 0;
}
