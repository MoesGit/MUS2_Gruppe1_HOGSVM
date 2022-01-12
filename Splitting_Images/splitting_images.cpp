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
#include <string>     

/** @brief Resizes images and crops them into NxM images.\n
 * Overlap of cropped images is possible. Cropped images are manually reviewed and evaluated if dooor knob is in picture or not.\n
 * This evaluated cropped pictures are saved into to different folders:\n
 *  "../Data/images/split_images/positiv"\n
 *  "../Data/images/split_images/negativ"\n
 * 
 * 
 * 
 * 
*/
int main(){

float resize_factor = 0.25; // HAS TO MATCH WITH SVM 

// get all images via GLOB
// https://stackoverflow.com/questions/31346132/how-to-get-all-images-in-folder-using-c

    std::vector<cv::String> fn;

    cv::glob("../Data/images/all_images/*.png", fn, false); //Achtung bildformat beachten .png
    std::vector<cv::Mat> images;
    size_t count = fn.size(); //number of png files in images folder

    std::cout << "We got " << count << " images in folder" << std::endl;
    int cropsplit_row = 3; //defines 3x3 Schnisl  //-> "Seitenverhaeltnis" von Ausschnit 
    int cropsplit_col = 2; //defines 3x3 Schnisl  
    int crop_itterator_row = 2; // wie sehr die schnippsl ueberlappen // ITTERATOR MUSS KLEINER/GLEICH SEIN ALS KLEINSTER CROSSPLIT 
    int crop_itterator_col = 2; // wie sehr die schnippsl ueberlappen 
    int cnt_operations = 0; // Counter fuer Schnippsl beschriftung 
    cv::namedWindow("test", 0);


    for (size_t i=0; i<count; i++){ // load each image
    std::cout << "new image started" << std::endl; 
        cv::Mat tmp_img; 
        cv::Mat tmp_img_resized;
        tmp_img = imread(fn[i], CV_LOAD_IMAGE_GRAYSCALE);
        cv::resize(tmp_img, tmp_img_resized, cv::Size(), resize_factor, resize_factor); //resize image 
        
        int rows_crop = int(tmp_img_resized.rows / cropsplit_row); 
        int cols_crop = int(tmp_img_resized.cols / cropsplit_col);

        //std:: cout << rows_crop << "  " << cols_crop << std::endl; 
        for (int i=0; i<cropsplit_row * crop_itterator_row  - crop_itterator_row+1 ; i++){
            for (int j=0; j<cropsplit_col * crop_itterator_col -crop_itterator_col+1  ; j++){ 
                
                //std::cout << "koord cols: " << j*cols_crop/crop_itterator_col + cols_crop << std::endl;
                //std::cout << "koord rows: " << i*rows_crop/crop_itterator_row + rows_crop<< std::endl;

                cv::Mat ROI(tmp_img_resized, cv::Rect(j*cols_crop/crop_itterator_col,i*rows_crop/crop_itterator_row,cols_crop,rows_crop)); // Get Region of Intrest
                cv::Mat croppedImage;
                ROI.copyTo(croppedImage);
                // input fürs bewerten croppedImage
                cv::imshow("test", croppedImage);
                char k = cv::waitKey(0);
                if(k == 'y'){
                    //std::cout << "image was a doorknob" << std::endl;
                    cv::imwrite("../Data/images/split_images/positiv/" + std::to_string(cnt_operations) + ".JPG", croppedImage); //used to safe image
                }else{
                    //std::cout << "image was NOT a doorknob" << std::endl;
                    cv::imwrite("../Data/images/split_images/negativ/" + std::to_string(cnt_operations) + ".JPG", croppedImage); //used to safe image
                }
                cnt_operations++;
            }

        } 


        std::cout << "Image " << i << "from " << count << "done" << std::endl;     
    } 
//All good
return 0;
}
