#ifndef TRACKING_AND_MAPPING
#define TRACKING_AND_MAPPING

///   tracking and mapping  ////

#include <iostream>	// for standard I/O
#include <string>   // for strings

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

#include "image_column.h"

#define    NUM_IMCOLS 6 //

using namespace std;
using namespace cv;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////  structs and classes for tracking and mapping //////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Maps { // to be constructed from the current image, to be used to generate the predictions of the next frame
  vector<UMat> depth, normal, reflectance, illumination, shading, transform;  
    Maps();
};

struct PredictionImageColumn{
        struct Layer{
            vector<UMat> hsvVec, hSharrVec, sSharrVec, vSharrVec;   // using same members names as ImageColumn
            Layer();
        };
        vector<Layer> column;
        PredictionImageColumn();
};
// moved to class App
// vector<PredictionImageColumn> prediction(NUM_IMCOLS );   
//numPredictions i.e. on-target and cluster around it.

struct CostVolume {  // defined in terms of the preceeing image, and 
    
    CostVolume(int numPredictions/*?*/, int layers = DEFAULT_LEVELS/* MAX_LAYERS*/ );
};


void calcTfmCost(PredictionImageColumn Mapping,  CostVolume &costVol,  ImgColumn imcol /*= imCols.at(current)*/ , ImgColumn keyframe /*=  imCols.at(0)*/ );

void calcModelCost();

void predictNextFrame();



#endif /*TRACKING_AND_MAPPING*/
