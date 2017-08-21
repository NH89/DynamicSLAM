///   tracking and mapping  ////

#include <iostream>                      // for standard I/O
#include <string>                           // for strings

#include <opencv2/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>   // OpenCV window I/O

#include "image_column.h"
#include "tracking_and_mapping.h"

using namespace std;
using namespace cv;


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////  structs and classes for tracking and mapping //////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Maps::Maps(){                                                                                   // set type, channels, and initial values      ??? which UMat CV_ types should I use  ????
        for(int i=0;i<DEFAULT_LEVELS;i++){
            depth.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_32FC1);   
            normal.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_32FC3);
            reflectance.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_8S);
            illumination.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_8S);
            shading.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_8S);
            transform.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_8S);
        }       
}
    
    
PredictionImageColumn::Layer::Layer(){                                         // ?? plan to refactor code with this as a base, and former as derived srtuct.
                UMat zeroUMat;                                                               // used to zero column layers
                zeroUMat.zeros(TILE_HEIGHT, TILE_WIDTH, CV_8S);
                hsvVec.reserve(3);
                hSharrVec.reserve(2);
                sSharrVec.reserve(2);
                vSharrVec.reserve(2);
                for(int i=0;i<3;++i){  hsvVec.push_back(zeroUMat); }
                for(int i=0;i<2;++i){
                    hSharrVec.push_back(zeroUMat); 
                    sSharrVec.push_back(zeroUMat); 
                    vSharrVec.push_back(zeroUMat);
                }
};
        
 PredictionImageColumn::PredictionImageColumn() {
        Layer zeroedLayer;
        column.reserve(DEFAULT_LEVELS);
        for(int i=0;i<DEFAULT_LEVELS;i++){
            column.push_back(zeroedLayer);
        }
};
        




CostVolume::CostVolume(int numPredictions/*?*/, int layers /*=  DEFAULT_LEVELS*/ /* MAX_LAYERS*/ ){
        
    }
//}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void calcTfmCost(ImgColumn Mapping,  CostVolume &costVol, ImgColumn imcol /*= imCols.at(current) */, ImgColumn keyframe/* =  imCols.at(0)*/ ){
    
    
}

void calcModelCost(){
  // find Fisher Information cost of complete scene model,  weighed against mutal_information between image_frames and model_predictions  
  
    // can it be found locally ?  I think so.
    
    // used to evaluate changes of depth, reflectance and illumination maps.
    
}

void predictNextFrame(){
    
    // select previous best prediction 
        // compare current ImCol to predictions
        
        // refine prediction 
            // NB if 
    
    // compute mappings for new predictions : based on previous frame-to-frame transforms
    
    // compute frame predictions : ImCol of volumes 
    
}

