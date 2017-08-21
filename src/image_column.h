#ifndef IMAGE_COLUMN
#define IMAGE_COLUMN

#include <iostream>	// for standard I/O
#include <string>   // for strings

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;


class ImgColumn                                                                 // creates & holds Gaussian and Lapacian pyramids 
{
public:
    ImgColumn(int maxLevels,  UMat img);                        // construct and initialize at the start
    ~ImgColumn(){};
    bool    updateColumn(UMat img);                                  // refresh with new frame
    
    int         getMaxLevels() { return maxLevels;}
    CvPoint getFovea() {return fovea;}                               // nb interger coord of fovea in base of pyramid (  x  )
    UMat     getCameraMatrix()	{return cameraMatrix;}
    UMat     getDistCoeffs(){return distCoeffs;}
    Size       getLayerSize() {return layerSize;}                   // set to 16x8 ?
    int         getDepth(){return depth;}
    
    void setMaxLevels(int value) { maxLevels = value;}
    void setFovea(CvPoint value) {fovea=value;}                 // nb interger coord of fovea in base of pyramid (  x  )
    void setCameraMatrix(UMat value)	{cameraMatrix =  value;}
    void setDistCoeffs(UMat value){distCoeffs=value;}
    void setLayerSize(Size value) {layerSize=value;}          // set to 16x8 ?
    void setDepth(int value){depth=value;}

#define X_BASE 2048                                                             // resized base image
#define Y_BASE 1024
#define DEFAULT_LEVELS 7    //8
#define TILE_WIDTH 32           //16
#define TILE_HEIGHT 16          //8
    
protected:
   
private:
    int                         maxLevels;
    CvPoint                 fovea;                                                     // nb interger coord of fovea in base of pyramid (  x  )
    UMat                     undist, scaled;
    vector<UMat>      Pyramid;
    UMat  	                 cameraMatrix;
    UMat  	                 distCoeffs;
    Size                       layerSize;                                               // set to 16x8 ?
    int                         depth = CV_8SC1;
public:
    int                         frameNum;
    
private:
 struct layer{
        vector<UMat> hsvVec, hSharrVec, sSharrVec, vSharrVec;
        void layerNextFrame(UMat frame);
        layer(UMat firstImg);
    };// struct layer ###############################
    
    vector<layer> column;

    class compositeDisplay{
    public:                                                                               //  nb -ve values are shown as zero, so convert to CV_8U before imshow
            UMat tile;                                                                   //  (column.at(0).hsvVec.at(0).size(), CV_8SC3 );                      
            UMat compositeImg;                                                //  ((tile.rows *10), (tile.cols *maxLevels), CV_8SC3);
            vector<UMat> tileChans;
            UMat zeroTileChan;                                                  // .zeros(tile.rows, tile.cols, CV_8S);
            int dtop, dbottom, dleft, dright=0;                        // ROI limits for compositeImg
            int rows ;
            int cols ;
            
            compositeDisplay();
        } compDisplay;
    
    void prepFrame(UMat& frame);
    
public:
    void ImgColNextFrame(UMat nextFrame);

public:
    ImgColumn(UMat firstFrame, InputArray  newCameraMatrix, InputArray  newDistCoeffs, int layers=DEFAULT_LEVELS);//ImgColumn constructor

private:
        void insertTile(int _x_pos, int &_y_pos, int _tileWidth, int _tileHeight);

public:
    UMat showColumn( );
    
};// Class ImgColumn  ########################################

# endif  /*IMAGE_COLUMN*/
