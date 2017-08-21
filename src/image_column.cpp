///  Image collumn ///

#include <iostream>	// for standard I/O
#include <string>   // for strings

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

#include "image_column.h"

using namespace std;
using namespace cv;


void ImgColumn::layer::layerNextFrame(UMat frame){
            split(frame, hsvVec);

            Scharr(hsvVec[0], hSharrVec[0], CV_8S,1,0); // hue x gradient
            Scharr(hsvVec[0], hSharrVec[1], CV_8S,0,1); // hue y gradient
            
            Scharr(hsvVec[1], sSharrVec[0], CV_8S,1,0); // saturation x gradient
            Scharr(hsvVec[1], sSharrVec[1], CV_8S,0,1); // saturation y gradient
            
            Scharr(hsvVec[2], vSharrVec[0], CV_8S,1,0); // value x gradient
            Scharr(hsvVec[2], vSharrVec[1], CV_8S,0,1); // value y gradient
        }
        
ImgColumn::layer::layer(UMat firstImg)
        {
            UMat zeroUMat; // used to zero column layers
            zeroUMat.zeros(firstImg.size[0],firstImg.size[1] , CV_8S);
       
            for(int i=0;i<3;++i){  hsvVec.push_back(zeroUMat); }

            for(int i=0;i<2;++i){
                hSharrVec.push_back(zeroUMat); 
                sSharrVec.push_back(zeroUMat); 
                vSharrVec.push_back(zeroUMat);
            }
            layerNextFrame(firstImg);
        }
        
        
ImgColumn::compositeDisplay::compositeDisplay(){
                tile=UMat::zeros(TILE_HEIGHT,  TILE_WIDTH, CV_8SC3);
                zeroTileChan=UMat::zeros(TILE_HEIGHT,  TILE_WIDTH, CV_8S);          // +1 magin around tiles, +2 l&r margins in cols, 128 blue background
                compositeImg=UMat(  ((TILE_HEIGHT+1)*10),  ((TILE_WIDTH+1) * (DEFAULT_LEVELS/*+1*/) +2),  CV_8UC3, 128  ); 
                this->tileChans.reserve(3);
                for(int i=0;i<3;i++) { this->tileChans.push_back(UMat::zeros(TILE_HEIGHT,  TILE_WIDTH, CV_8S)); }                  
                this->rows = TILE_WIDTH;
                this->cols = TILE_HEIGHT;
            }

            
void ImgColumn::prepFrame(UMat& frame)
    {
        resize(frame, scaled, Size(X_BASE ,Y_BASE) );
        undist = scaled.clone();
        undistort(scaled, undist, cameraMatrix, distCoeffs);
        buildPyramid(undist /*frame*/, Pyramid, maxLevels);
    }//prepFrame 
    
    

void ImgColumn::ImgColNextFrame(UMat nextFrame)        // fixed central fovea version
    {
        frameNum++ ;
        prepFrame(nextFrame);                                   // resize, undistort, buildPyramid
        int width =  TILE_WIDTH;                                //X_IMG_COL;// size of UMat in each column layer
        int height = TILE_HEIGHT ;
        CvPoint layerFovea = fovea;
        UMat ROI;
        int x = layerFovea.x - (width/2);
        int y = layerFovea.y  - (height/2);
        
       for(int layer=0;layer<maxLevels;layer++)
       {
           Rect RectangleToSelect(x,y,width,height);  // x,y = top left corner of rect
           ROI=Pyramid.at(layer)(RectangleToSelect);
           column.at(layer).layerNextFrame(ROI);           // copy ROI to imageCollumn
           layerFovea.x = layerFovea.x/2;                       // assumes each Pyramid layer is 1/2 previous size
           layerFovea.y = layerFovea.y/2;
           x = layerFovea.x - (width/2);
           y = layerFovea.y - (height/2);
       }
    }//ImgColNextFrame


ImgColumn::ImgColumn(UMat firstFrame, InputArray  newCameraMatrix, InputArray  newDistCoeffs, int layers/*=DEFAULT_LEVELS*/)// fixed central fovea version
    {
        frameNum = 0;
        maxLevels = layers;
        newCameraMatrix.copyTo(cameraMatrix);
        newDistCoeffs.copyTo(distCoeffs);
        
        prepFrame(firstFrame);                                      // resizes firstFrame
        fovea.x = firstFrame.cols/2;                                // fixed central fovea
        fovea.y = firstFrame.rows/2;

        int width =  TILE_WIDTH;//X_IMG_COL;               // size of UMat in each column layer
        int height = TILE_HEIGHT ;
        setLayerSize(Size(height, width));
        CvPoint layerFovea = fovea;
        UMat ROI;
        int x = layerFovea.x - (width/2);
        int y = layerFovea.y  - (height/2);
        
        for(int layer=0; layer<maxLevels; layer++)
        {
            Rect RectangleToSelect(x,y,width,height);  // x,y = top left corner of rect

            ROI=Pyramid.at(layer)(RectangleToSelect);
            struct layer newLayer(ROI);                         // layer constructor
            column.push_back(newLayer);                        // NB constructr => push_back(), not at() .
            
            layerFovea.x = layerFovea.x/2;                       // assumes each Pyramid layer is 1/2 previous size
            layerFovea.y = layerFovea.y/2;
            x = layerFovea.x - (width/2);
            y = layerFovea.y - (height/2);
        }
        
        //compDisplay(this->layerSize, this->maxLevels); 
        
        //compDisplay.initROIparams(this->layerSize, this->maxLevels);  
        
    }//ImgColumn constructor

    
void ImgColumn::insertTile(int _x_pos, int &_y_pos, int _tileWidth, int _tileHeight)
        {
//cout<<"_y_pos="<<_y_pos<<flush;
            UMat image_roi = compDisplay.compositeImg( Rect(_x_pos, _y_pos,  _tileWidth,  _tileHeight ) );
            compDisplay.tile.copyTo(image_roi);
             _y_pos += (_tileHeight +1); // +1 for margin around tile
        }

        
UMat ImgColumn::showColumn( )
    {
        int tileHeight = TILE_HEIGHT;
        int tileWidth =  TILE_WIDTH;

        for (int layer=0; layer<maxLevels; layer++)                                                     // loop through the vectors, but in a particular order 
        {
            int x_pos = (tileWidth + 1) * (layer);                                                              // +1 for magin around tile
            int y_pos = 0;
            
            compDisplay.tileChans = column.at(layer).hsvVec;
            merge(column.at(layer).hsvVec, compDisplay.tile);                                       // hsv img >> tile  >> rgb >> top row of compositeImg
            cvtColor (compDisplay.tile, compDisplay.tile, COLOR_HSV2RGB, 3) ;            // nb would be more efficient to store the rgb original.
            insertTile( x_pos,  y_pos,  tileWidth, tileHeight);
            
            cvtColor (column.at(layer).hsvVec.at(0), compDisplay.tile, COLOR_GRAY2RGB, 3) ;    // hue     
            insertTile( x_pos,  y_pos,  tileWidth, tileHeight);

            for (auto&& axis : column.at(layer).hSharrVec)                                           // range for by reference over vector (of two UMats)
            {
                UMat temp1 = UMat(axis.rows, axis.cols, CV_8U, 128);                            // axis is CV_8S, needs to be changed to unsigned eg CV_8U
                add(axis,  128, temp1, noArray(), 0 );// CV_8U = 0 
                cvtColor (temp1, compDisplay.tile, COLOR_GRAY2RGB, 3) ;                                    //hue dx, hue dy
                insertTile( x_pos,  y_pos,  tileWidth, tileHeight);
            }
            cvtColor (column.at(layer).hsvVec.at(1), compDisplay.tile, COLOR_GRAY2RGB, 3) ;   // saturation
            insertTile( x_pos,  y_pos,  tileWidth, tileHeight);
            
            for (auto&& axis : column.at(layer).sSharrVec)
            {
                UMat temp1 = UMat(axis.rows, axis.cols, CV_8U, 128);
                add(axis,  128, temp1, noArray(), 0 );                                                                         // CV_8U = 0 
                cvtColor (temp1, compDisplay.tile, COLOR_GRAY2RGB, 3) ;                                       //saturation dx, saturation dy
                insertTile( x_pos,  y_pos,  tileWidth, tileHeight);
            }
            cvtColor (column.at(layer).hsvVec.at(2), compDisplay.tile, COLOR_GRAY2RGB, 3) ;      // value
            insertTile( x_pos,  y_pos,  tileWidth, tileHeight);
            
            for (auto&& axis : column.at(layer).vSharrVec)
            {
                UMat temp1 = UMat(axis.rows, axis.cols, CV_8U, 128);
                add(axis,  128, temp1, noArray(), 0 );                                                                        // CV_8U = 0 
                cvtColor (temp1, compDisplay.tile, COLOR_GRAY2RGB, 3) ;                                      //value dx, value dy 
                insertTile( x_pos,  y_pos,  tileWidth, tileHeight);
            }   
        }// loop through layers of column
//cout << endl << "compDisplay.compositeImg.size()="<<compDisplay.compositeImg.size()<<endl<<flush;
        return  compDisplay.compositeImg;
    }; // UMat showColumn( ) ###################
