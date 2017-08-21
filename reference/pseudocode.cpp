#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

//ImgColumn
//holds hsv, & Sharr(x,y)grad of hsv for n column layers, taken by ROI foveating a pyramid.__amd64

class ImgColumn            // creates & holds Gaussian and Lapacian pyramids 
{
public:
    ImgColumn(int maxLevels,  UMat img);  // construct and initialize at the start
    ImgColumn(int layers);                             //  construct prediction column  
    ~ImgColumn();
    bool updateColumn(UMat img);                // refresh with new frame
    void showColumn();                                   // display images ... may belong in App 
    void setFovea(CvPoint newFovea ){fovea=newFovea;};
    void getFovea(CvPoint& currentFovea){currentFovea.x =fovea.x; currentFovea.y=fovea.y;};
    
#define X_BASE 2048         // resized base image
# define Y_BASE 1024
# define X_IMG_COL 16
#define Y_IMG_COL 8
   
#define DEFAULT_LEVELS 7    //8
#define TILE_WIDTH 32           //16
#define TILE_HEIGHT 16          //8
#define NUM_IMCOLS 6
    
protected:
   
private:
    int maxLevels;
    CvPoint fovea; // nb interger coord of fovea in base of pyramid (  x  )
    UMat undist, scaled;
    vector<UMat> Pyramid;
    UMat  	cameraMatrix;
    UMat  	distCoeffs;
    Size layerSize; // set to 16x8 ?
    CvType layerType; // set to CV_8S
    
    struct layer{
        vector<UMat> hsvVec, hSharrVec, sSharrVec, vSharrVec;
        
        void layerNextFrame(UMat frame){
            split(frame, hsvVec);

            Scharr(hsvVec[0], hSharrVec[0], CV_8S,1,0); // hue x gradient
            Scharr(hsvVec[0], hSharrVec[1], CV_8S,0,1); // hue y gradient
            
            Scharr(hsvVec[1], sSharrVec[0], CV_8S,1,0); // saturation x gradient
            Scharr(hsvVec[1], sSharrVec[1], CV_8S,0,1); // saturation y gradient
            
            Scharr(hsvVec[2], vSharrVec[0], CV_8S,1,0); // value x gradient
            Scharr(hsvVec[2], vSharrVec[1], CV_8S,0,1); // value y gradient
        }
        
        layer(UMat firstImg)
        {
            UMat zeroUMat; // used to zero column layers
            zeroUMat.zeros(firstImg.size[0],firstImg.size[1] , CV_8S);
       
            for(int i=0;i<3;++i){
                hsvVec.push_back(zeroUMat); 
            }

            for(int i=0;i<2;++i){
                hSharrVec.push_back(zeroUMat); 
                sSharrVec.push_back(zeroUMat); 
                vSharrVec.push_back(zeroUMat);
            }
            layerNextFrame(firstImg);
        }
    }// struct layer ###############################
    vector<layer> column;

    void prepFrame(UMat frame)
    {
        resize(frame, scaled, Size(X_BASE ,Y_BASE) );
        undistort(scaled, undist, cameraMatrix, distCoeffs);
        buildPyramid(frame, Pyramid, maxLevels);
    }//prepFrame

    void ImgColNextFrame(UMat nextFrame)// fixed central fovea version
    {
        prepFrame(nextFrame); // resize, undistort, buildPyramid
        int width =  X_IMG_COL;// size of UMat in each column layer
        int height = Y_IMG_COL;
        CvPoint layerFovea = fovea;
        UMat ROI;
        int x = layerFovea.x - (width/2);
        int y = layerFovea.y  - (height/2);

       for(int layer=0;layer<maxLevels;layer++)
       {
           Rect RectangleToSelect(x,y,width,height);  // x,y = top left corner of rect
           ROI=Pyramid.at(layer)(RectangleToSelect);
           column.at(layer).layerNextFrame(ROI);          // copy ROI to imageCollumn
           
           layerFovea.x = layerFovea.x/2;                       // assumes each Pyramid layer is 1/2 previous size
           layerFovea.y = layerFovea.y/2;
           x = layerFovea.x - (width/2);
           y = layerFovea.y - (height/2);
       }
    }//ImgColNextFrame

    ImgColumn(UMat firstFrame, int layers=DEFAULT_LEVELS, CvPoint newFovea, Size imgColSize, InputArray  newCameraMatrix, InputArray  newDistCoeffs)// fixed central fovea version
    {
        maxLevels = layers;
        newCameraMatrix.copyTo(cameraMatrix);
        newDistCoeffs.copyTo(distCoeffs);
        fovea.x = firstFrame.cols/2;                    // fixed central fovea
        fovea.y = firstFrame.rows/2;
        
        prepFrame(firstFrame);
        int width =  X_IMG_COL;// size of UMat in each column layer
        int height = Y_IMG_COL;
        CvPoint layerFovea = fovea;
        UMat ROI;
        int x = layerFovea.x - (width/2);
        int y = layerFovea.y  - (height/2);
        
        for(int layer=0; layer<maxLevels; layer++)
        {
            Rect RectangleToSelect(x,y,width,height);  // x,y = top left corner of rect
            ROI=Pyramid.at(layer)(RectangleToSelect);
            layer newLayer(ROI);                                     // layer constructor
            column.push_back(newLayer);                        // NB constructr => push_back(), not at() .
            
            layerFovea.x = layerFovea.x/2;                       // assumes each Pyramid layer is 1/2 previous size
            layerFovea.y = layerFovea.y/2;
            x = layerFovea.x - (width/2);
            y = layerFovea.y - (height/2);
        }
        
    }//ImgColumn constructor
    
}// Class ImgColumn  ########################################


class App
{
public:
    App(CommandLineParser& cmd);
    ~App();
    int initVideoSource();
    int run();

protected:
    bool nextFrame(cv::UMat& frame){ return m_cap.read(frame); }
    void handleKey(char key);
    void buildPyramids(cv::UMat& frame);
    void trackCameraTranspose();
    void showImages();    

private:
    Size                          hdSize = {2048, 1024};        // resize to 2^11 :2:^10 , good for pyramids.
    InputArray  	            cameraMatrix;                      // undistort params 
    InputArray  	            distCoeffs;
    string                       m_file_name;
    int                            m_camera_id;
    cv::VideoCapture     m_cap;
    cv::UMat                  m_frame;
    ImgColumn              newCol, oldCol, keyCol;
};



/////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////


//Notes on img rotation, and depth/motion mapping

//Must store maps of frame-to-frame transformation, AND  predict next transformation, BEFORE the next frame arrives 
// Can make set-of-predictions, for next frame, and select from best fit, for net prediction.

struct Maps { // to be constructed from the current image, to be used to generate the predictions of the next frame
  vector<UMat> depth, normal, reflectance, illumination, shading, transform;  
    Maps(){                    // set type, channels, and initial values
        for(int i=0;i<DEFAULT_LEVELS;i++){
            depth.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_32FC1);   
            normal.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_32FC3);
            reflectance.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_8S);
            illumination.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_8S);
            shading.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_8S);
            transform.at(i).zeros(TILE_HEIGHT, TILE_WIDTH, CV_8S);
        }       
    }
};
 
#define TILES_PER_LAYER 9  // hsvVec(3),+  hSharrVec(2), + sSharrVec(2), + vSharrVec(2);

struct PredictionImageColumn {
        struct layer{
            vector<UMat> hsvVec, hSharrVec, sSharrVec, vSharrVec;   // using same members names as ImageColumn
            layer(){                                                                                        // plan to refactor code with this as a base, and former as derived srtuct.
                UMat zeroUMat; // used to zero column layers
                zeroUMat.zeros(TILE_HEIGHT, TILE_WIDTH, CV_8S);
                for(int i=0;i<3;++i){  hsvVec.push_back(zeroUMat); }
                for(int i=0;i<2;++i){
                    hSharrVec.push_back(zeroUMat); 
                    sSharrVec.push_back(zeroUMat); 
                    vSharrVec.push_back(zeroUMat);
                }
            };
        };
        vector<layer> column(DEFAULT_LEVELS);
};

vector<PredictionImageColumn> prediction[NUM_IMCOLS ];  //numPredictions


struct CostVolume {  // defined in terms of the preceeing image, and 
    
    CostVolume(int numPredictions, int layers = MAX_LAYERS ){
        
    }
}


void calcTfmCost(imageColumn Mapping,  CostVolume &costVol,  imageCollumn imcol = imCols.at(current) , imageCollumn keyframe =  imCols.at(0) ){
    
    
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


/*
void  rotImgCol( DepthMap depthMap,   imageColumnMapping &mapping  float pitch, float yaw, float roll){
    // get UMat of points     
    // matrixMul  =>  mapping
}


void tfmImgCol(DepthMap depthMap,  imageColumnMapping &mapping  float pitch, float yaw, float roll, float xtrans, float ytrans, float ztrans){
    
}*/


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Shannon entropy of a grayscale ? image. From http://stackoverflow.com/questions/24930134/entropy-for-a-gray-image-in-opencv

    if (frame.channels()==3) cvtColor(frame,frame,CV_BGR2GRAY);
    /// Establish the number of bins
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    /// Compute the histograms:
    calcHist( &frame, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    hist /= frame.total();

    Mat logP;
    cv::log(hist,logP);

    float entropy = -1*sum(hist.mul(logP)).val[0];

    cout << entropy << endl;

////////////////////////////////////////////////////////
    
     float entropy(Mat seq, Size size, int index)
{
  int cnt = 0;
  float entr = 0;
  float total_size = size.height * size.width; //total size of all symbols in an image

  for(int i=0;i<index;i++)
  {
    float sym_occur = seq.at<float>(0, i); //the number of times a sybmol has occured
    if(sym_occur>0) //log of zero goes to infinity
      {
        cnt++;
        entr += (sym_occur/total_size)*(log2(total_size/sym_occur));
      }
  }
  cout<<"cnt: "<<cnt<<endl;
  return entr;

}

// myEntropy calculates relative occurrence of different symbols within given input                sequence using histogram
Mat myEntropy(Mat seq, int histSize)
{ 

  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat hist;

  /// Compute the histograms:
  calcHist( &seq, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

  return hist;
}























