
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
    int                         frameNum;
    
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
       
            for(int i=0;i<3;++i){  hsvVec.push_back(zeroUMat); }

            for(int i=0;i<2;++i){
                hSharrVec.push_back(zeroUMat); 
                sSharrVec.push_back(zeroUMat); 
                vSharrVec.push_back(zeroUMat);
            }
            layerNextFrame(firstImg);
        }
    };// struct layer ###############################
    vector<layer> column;

    class compositeDisplay{
    public:                                                                                                             //  nb -ve values are shown as zero, so convert to CV_8U before imshow
            UMat tile;                                                                                                 //(column.at(0).hsvVec.at(0).size(), CV_8SC3 );                      
            UMat compositeImg;                                                                              //((tile.rows *10), (tile.cols *maxLevels), CV_8SC3);
            vector<UMat> tileChans;
            UMat zeroTileChan;                                                                                //.zeros(tile.rows, tile.cols, CV_8S);
            int dtop, dbottom, dleft, dright=0;                                                      // ROI limits for compositeImg
            int rows ;
            int cols ;
            
            compositeDisplay(){
                tile=UMat::zeros(TILE_HEIGHT,  TILE_WIDTH, CV_8SC3);
                zeroTileChan=UMat::zeros(TILE_HEIGHT,  TILE_WIDTH, CV_8S);          // +1 magin around tiles, +2 l&r margins in cols, 128 blue background
                compositeImg=UMat(  ((TILE_HEIGHT+1)*10),  ((TILE_WIDTH+1) * (DEFAULT_LEVELS/*+1*/) +2),  CV_8UC3, 128  ); 
                this->tileChans.reserve(3);
                for(int i=0;i<3;i++) { this->tileChans.push_back(UMat::zeros(TILE_HEIGHT,  TILE_WIDTH, CV_8S)); }                  
                this->rows = TILE_WIDTH;
                this->cols = TILE_HEIGHT;
            }
        } compDisplay;
    
    void prepFrame(UMat& frame)
    {
        resize(frame, scaled, Size(X_BASE ,Y_BASE) );
        undist = scaled.clone();
        undistort(scaled, undist, cameraMatrix, distCoeffs);
        buildPyramid(undist /*frame*/, Pyramid, maxLevels);
    }//prepFrame 
    
public:
    void ImgColNextFrame(UMat nextFrame)        // fixed central fovea version
    {
        frameNum++;
cout<<"frameNum="<<frameNum;
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
           column.at(layer).layerNextFrame(ROI);          // copy ROI to imageCollumn
           layerFovea.x = layerFovea.x/2;                       // assumes each Pyramid layer is 1/2 previous size
           layerFovea.y = layerFovea.y/2;
           x = layerFovea.x - (width/2);
           y = layerFovea.y - (height/2);
       }
    }//ImgColNextFrame

public:
    ImgColumn(UMat firstFrame, InputArray  newCameraMatrix, InputArray  newDistCoeffs, int layers=DEFAULT_LEVELS)// fixed central fovea version
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

private:
        void insertTile(int _x_pos, int &_y_pos, int _tileWidth, int _tileHeight)
        {
//cout<<"_y_pos="<<_y_pos<<flush;
            UMat image_roi = compDisplay.compositeImg( Rect(_x_pos, _y_pos,  _tileWidth,  _tileHeight ) );
            compDisplay.tile.copyTo(image_roi);
             _y_pos += (_tileHeight +1); // +1 for margin around tile
        }

public:
    UMat showColumn( )
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
};// Class ImgColumn  ########################################



class App
{
public:
    App(CommandLineParser& cmd);
    ~App();
    int initVideoSource();
    int run();
    
    void setRunning(bool running)      { m_running = running; }
    bool isRunning()                              { return m_running; }

protected:
    bool nextFrame(cv::UMat& frame) { return m_cap.read(frame); }
    void handleKey(char key);
    void trackCameraTranspose();
    void showImages(int current);    

private:  
    bool                           m_running;
    Size                           hdSize = {1024, 2048};        // resize to 2^11 :2:^10 , good for pyramids.
    UMat                         cameraMatrix;
    UMat                         distCoeffs;
    string                         m_file_name;
    int                              m_camera_id;
    cv::VideoCapture       m_cap;
    cv::UMat                    m_frame;
    vector<ImgColumn> ImCols;  // could set a fixed size to limit growth. // loop around vector by modulo iterator.
#define    NUM_IMCOLS 6 // 

    // windows
    const char* WIN_compImg = "ImCol_compImg_Window";
};

App::App(CommandLineParser& cmd)
{
    // open video source
cout << "\nPress ESC to exit\n" << endl;
    m_camera_id  = cmd.get<int>("camera");
    m_file_name  = cmd.get<string>("video");
cout<<"App::App constructor  m_file_name=" << m_file_name << ".\n";
    
    cameraMatrix.create(3,3,CV_32FC1,USAGE_DEFAULT);
    float defaultCamMatrix[9] = {1462, 0, 1024,   0, 1462, 512,   0, 0, 1};//intrinsic matrix of camera
    Mat temp = Mat(3, 3, CV_32FC1, &defaultCamMatrix);
    // 1462pixels = 50*1024/35, i.e. FL of 50mm lens on '35mm' sensor with 1024 rows of pixels
    cameraMatrix=temp.getUMat(CV_32FC1);
    
    distCoeffs=UMat::zeros(14,1,CV_32FC1);
    //distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed. 
    // fit values from sensor data. 
    
    ImCols.reserve(NUM_IMCOLS);                                                  // construct vector<ImgColumn> ImCols;
    UMat zeroUMat=UMat::zeros(1024 , 2048,  CV_32FC3);           // need to use a standard 'Size'   ## ## ## 
    //zeroUMat.zeros(2048, 1024 , CV_8SC3);                                   //resized anyway, but feed exact size.
//cout << "construct App:  zeroUMat.rows=" << zeroUMat.rows << endl;  
    for(int col=0;col<NUM_IMCOLS;col++)
    {
//cout << "construct App:  zeroUMat.rows=" << zeroUMat.rows << "  col=" << col << endl << flush;        
        ImgColumn ImCol(zeroUMat, cameraMatrix, distCoeffs);
        ImCols.push_back(ImCol); 
    }
    // windows
    namedWindow ( WIN_compImg,  WINDOW_NORMAL );  //WINDOW_AUTOSIZE
} // constructor

App::~App()
{
} // destructor

int App::initVideoSource()
{
    try
    {
        if (!m_file_name.empty() && m_camera_id == -1)
        {
            m_cap.open(m_file_name.c_str());
            if (!m_cap.isOpened())
            {
                throw std::runtime_error(std::string("can't open video file: " + m_file_name));
            }
        }
        else if (m_camera_id != -1)
        {
            m_cap.open(m_camera_id);
            if (!m_cap.isOpened())
            {
                std::stringstream msg;
                msg << "can't open camera: " << m_camera_id;
                throw std::runtime_error(msg.str());
            }
        }
        else
        {
            throw std::runtime_error(std::string("specify video source"));
        }
    }

    catch (std::exception e)
    {
        cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} // initVideoSource()

void App::trackCameraTranspose()
{
    
}

void App::showImages(int current)
{
//     ///  1) matrix display of single ImgCol    

     UMat compositeImg = ImCols.at(current).showColumn( )    ; // make large enough for all UMats of column
     
//cout<<"compositeImg.rows="<<compositeImg.rows<<"  compositeImg.cols="<<compositeImg.cols<<"  compositeImg.type()="<<compositeImg.type()<<endl<<flush;     
//cout << "WIN_compImg="<<WIN_compImg<<endl<<flush;

     imshow(WIN_compImg, compositeImg);
     
cout<<"Showing compositeImg"<<endl<<flush;
     /// 2) make foveated view rgb and value edges.
     
     /// 3) make rainbow edge image of set of imgColumns 
     
}

int App::run()
{
    if (0 != initVideoSource())  return -1;
    setRunning(true);                                                                // set running state until ESC pressed
    int ImCols_Iter =0;
    //cout << "PrepFrame:  frame.rows=" << frame.rows << "   frame.cols=" << frame.cols << "   frame.depth="  << frame.depth() << endl;        

    // test VideoSource
    
    
    //
    
    while (isRunning() && nextFrame(m_frame))                      // Iterate over all frames
    {
        if (m_frame.empty() ){
            setRunning(false);
            continue;
        }
//cout << "PrepFrame:  m_frame.rows=" << m_frame.rows << "   m_frame.cols=" << m_frame.cols << "   m_frame.depth="  << m_frame.depth() << endl;        

        
        ImCols_Iter=ImCols_Iter%((int)ImCols.size());                   // modulo iterator to loop round ImgCols
        
//cout << "ImCols_Iter=" << ImCols_Iter  << endl;           
        
        ImCols.at(ImCols_Iter).ImgColNextFrame(m_frame);        // update with next frame
        
        
        
        trackCameraTranspose();
        showImages(ImCols_Iter);                                                  // show image column
        handleKey((char)waitKey(30));                                          // delay 0 => until keystroke  33ms => 30fps
        ImCols_Iter++;
    }
    return 0;
}

void App::handleKey(char key)
{
    switch (key)
    {
    case 27:                                                                                // ESC key
        setRunning(false); 
        break;
        
    default:
        break;
    }
}

int main(int argc, char** argv)
{
    const char* keys =
        "{ help h ?    |          | print help message }"
        "{ camera c    | -1       | use camera as input }"
        "{ video  v    |          | use video as input }";

    CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help"))
    {
        cmd.printMessage();
        return EXIT_SUCCESS;
    }
    string                      test_string;
    App app(cmd);
    try {app.run(); }
    catch (const cv::Exception& e){
        cout << "error: " << e.what() << endl;
        return 1;
    }
    catch (const std::exception& e){
        cout << "error: " << e.what() << endl;
        return 1;
    }
    catch (...)    {
        cout << "unknown exception" << endl;
        return 1;
    }
    return EXIT_SUCCESS;
} // main()
