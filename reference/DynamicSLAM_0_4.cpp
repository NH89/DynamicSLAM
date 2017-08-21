
#include <iostream>	// for standard I/O
#include <string>   // for strings

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

#include "image_column.h"
#include "tracking_and_mapping.h"

using namespace std;
using namespace cv;



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
