
#include <iostream>	// for standard I/O
#include <string>   // for strings

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

class ImgColumn            // creates & holds Gaussian and Lapacian pyramids 
{
public:
    ImgColumn(int maxLevels, int numChannels, UMat img); // normally 4 levels, and 3 channels.
    ~ImgColumn();
    bool initImgColumn(UMat img);
    void showColumn();
    void setFovea(int x, int y){fovea_x=x; fovea_y=y;};
    void getFovea(int& x, int& y){x*=fovea_x; y*=fovea_y;};
protected:
   
private:
    int maxLevels;
    int channels;
    int fovea_x, fovea_y;
    vector<int> inputSize;
    int inputDims;

    vector<vector<UMat>> gaussColumn;           // split 3 channel arrays 
    vector<vector<UMat>> laplaceColumn;
    vector<std::string> winName;
//    vector< ?? > windows;       //?? is this the best holder of namedWindows?
}
    
ImgColumn::ImgColumn(int initMaxLevels, int numChannels, UMat img)
{
    maxLevels=initMaxLevels;
    channels=numChannels;
    inputDims=img.dims;
    inputSize.reserve(inputDims);
    for(int dim=0;dim<inputDims;dim++) {inputSize.push_back(img.size[dim]); }
    gaussColumn.reserve(maxLevels);
    laplaceColumn.reserve(maxLevels);
    winName.reserve(maxLevels);
    
    for(int level=0;level<maxLevels;level++)
    {
        gaussColumn.at(level).reserve(channels);
        laplaceColumn.at(level).reserve(channels);
        winName.at(level).reserve(channels);
    }
    //windows.reserve(maxLevels)  ;  //?? is this the best holder of namedWindows?
}

ImgColumn::~ImgColumn()
{
    for(int level=0;level<maxLevels;level++)
    {
        destroyWindow(... );
    }
    // clear & release vectors
}

bool ImgColumn::initImgColumn(UMat img)
{
    size=img.size;
    gaussColumn.at(0);
    split(img, gaussColumn.at(0) );   /////    messy partly changed from Pyramid to collumn.
                                                        /// need to complete and cleanup, as per notes.
    
    buildPyramid(img, gaussPyr, maxLevels);
    for(int level=0;level<maxLevels;level++)
    {
        laplacePyr.push_back(gaussPyr.at(level));
        // Laplacian(tempUMatGauss, tempUMatLapl, CV_8UC3 );
        // void Scharr(InputArray src, OutputArray dst, int ddepth, int dx, int dy, double scale=1, double delta=0, int borderType=BORDER_DEFAULT )
        winName.push_back(  ) ;
//        windows ;
        
    }
    return true;
}

void showColumn()
{
    for(int level=0;level<maxLevels;level++)
    {
        gaussPyr   ;
        laplacePyr ;
        winName ;
        windows ;
        
        imshow ;
    } 
}

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

/*
void cv::resize 	( 	InputArray  	src,
		OutputArray  	dst,
		Size  	dsize,
		double  	fx = 0,
		double  	fy = 0,
		int  	interpolation = INTER_LINEAR 
	) 		
void cv::undistort 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	cameraMatrix,
		InputArray  	distCoeffs,
		InputArray  	newCameraMatrix = noArray() 
	) 	
	*/
/*
bool App::nextFrame(cv::UMat& frame) // gets, resizes, undistorts next frame
{ 
    bool rtn = m_cap.read(frame); 
    resize(frame, frame, hdSize, 0,0, INTER_LINEAR );
    // undistort(frame, frame, cameraMatrix, distCoeffs); 
    return rtn;
}*/

App::App(CommandLineParser& cmd)
{
    cout << "\nPress ESC to exit\n" << endl;
    m_camera_id  = cmd.get<int>("camera");
    m_file_name  = cmd.get<string>("video");
    cout<<"App::App constructor  m_file_name=" << m_file_name << ".\n";
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


void App::buildPyramids(cv::UMat& m_frame)
{
    
}

void App::trackCameraTranspose()
{
    
}

void App::showImages()
{
    
}


int App::run()
{
    if (0 != initVideoSource())  return -1;
    // set running state until ESC pressed
    setRunning(true);
    // Iterate over all frames
    while (isRunning() && nextFrame(m_frame))
    {
        prepFrame(m_frame); // resize, undistort, mkImgCol
        imgCol.next(m_frame);
        trackCameraTranspose();
        showImages();                               
        handleKey((char)waitKey(3));
    }

    return 0;
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

    try
    {
        app.run();
    }

    catch (const cv::Exception& e)
    {
        cout << "error: " << e.what() << endl;
        return 1;
    }

    catch (const std::exception& e)
    {
        cout << "error: " << e.what() << endl;
        return 1;
    }

    catch (...)
    {
        cout << "unknown exception" << endl;
        return 1;
    }

    return EXIT_SUCCESS;
} // main()

