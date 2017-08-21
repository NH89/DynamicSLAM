#include <iostream>	// for standard I/O
#include <string>   // for strings

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

static void help()
{
    cout
        << "------------------------------------------------------------------------------" << endl
        << "DynamicSLAM_01."                                     << endl
        << "First OpenCV implementation "                   << endl
        << "Usage:"                                                        << endl
        << "./DynamicSLAM_06 <input_video_name> "            << endl
        << "Input should be an avi file. E.g. DynamicSLAM_06 --video=Megamind.avi "            << endl
        << "Output will also be an avi file.  "            << endl
        << "----------------------------------------------------------------------------" << endl
        << endl;
}

int main(int argc, char *argv[])
{
    help();
    if (argc != 2)
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }
    const string source      = argv[1];                  // the source file name, from video-write.cpp
    VideoCapture inputVideo(source);                // Open input
    if (!inputVideo.isOpened())
    {
        cout  << "Could not open the input video: " << source << endl;
        return -1;
    }
    // General purpose window for debugging
    const char* TEST_WIN = "Test_Window";
    namedWindow(TEST_WIN, WINDOW_AUTOSIZE);

    // get codec of input video  ####################################
    string::size_type pAt = source.find_last_of('.');                                             // Find extension point
    const string NAME = source.substr(0, pAt) + "-output" + ".avi";                // Form the new name with container
    int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));                // Get Codec Type- Int form

    // get size & fps of input video  ###############################
    Size S = Size((int) inputVideo.get(CAP_PROP_FRAME_WIDTH),                     // Acquire input size
                (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT));

    // open output video   #####################################
    VideoWriter outputVideo;                                                                            // Open the output
    outputVideo.open(NAME, ex, inputVideo.get(CAP_PROP_FPS), S, true);       // use input codec type
    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << source << endl;
        return -1;
    }
    /*
    Transform from int to char via Bitwise operators
     char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
       cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
       << " of nr#: " << inputVideo.get(CAP_PROP_FRAME_COUNT) << endl;
       cout << "Input codec type: " << EXT << endl;*/

    //  working variables for the main loop  ###########################
    UMat src, displayUmat;
    VideoCapture captRefrnc(source);                                                         // strutures for video display 
    if (!captRefrnc.isOpened() )
    {
        cout  << "Could not open reference " << source << endl; 
        return -1;
    }
    
    // window   ############################################
    const char* WIN_RF = "Reference";
    namedWindow(WIN_RF, WINDOW_NORMAL/*WINDOW_AUTOSIZE*/);
    moveWindow(WIN_RF, 400       , 0);         //750,  2 (bernat =0)
/*
    // size of input video 
    //  Size refS = Size((int) captRefrnc.get(CAP_PROP_FRAME_WIDTH),  (int) captRefrnc.get(CAP_PROP_FRAME_HEIGHT)) ;
    //    cout << "Reference frame resolution: Width=" << refS.width << "  Height=" << refS.height << " of nr#: " << captRefrnc.get(CAP_PROP_FRAME_COUNT) << endl;
 */
    // UMats for main loop  #####################################
    const int maxPyrLevels=4;
    //#define maxPyrLevels 4
    vector<UMat> newFrameGauPyr, previousFrameGauPyr, keyFrameGauPyr;			          //Gaussian pyramids
    vector<UMat> newFrameLaplPyr, previousFrameLaplPyr, keyFrameLaplPyr;		          //Laplacian pyramids
    vector<UMat> newFrameVGradPyr, previousFrameVGradPyr, keyFrameVGradPyr; 		  //(x,y) Vector gradient pyramids ? are these needed ? 
    
    newFrameGauPyr.reserve(maxPyrLevels);
    previousFrameGauPyr.reserve(maxPyrLevels);
    keyFrameGauPyr.reserve(maxPyrLevels);			      //Gaussian pyramids
    newFrameLaplPyr.reserve(maxPyrLevels);
    previousFrameLaplPyr.reserve(maxPyrLevels); 
    keyFrameLaplPyr.reserve(maxPyrLevels);		          //Laplacian pyramids
    newFrameVGradPyr.reserve(maxPyrLevels);
    previousFrameVGradPyr.reserve(maxPyrLevels); 
    keyFrameVGradPyr.reserve(maxPyrLevels); 		      //(x,y) Vector gradient 

    std::vector<cv::String> PYR_WIN;                        // array of windows for showing pyramid
    PYR_WIN.reserve(maxPyrLevels);
    
    // initial pre-loop  #######################################
    char c; 
    const int delay=100;                       						  // delay in mili-seconds for keystroke at end of loop    
    inputVideo >> src;                                                    // read
    if (src.empty()) 
    {
        cout << "initial frame empty" << endl;
        return -1;                                                              // check if at end
    }

    // set first key frame   & initialize (shape, illumination, shading, reflectance) maps  ##############
    buildPyramid(src,keyFrameGauPyr, maxPyrLevels);
     for(int i=0;i<maxPyrLevels;i++)
    {
        keyFrameLaplPyr.push_back(keyFrameGauPyr.at(i));                                                             // expand the keyFrameLaplPyr & newFrameLaplPyr vectors for the first time
 //       newFrameLaplPyr.push_back(keyFrameGauPyr.at(i));                                                          // make Laplacian of keyframe
        Mat tempUMatGauss(keyFrameGauPyr.at(i).clone() );
                                                                                                                                                        // cvt color to HSV, then do Laplacian on each channel nb expect different significance wtr material, shape and illumination
      //  cvtColor(tempUMatGauss, tempUMatGauss, COLOR_RGB2GRAY);
        Mat tempUMatLapl(tempUMatGauss.clone() );
        Laplacian(tempUMatGauss, tempUMatLapl, CV_8UC3 );  /*
                                                                                                                                                        CV_EXPORTS_W void Laplacian( InputArray src, OutputArray dst, int ddepth,
                                                                                                                                                                                                            int ksize = 1, double scale = 1, double delta = 0,
                                                                                                                                                                                                            int borderType = BORDER_DEFAULT );
    */
    }
    /*
    displayUmat = keyFrameGauPyr.at(0);                                                                                                                    // Display keyFrameGauPyr.at(0); in TEST_WIN
    imshow(TEST_WIN, displayUmat); 
    */
    std::copy(keyFrameLaplPyr.begin(), keyFrameLaplPyr.end(), std::back_inserter(previousFrameLaplPyr));                            // copy current=>previous keyframe
    std::copy(keyFrameGauPyr.begin(), keyFrameGauPyr.end(), std::back_inserter(previousFrameGauPyr));

    for(int i=0;i<maxPyrLevels;i++)
    {
        std::stringstream ss;
        ss << "PYR_WIN_" << (char)i ;
        PYR_WIN.push_back(ss.str());
        namedWindow(PYR_WIN.at(i), WINDOW_AUTOSIZE );       // WINDOW_NORMAL
        moveWindow(PYR_WIN.at(i), 400*i       , 200);                   //750,  2 (bernat =0)
    }

    std::cout << std::endl << "starting main loop:" <<std::endl << std::flush;
    // the main loop #########################################
    // ##################################################
    bool endOfVideo = false;
    do
    {
        // display video ####################
        imshow(WIN_RF, src);                                                                                                   // display keyframe
        c = (char)waitKey(delay);  
        if (c == 27) break;       
        buildPyramid(src,keyFrameGauPyr, maxPyrLevels);                                                     // make gaussian pyramid
        for(int level=0; level<maxPyrLevels; level++)  
        {
            Laplacian(keyFrameGauPyr.at(level), keyFrameLaplPyr.at(level), CV_8UC3 );         // make Laplacian pyramid
            imshow(PYR_WIN.at(level) ,keyFrameLaplPyr.at(level));                                           // display pyramids
            /*
            std::cout <<"sample pixel type = "<<  keyFrameGauPyr.at(level).type();  // "16"=type CV_U8 3channels, ie uchar
            Mat testMat=keyFrameGauPyr.at(level).getMat(ACCESS_READ);
            std::cout <<", sample pixel value = "<<  testMat.at<Vec3b>(50,50); 
            testMat.release();
            std::cout <<", keyFrameGauPyr.at("<<level<<").size()=" << keyFrameGauPyr.at(level).size() << std::endl << std::flush;
            */
        }
        
    for(int j=0;j<6;j++)                                                                                                          // tracking (inner loop)  (nb pixelwise & anisotropic  TV-L1 regularize) 
    {
                // get next frame
                inputVideo >> src;                              // read
                if (src.empty())                                    // check if at end
                {
                    std::cout << "end of video file" << std::endl << std::flush;
                    endOfVideo=true;
                    break;          
                }
                std::cout << j << " ";
               buildPyramid(src,newFrameGauPyr, maxPyrLevels);
               for(int level=0; level<maxPyrLevels; level++){Laplacian(newFrameGauPyr.at(level), newFrameLaplPyr.at(level), CV_8UC3 ); } 
  /*             
                    // gaussian pyramid
                    //* buildPyramid(src, newFramePyr, maxPyrLevels);
                    
                        //  optional - foveal tracking & saccading
                            // take top down tracking instruction (if any)
                            // saliency test - unexpected movement
                            // flow at the fovea => predict next step + catchup

                    // foveate : take central collumn of pyramid
                    // diff gauss edges + Canny extension
            
            // Track (each level against higher level from previous frame + depth map)
                //  start from prediction from previous two time steps (6D vel & accel, nb pixelwsie inertia vs camera shake rotation)
                //  rotate top level
                //  rotate & translate top level
                // descend levels
           */     
    }
   /*  
        // mapping (outer loop)  
        // if there has been parallax flow (rotation fit != 100%) & there is trackable texture (non-zero diff gaussians)
        
            // compare current frame to keyframe
            
            // consider resticted range of depth & anisotropic TV-L1 regularize
        
        // shape, illumination, shading, reflectance maps
            // joint optimisation & regularization (smoothness, convexity, parsimony)
            // nb penalize complexity  - i.e. use Fisher information, mutual information etc to eliminate oscillations between the maps
   */     
    // write output video to file  ############################
    outputVideo << src.getMat(cv::ACCESS_READ);
    std::cout << std::endl << std::flush;
    }while(!endOfVideo);

    destroyAllWindows();
    std::cout << "Finished writing" << std::endl << std::flush;
    return 0;    // nb relies on destructors to close the file.
}

// test difference
