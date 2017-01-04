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
        << "./video-write <input_video_name> "            << endl
        << "Input should be an avi file. Exclude the '.avi. suffix. "            << endl
        << "Output will also be an avi file.  "            << endl
        << "------------------------------------------------------------------------------" << endl
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

    
    // get codec of input video  ####################################
    string::size_type pAt = source.find_last_of('.');                                             // Find extension point
    const string NAME = source.substr(0, pAt) + "-output" + ".avi";                // Form the new name with container
    int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));                // Get Codec Type- Int form

    // Transform from int to char via Bitwise operators
    char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};

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
    cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
         << " of nr#: " << inputVideo.get(CAP_PROP_FRAME_COUNT) << endl;
    cout << "Input codec type: " << EXT << endl;

    //  working variables for the main loop  ###########################
    Mat src;
    
    // strutures for video display 
    VideoCapture captRefrnc(source);
    if (!captRefrnc.isOpened() )
    {
        cout  << "Could not open reference " << source << endl; 
        return -1;
    }
    // size of input video 
    Size refS = Size((int) captRefrnc.get(CAP_PROP_FRAME_WIDTH),
                     (int) captRefrnc.get(CAP_PROP_FRAME_HEIGHT)) ;
    // window   
    const char* WIN_RF = "Reference";
    namedWindow(WIN_RF, WINDOW_AUTOSIZE);
    moveWindow(WIN_RF, 400       , 0);         //750,  2 (bernat =0)

    cout << "Reference frame resolution: Width=" << refS.width << "  Height=" << refS.height
         << " of nr#: " << captRefrnc.get(CAP_PROP_FRAME_COUNT) << endl;

     
    // initial pre-loop     
        //        inputVideo >> src;              // read
        //        if (src.empty()) break;        // check if at end // also wait until working OR interrupted
    // set first key frame   & initialize (shape, illumination, shading, reflectance) maps  
         
    char c; 
    const int delay=500;                       // delay for keystroke at end of loop
    // the main loop #########################################
    // ##################################################
    for(;;) //get each frame when ready.
    {


        // #### main working code here ####

        // tracking (inner loop)  (nb pixelwise & anisotropic  TV-L1 regularize) 
            // get next frame
                inputVideo >> src;              // read
                if (src.empty()) break;        // check if at end
                    // gaussian pyramid
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
           
        
        
        // mapping (outer loop)  
        // if there has been parallax flow (rotation fit != 100%) & there is trackable texture (non-zero diff gaussians)
        
            // compare current frame to keyframe
            
            // consider resticted range of depth & anisotropic TV-L1 regularize
        
        
        // shape, illumination, shading, reflectance maps
            // joint optimisation & regularization (smoothness, convexity, parsimony)
            // nb penalize complexity  - i.e. use Fisher information, mutual information etc to eliminate oscillations between the maps
        
        
    
        
       // display video ####################
        imshow(WIN_RF, src);
        c = (char)waitKey(delay);  
        if (c == 27) break;       
       // write output video to file  ############################
       outputVideo << src;
    }

    cout << "Finished writing" << endl;
    return 0;    // nb relies on destructors to close the file.
}
