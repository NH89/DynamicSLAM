#ifndef dynamicSLAM
#define dynamicSLAM

#include <gsl/gsl>
#include <iostream>	                      // for standard I/O
#include <string>                             // for strings

#include </usr/include/eigen3/Eigen/Dense> // must be before #include <opencv2/core/eigen.hpp>

#include <opencv2/core.hpp>         // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>   // Gaussian Blur
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>    // OpenCV window I/O

#include <ceres/cubic_interpolation.h>   // used for bicubic interpolation
#include "ceres/jet.h"
#include "ceres/internal/scoped_ptr.h"

#include "image_column.h"
#include "tracking_and_mapping.h"

#define ROWS       240                     // (240p = 426x240, 16:9 for youtube, 24/25/30/48/50/60fps)
#define COLS         426
#define PIXELS     ROWS * COLS

using namespace std;
using namespace cv; 

struct sample{ double  f[1], dfdr[1], dfdc[1];}    //  needed to work with Ceres bicubic interpolator
typedef vector<sample> samples;

class App
{
public:
    App(CommandLineParser& cmd);
    ~App();
    int initVideoSource();
    int run();
    void setRunning(bool running)      { m_running = running; }
    bool isRunning()                              { return m_running; }
    
    Mat hatSO3(float rot[3]);
    Mat hatSE3(float tfm[6]);
    Mat expMapSO3(float rot[3]);
    Mat expMapSE3(float tfm[6]);
    Mat genWarpGrad(float fx, float fy, float px, float py, float pz);
    Mat pixels2points(Mat pixels, Mat depthMap, float fx, float fy, float cx, float cy);
    Mat points2Pixels(Mat transformedEdgePoints);
    Mat genJacobian(samples sampledData,  Mat  edgePixels_du, Mat edgePixels_dv, Mat _edgePoints, float fx, float fy);
    samples sample_curr_frame(Mat sampleAt);
    Mat photometricCost(samples _sampledData); 
    
    Mat build_intrinsic();
    Mat invert_intrinsic();    
    void prepFrame();
    void prepKeyFrame();
    void track_camera_pose();
    void depth_mapping();
    //void dynamicSLAM();
    
protected:
    const char* WIN_m_frame       = "m_frame_Window";
    const char* WIN_n_frame        = "n_frame_Window";
    const char* WIN_prev_frame  = "prev_frame_Window";

    void handleKey(char key);
    int    appFrameNum;
    bool nextFrame(cv::Mat& frame) { 
        appFrameNum++;
        cout<<endl<<" appFrameNum="<<appFrameNum<<endl<<flush;  
        return m_cap.read(frame); 
    }
    Mat img, flow, depthmap, vel_map, projection, intrinsic, inverse_intrinsic, extrinsic_kf, extrinsic_f2f, dewarp_map_xy, dewarp_map_interpol_tables;                                         //CV_16SC2, CV_16UC1
    
    typedef Vec<char, 3> Vec3c;                                            // for CV8C3 Mat elements ie char -128~127,  nb Vec3b => uchar  0~255
    typedef Vec<char, 2> Vec2c;
    typedef Vec<char, 1> Vec1c;
    
private:  
    bool                 m_running;
    Size                 hdSize = {1024, 2048::};                           // resize to 2^11 :2:^10 , good for pyramids.
    Mat                  cameraMatrix;
    float                 fx, skew,  x0, fy, y0;
    Mat                  distCoeffs;
    string               m_file_name;
    int                    m_camera_id;
    VideoCapture  m_cap;
    Mat                  m_frame, n_frame, prev_frame, n_frame_dx, n_frame_dy;            
                            // current frame, downsized & dewarped, previous frame. NB some could be const size Mat
    Mat                  frames[6];                                                   // store of past frames.
    
    float se3[6] = {0,0,0,  0,0,0};                                             // Lie algebra as vector {rot_x, rot_y, rot_z,    trans_x, trans_y, trans_z}
    float prev_se3[3][6] = {{0,0,0,  0,0,0},  {0,0,0,  0,0,0},   {0,0,0,  0,0,0}};
    
    ////////////////////////////////////////////////////////////////////////////
    Mat keyframe, keyframe_dx, keyframe_dy, keyframe_edges;
    Mat edgePixels, edgePixels_du, edgePixels_dv, edgePixels_values;
    Mat edgePoints;
    Mat transformedEdgePoints, transformedEdgePixels;
    
    //precomputed jacobians

    // Lie SE(3) generators 1,2,3,4,5,6
    Mat gen_[6];

    // jacobian_camera_intrinsic_matrix

    // jacobian_lens_distortion___params  4, 5, 8, 12 or 14 params
    
    // jacobian_depth_map ?
    
    // jacobian_vel_map ?
    
    // jacobian_reflectance_map ?
    
    // jacobian_illumination_map ?
    
};
#endif
