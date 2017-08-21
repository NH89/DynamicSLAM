//#ifndef dynamicSLAM
//#define dynamicSLAM

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
    Mat build_rotation_matrix(float rot[3]);
    Mat build_extrinsic(float transfm[6]);
    Mat build_intrinsic();
    Mat invert_intrinsic();
    
    void buildMap(Mat img, Mat flow, Mat img_map );
    void photometric_cost(Mat old_img, Mat new_img, Mat cost_map, float& tot_cost );
    
    void predict_next_frame();
    void prepFrame();
    void prepKeyFrame();
    samples sample_curr_frame(Mat sampleAt);
    //float photometricCost(samples _sampledData);
    Mat photometricCost(samples _sampledData);
    float photometricCost(samples _sampledData, Mat transformedEdgepoints);
    void dwarp_dSE3( Vec4f edgePoint, float hom_pixel_z, Vec2f dw_dse3[6] );
    //void compute_next_step(samples _sampledData, Mat transformedEdgepoints);
    void compute_next_step( samples _sampledData,  Mat edgepoints  , Mat transformedEdgepoints  );
    
    
    void track_camera_rotation();
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
    Mat img, flow, depthmap, vel_map, projection, intrinsic, inverse_intrinsic, extrinsic, dewarp_map_xy, dewarp_map_interpol_tables; //CV_16SC2, CV_16UC1
    typedef Vec<char, 3> Vec3c;   // for CV8C3 Mat elements ie char -128~127,  nb Vec3b => uchar  0~255
    typedef Vec<char, 2> Vec2c;
    typedef Vec<char, 1> Vec1c;
    
private:  
    bool                 m_running;
    Size                 hdSize = {1024, 2048::};                               // resize to 2^11 :2:^10 , good for pyramids.
    Mat                  cameraMatrix;
    float                 fx, skew,  x0, fy, y0;
    Mat                  distCoeffs;
    string               m_file_name;
    int                    m_camera_id;
    VideoCapture  m_cap;
    Mat                  m_frame, n_frame, prev_frame, n_frame_dx, n_frame_dy;             // current frame, downsized & dewarped, previous frame. NB some could be const size Mat
    Mat                  frames[6];                                                   // store of past frames.
    
    float prev_transfrom[3][3];
    float transform[3];
    float prev_rotation[3][3];
    float rotation[3]; // expected  pitch, yaw, roll in degrees
    // nb rotational accel = rot_n-1 - ron_n-2
    // expected rotation = rot_n-1 + rot_accel = 2*rot_n-1 - rot_n-1
    Mat rot_x, rot_y, rot_z, rot_xyz, 
    Mat trans ;
    
    ////////////////////////////////////////////////////////////////////////////
    Mat keyframe, keyframe_dx, keyframe_dy, keyframe_edges;
    Mat edgepoints, edgepoints_dx, edgepoints_dy, edgepoints_values;
    Mat transformedEdgepoints ;
    
    //precomputed jacobians

    // Lie SE(3) generators 1,2,3,4,5,6
    //static const 
    Eigen::Matrix4f gen[6];
    Mat gen_[6];
    /*
    // jacobian_transformation
    static const Eigen::Matrix4f gen[0]    =  (Eigen::Matrix4f() << 0,0,0,1,  0,0,0,0,  0,0,0,0,  0,0,0,0 ).finished() ;  //trans_x
    static const Eigen::Matrix4f gen[1]    =  (Eigen::Matrix4f() << 0,0,0,0,  0,0,0,1,  0,0,0,0,  0,0,0,0 ).finished() ;  //trans_y
    static const Eigen::Matrix4f gen[2]    =  (Eigen::Matrix4f() << 0,0,0,0,  0,0,0,0,  0,0,0,1,  0,0,0,0 ).finished() ;  //trans_z 
    // jacobian_rotation
    static const Eigen::Matrix4f gen[3]       =  (Eigen::Matrix4f() << 0,0,0,0,  0,0,1,0,  0,-1,0,0,  0,0,0,0 ).finished() ;  //roll  
    static const Eigen::Matrix4f gen[4]       =  (Eigen::Matrix4f() << 0,0,-1,0,  0,0,0,0,  1,0,0,0,  0,0,0,0 ).finished() ;  //pitch 
    static const Eigen::Matrix4f gen[5]       =  (Eigen::Matrix4f() << 0,1,0,0,  -1,0,0,0,  0,0,0,0,  0,0,0,0 ).finished() ;  //yaw   
    */
    
    // jacobian_camera_intrinsic_matrix
    
    
    // jacobian_lens_distortion___params  4, 5, 8, 12 or 14 params
    
    // 
    
    // jacobian_depth_map ?
    
    // jacobian_vel_map ?
    
    // jacobian_reflectance_map ?
    
    // jacobian_illumination_map ?
    
    
    
    
    
};
//#endif
