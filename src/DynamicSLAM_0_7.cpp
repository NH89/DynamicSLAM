#include "DynamicSLAM_0_7.h"

using namespace std;
using namespace cv;

App::App(CommandLineParser& cmd) // called by   main()
{
    appFrameNum=0;                                                                        
    cout << "\nPress ESC to exit\n" << endl;                                    // open video source
    m_camera_id  = cmd.get<int>("camera");
    m_file_name  = cmd.get<string>("video");                                 
    cout<<"App::App constructor  m_file_name=" << m_file_name << ".\n";
    
    img                   = Mat::zeros(ROWS,COLS, CV_32FC1);
    flow                  = Mat::zeros(ROWS,COLS,CV_32FC1);
    depthmap        = Mat::ones(ROWS,COLS, CV_32FC1);                 //ones
    vel_map           = Mat::zeros(ROWS,COLS, CV_32FC3);                //x,y,z chans
    extrinsic_kf      = Mat::eye(4,4,CV_32F);                                      //Identity matrix  i.e. initial extrinsic does nothing
    extrinsic_f2f     = Mat::eye(4,4,CV_32F);                                      // _kf & f2f transforms from keyframe and prev_frame
    
    fx = COLS;                                                                                    // fx,fy  Focal Length  -  default 90 degree horiz angle of view
    fy = COLS;
    skew = 0;
    x0 = COLS/2;                                                                                // x0,y0  image centre
    y0 = ROWS/2;
    float defaultCamMatrix[9]={fx, skew, x0,  0, fy, y0,  0, 0, 1}; 
    Mat temp    = Mat(3, 3, CV_32FC1, &defaultCamMatrix);
    temp.copyTo(intrinsic);                                                                 // defaultCamMatrix[] remains available for resetting intrinsic.
    inverse_intrinsic = invert_intrinsic();
    
    distCoeffs    = Mat::zeros(14,1,CV_32FC1);                                  /* distortion coefficients
    (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])  of 4, 5, 8, 12 or 14 elements.  
    If the vector is NULL/empty, the zero distortion coefficients are assumed. */
    n_frame       = Mat::zeros(ROWS,COLS, CV_8UC3);                      // resized new frame.
    n_frame.copyTo(prev_frame);
    for(int i=0; i<6;++i) {n_frame.copyTo(frames[i]);}
    
    // Lie SE(3) generators 1,2,3,4,5,6
    // jacobian_transformation
    //  for cv::Mat version of generators
    float gen_0[16]={0,0,0,1,  0,0,0,0,  0,0,0,0,  0,0,0,0 }; //trans_x
    float gen_1[16]={0,0,0,0,  0,0,0,1,  0,0,0,0,  0,0,0,0};  //trans_y
    float gen_2[16]={0,0,0,0,  0,0,0,0,  0,0,0,1,  0,0,0,0 };  //trans_z
    float gen_3[16]={0,0,0,0,  0,0,1,0,  0,-1,0,0,  0,0,0,0};  //rot_x
    float gen_4[16]={0,0,-1,0,  0,0,0,0,  1,0,0,0,  0,0,0,0};  //rot_y
    float gen_5[16]={0,1,0,0,  -1,0,0,0,  0,0,0,0,  0,0,0,0};  //rot_z
    
    //Mat gen[6];
    temp = Mat(3, 3, CV_32FC1, &gen_0);
    temp.copyTo(gen_[0]);
    temp = Mat(3, 3, CV_32FC1, &gen_1);
    temp.copyTo(gen_[1]);
    temp = Mat(3, 3, CV_32FC1, &gen_2);
    temp.copyTo(gen_[2]);
    temp = Mat(3, 3, CV_32FC1, &gen_3);
    temp.copyTo(gen_[3]);
    temp = Mat(3, 3, CV_32FC1, &gen_4);
    temp.copyTo(gen_[4]);
    temp = Mat(3, 3, CV_32FC1, &gen_5);
    temp.copyTo(gen_[5]);
    
    namedWindow(WIN_n_frame, WINDOW_AUTOSIZE );
    namedWindow(WIN_prev_frame, WINDOW_AUTOSIZE );
}

App::~App()  {} 

int App::initVideoSource(){ // called by   App::run()
    try
    {  if (!m_file_name.empty() && m_camera_id == -1)
        {  m_cap.open(m_file_name.c_str());
            if (!m_cap.isOpened())   {  throw std::runtime_error(std::string("can't open video file: " + m_file_name));}
        }
        else if (m_camera_id != -1)
        {  m_cap.open(m_camera_id);
            if (!m_cap.isOpened())
            {  std::stringstream msg;
                msg << "can't open camera: " << m_camera_id;
                throw std::runtime_error(msg.str());
            }
        }else{ throw std::runtime_error(std::string("specify video source"));
        }
    }catch (std::exception e)
    {  cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }return 0;
} // initVideoSource()

Mat App::hatSO3(float rot[3]){                                                    // generates skew symetric matrix representation of so(3) Lie Algebra 
    float so3Mat[9] = {0, -rot[0],  rot[1],     rot[0], 0, -rot[2],    -rot[1],  rot[2], 0 };
    Mat omega = cv::Mat(3,3,CV_32F, &so3Mat);
    return omega;
}

Mat App::hatSE3(float tfm[6]){                                                    // generates skew symetric 4x4 matrix representation of se(3) Lie Algebra 
    float se3Mat[16] = {0,-tfm[0],tfm[1],tfm[3],     tfm[0],0,-tfm[2],tfm[4],     -tfm[1],tfm[2],0,tfm[5],     0,0,0,1};
    Mat se3 = cv::Mat(3,3,CV_32F, &se3Mat);                                //  NB rot_xyz then trans_xyz  ordering of se(3) array.
    return se3;
}

Mat App::expMapSO3(float rot[3]){                                          //  so(3) array -> SO(3) rot matrix
    float theta = sqrt(rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2]);// from p43, sec 9.4.1 of "A tutorial on SE(3)..." (Blanco 2014)
    float u[3]={rot[0]/theta, rot[1]/theta, rot[2]/theta};
    Mat A = hatSO3(u);
    Mat I = Mat::eye(3,3,CV_32F);
    Mat SO3Mat = I + 2 * cos(theta/2) * sin(theta/2) * A  + 2 * sin(theta/2) * sin(theta/2) * A * A;
    return SO3Mat;
}

Mat App::expMapSE3(float tfm[6]){                                           // se(3) array -> SE(3) tfm matrix
    float rot[3] = {tfm[0],tfm[1],tfm[2]};                                       // from p44, sec 9.4.2 of "A tutorial on SE(3)..." (Blanco 2014)
    Mat SO3Mat = expMapSO3(rot);
    float theta = sqrt(rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2]);
    Mat I = Mat::eye(3,3,CV_32F);
    Mat V = I + (1 - cos(theta) / (theta * theta) ) * SO3Mat  +  ( (theta - sin(theta))/(theta*theta*theta) ) * SO3Mat * SO3Mat ;
    float trans[3] = {tfm[3], tfm[4], tfm[5]};
    Mat t = Mat(3,1,CV_32F, trans);
    t = V * t;
    Mat SE3Mat;
    hconcat(SO3Mat, t, SE3Mat);
    SE3Mat.resize(4,0);
    SE3Mat.at<float>(4,4) = 1;
    return SE3Mat;
}

Mat App::genWarpGrad(float fx, float fy, float px, float py, float pz){  // from  appendix A2 of "A tutorial on SE(3)..." (Blanco 2014)
    Mat warpGrad = Mat::zeros(2,6,CV_32F);
    warpGrad.at<float>(1,1)=   fx/px;
    warpGrad.at<float>(1,3)=   -fx * px / (pz * pz);
    warpGrad.at<float>(1,4)=   -fx * px * py / (pz * pz);
    warpGrad.at<float>(1,5)=   fx * ( 1 + (px * px)/(pz * pz) );
    warpGrad.at<float>(1,6)=   -fx * py / pz;

    warpGrad.at<float>(2,2)=   fy/py;
    warpGrad.at<float>(2,3)=   -fy * py / (pz * pz);
    warpGrad.at<float>(2,4)=   -fy * (1 + (py * py)/(pz * pz) );
    warpGrad.at<float>(2,5)=   fy * px * py / (pz * pz);
    warpGrad.at<float>(2,6)=   fy * px/pz;
    
    return warpGrad;
}

Mat App::pixels2points(Mat pixels, Mat depthMap, float fx, float fy, float cx, float cy){
    Mat points = Mat::zeros(pixels.rows, 3, CV_32F);
    for (int i = 0; i<pixels.rows; i++){
        float u = pixels.at<float>(i,1);
        float v = pixels.at<float>(i,2);
        float depth = depthMap.at<float>(u,v);
        points.at<float>(i,1) = (u - cx) * (depth/fx);
        points.at<float>(i,2) = (v - cy) * (depth/fy);
        points.at<float>(i,3) = depth;
    }
    return points;
}

Mat App::points2Pixels(Mat Points){                                                             // takes transformedEdgePoints  i.e. after intrinsic.
    CV_Assert(Points.cols = 3);
    int rows = Points.rows;
    Mat pixels = Mat(rows ,2, CV_32F);
    for (int i=0; i<rows; i++){
        pixels.at<float>(i,1) = Points.at<float>(i,1)/Points.at<float>(i,3);
        pixels.at<float>(i,2) = Points.at<float>(i,2)/Points.at<float>(i,3);
    }
    return pixels;
}

Mat App::genJacobian(samples sampledData,  Mat  edgePixels_du, Mat edgePixels_dv, Mat _edgePoints, float fx, float fy){
                // Sum of Jacobians  - from (Malis 2004) "Improving vision-based control using efficient second-order minimization techniques"
                // (sum of image gradients * warp gradient),  this is later used for pseudoinverse of jacobians, ie "efficient 2nd order minimisation" (ESM)
    Mat Jacobian = Mat::zeros(0, 6, CV_32F);
    for (int i = 0 ; i < _edgePoints.rows; i++){
    //    if ( pixel in curr frame   ){..} else {...}  // nb need to create valid matrices AND valid answers 
        Mat warpGrad = genWarpGrad(fx, fy, _edgePoints.at<float>(i,1), _edgePoints.at<float>(i,2), _edgePoints.at<float>(i,3));  
        Mat imgGrad = Mat(2,1,CV_32F);
        imgGrad.at<float>(1)=sampledData.at(i).dfdc[0] + edgePixels_du.at<float>(i);                                                          // cols
        imgGrad.at<float>(2)=sampledData.at(i).dfdr[0]  + edgePixels_dv.at<float>(i) ;                                                         // rows
        Mat Jacobian_row = Mat(1,6,CV_32F);
        Jacobian_row = imgGrad * warpGrad;
        Jacobian.push_back(Jacobian_row);
    }
    Jacobian.resize(keyframe_edges.rows, 6);
    return Jacobian;
}

Mat App::build_intrinsic(){  // called by   track_camera_pose()     //fx, skew,  x0, fy, y0;
    float CamMatrix[9]={fx, 0, x0,  0, fy, y0,  0, 0, 1};  // handle skew in dewarp 
    intrinsic = Mat(3, 3, CV_32FC1, &CamMatrix);
    return intrinsic;
}

Mat App::invert_intrinsic(){  // called by   track_camera_pose()
    CV_Assert(fx && fy );
    float InverseCamMatrix[9]={1/fx, 0, -x0/fx,  0, 1/fy, -y0/fy,  0, 0, 1};  
    inverse_intrinsic = Mat(3, 3, CV_32FC1, &InverseCamMatrix);
    return inverse_intrinsic;
}

void App::prepFrame()   //  called by App::run()
{
    cvtColor(m_frame,m_frame,COLOR_RGB2GRAY, 1);  
    remap(m_frame, n_frame, dewarp_map_xy, dewarp_map_interpol_tables, INTER_CUBIC); 
    // NB can adjust the scene 'intrinsic' to scale the image. This could eliminate Resize() below.
    resize(n_frame, n_frame,Size(COLS,ROWS), 0, 0, INTER_AREA);  
    //spatialGradient(n_frame, n_frame_dx, n_frame_dy);                // compute where needed for edges - or for all pixels for depth reconstruction ?
}

void App::prepKeyFrame(){  //  called by App::run()
    n_frame.copyTo(keyframe);
    spatialGradient(keyframe, keyframe_dx, keyframe_dy);
//    n_frame_dx.copyTo(keyframe_dx);
//    n_frame_dy.copyTo(keyframe_dy);                                         // nb would be more efficient to swap pointers.
    double threshold1= 1 ;                                                           // canny threshold  => vector of points for camera tracking   (for each layer and channel of pyramid) 
    double threshold2= threshold1 * 3 ;    
    keyframe_edges.zeros(keyframe.rows, keyframe.cols, CV_8U);
    Canny(keyframe_dx, keyframe_dy, keyframe_edges, threshold1, threshold2, false );
    edgePixels = Mat(0,0,CV_32F);                                                  // clear for Mat(CV_32FC2)
    edgePixels_du = Mat(0,0,CV_32FC1);
    edgePixels_dv = Mat(0,0,CV_32FC1);
    edgePixels_values = Mat(0,0,CV_32FC1);
    int numPoints=0;
    for(int i=3; i<keyframe_edges.rows-2; i++){                           // use only points with safe margin from edge of image
        for(int j=3; j<keyframe_edges.cols-2; j++){
            if (keyframe_edges.at<uchar>(i,j) ) {
                numPoints++;
                Point3f p(i,j,1);                                                               // 2D homogeneous coords // ##### need depthmap here #####
                edgePixels.push_back(p);
                edgePixels_du.push_back(keyframe_dx.at<float>(i,j) );
                edgePixels_dv.push_back(keyframe_dy.at<float>(i,j) );
                edgePixels_values.push_back(keyframe.at<float>(i,j) );
            }
        }
    }
    edgePixels.resize(numPoints);
    edgePoints = pixels2points(edgePixels, depthmap, fx, fy, x0, y0); //Mat pixels, Mat depthMap, float fx, float fy, float cx, float cy
}

samples App::sample_curr_frame (Mat sampleAt){                      // called by   track_camera_pose()
    //CV_assert(sampleAt.type()==CV_32F );
    samples Samples;
    for (int i=0; i<sampleAt.rows; i++){
        double row = sampleAt.at<float>(i,1);
        double col = sampleAt.at<float>(i,2);
        sample  data;                                                                            //double f[1], dfdr[1], dfdc[1];
        Eigen::Matrix4d patch;
        Mat temp = Mat::Mat(n_frame, Rect(col, row, 4, 4) );
        cv2eigen(temp, patch);
                    /* Ceres bicubic interpolatioin
                    //template <typename T, int kDataDimension = 1, bool kRowMajor = true, bool kInterleaved = true>
                    //(const T* data,  const int row_begin,    const int row_end,         const int col_begin,    const int col_end)      : data_(data),
                    //void Evaluate(double r, double c, double* f, double* dfdr, double* dfdc)  
                    */
        ceres::Grid2D<double, 1>   grid(patch.data(), 0, patch.rows() , 0, patch.cols() );
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > interpolator(grid);       // *** may be more efficient to use linear, until final iter(s)
        interpolator.Evaluate(row, col, data.f, data.dfdr, data.dfdc);    // nb gives value & grad at sampled point => don't need img grad of curr frame
        Samples.push_back(data);
    }
     return Samples;
}

Mat App::photometricCost(samples _sampledData){
    Mat photometricCost(0,0,CV_32F);
    float photoDiff, e2 ;
    float sigma=2;                                                                             // Geman-McLure loss fn
    for (int i=0; i<_sampledData.size() ; i++){
        sample data = _sampledData[i]   ; 
        photoDiff = data.f[0] -  edgepoints_values.at<float>(i);
        e2 = (photoDiff * photoDiff) ; 
        photometricCost.push_back( e2 /(sigma + e2) );
    }
    return photometricCost;
}

void App::track_camera_pose(){    // called by App::run()   // given a new frame find the pose
                                                                                            // early version just use small image, no pyramid
                                                //  from (Malis 2004) "Improving vision-based control using efficient second-order minimization techniques"
                                                //  pseudoinverse of jacobians isused in place of the Hessian, i.e. "efficient 2nd order minimisation"
    build_intrinsic();
    invert_intrinsic();                                                             // nb option of fitting intrinsic and dewarp, if overall fit deteriorates.
    int i = appFrameNum % 3;                                              //predict next frame to frame transpose
    for(int j=0; j<6; j++){
        prev_se3[i][j] = se3[j];
        se3[j] +=  se3[j] - prev_se3[i-1][j];
    }
    int iter=0, max_iter=6;
    do {                                                                                // 6DoF transpose optimisation looop
        iter++;
        Mat extrinsic = expMapSE3(se3);                             // se3  updated at end of this loop.
        Mat trans = intrinsic * extrinsic;                                // nb more efficient to construct trans matrix before applying it to points
        transformedEdgePoints =  trans * edgePoints ; 
        transformedEdgePixels = points2Pixels(transformedEdgePoints);
        samples sampledData = sample_curr_frame(transformedEdgePixels) ;  
                                                                                            // interpolatioin of value and gradient of curr frame at transformedEdgepoints.
        Mat  SumJacobians = genJacobian(sampledData, edgePixels_du, edgePixels_dv, edgePoints, fx, fy);
                                                // (samples sampledData,  Mat  edgePixels_du, Mat edgePixels_dv, Mat _edgePoints, float fx, float fy)
        Mat delta_x =   SumJacobians.inv(DECOMP_SVD)  * photometricCost(sampledData); 
        int elems;
        if(iter==0) elems=3; else elems =6;                          // first iter, update rotation only.
        for(int i=0; i<elems; i++){  se3[i]+=delta_x.at<float>(i);  }
    }while (iter < max_iter  );
}

void App::depth_mapping(){
    // here params are depth map, not se3. 
    // nb must also regularize -- anisotropic TV_L1
    // use Legendre-Fenchel primal-dual
    
    
    
    
    
}

int App::run() // called by main()
{
    if (0 != initVideoSource())  return -1;
    setRunning(true);                                                           // set running state until ESC pressed
    int Iter =0;
    /* initUndistortRectifyMap 	(
        InputArray  	cameraMatrix,              intrinsic
		InputArray  	distCoeffs,
		InputArray  	R,                                   Mat::eye(3,3,CV_32F) ?
		InputArray  	newCameraMatrix,        intrinsic
		Size  	size, (of new image)               Size(ROWS,COLS)
		int  	m1type,                                     CV_16SC2 or  CV_32FC1
		OutputArray  	map1,                        dewarp_map_xy,
		OutputArray  	map2                        none if map1 is 2channel.
	) 	
     */
    nextFrame(m_frame);
    if (m_frame.empty() ){
            std::cout<<"video frame empty"<<std::endl<<std::flush;
            return 1;
        }
    Size imgSize = m_frame.size();
std::cout<<"  imgSize="<<imgSize<<std::endl<<std::flush;
    initUndistortRectifyMap(intrinsic, distCoeffs, Mat::eye(3,3,CV_32F),   intrinsic,   
                            imgSize,   CV_16SC2,   dewarp_map_xy,   dewarp_map_interpol_tables);
    prepKeyFrame();                                                                   // cvtColor remap resize spatialGradient
    while (isRunning() && nextFrame(m_frame))                      // Iterate over all frames   ///////////  Principal loop
    {
        for(int frame =0; frame <6; frame++){
            if (m_frame.empty() ){
                setRunning(false);
                continue;
            }
            prepFrame();                                                                //imshow(WIN_n_frame, n_frame);
            track_camera_pose();                                                   //transform_rpy_xyz();
            handleKey((char)waitKey(30));                                     // delay 0 => until keystroke  33ms => 30fps
            Iter++;
        }
        prepKeyFrame();                                                               // new keframe every 6 frames 
        depth_mapping();
//   intrinsic&warp_params()
//   dynamicSLAM();
        
    }
    return 0;
}

void App::handleKey(char key) // called by App::run()  above
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
    App app(cmd);
    try {app.run(); }
    catch (const cv::Exception& e){
        cout << "error: cv::Exception" << e.what() << endl;
        return 1;
    }
    catch (const std::exception& e){
        cout << "error: std::exception" << e.what() << endl;
        return 1;
    }
    catch (...)    {
        cout << "unknown exception" << endl;
        return 1;
    }
    return EXIT_SUCCESS;
} // main()
