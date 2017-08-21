#include "DynamicSLAM_0_6.h"

using namespace std;
using namespace cv;

App::App(CommandLineParser& cmd) // called by   main()
{
    appFrameNum=0;                                                                        
    cout << "\nPress ESC to exit\n" << endl;                                    // open video source
    m_camera_id  = cmd.get<int>("camera");
    m_file_name  = cmd.get<string>("video");                                 
    cout<<"App::App constructor  m_file_name=" << m_file_name << ".\n";
    
    img             = Mat::zeros(ROWS,COLS, CV_32FC1);
    flow            = Mat::zeros(ROWS,COLS,CV_32FC1);
    depthmap  = Mat::ones(ROWS,COLS, CV_32FC1);                       //ones
    vel_map     = Mat::zeros(ROWS,COLS, CV_32FC3);                      //x,y,z chans
    extrinsic     = Mat::zeros(4,4,CV_32F);
    projection   = Mat::zeros(3,4,CV_32F);
    fx = COLS;                                                                                    // fx,fy  Focal Lenghth  -  default 90 degree horz angle of view
    fy = COLS;
    skew = 0;
    x0 = COLS/2;                                                                                // x0,y0  image centre
    y0 = ROWS/2;
    float defaultCamMatrix[9]={fx, skew, x0,  0, fy, y0,  0, 0, 1}; 
    Mat temp    = Mat(3, 3, CV_32FC1, &defaultCamMatrix);
    temp.copyTo(intrinsic);                                                                 // defaultCamMatrix[] remains available for resetting intrinsic.
    distCoeffs    =Mat::zeros(14,1,CV_32FC1);                                   /* distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) 
                                                                                                         // of 4, 5, 8, 12 or 14 elements.  
                                                                                                         // If the vector is NULL/empty, the zero distortion coefficients are assumed. */
    n_frame       = Mat::zeros(ROWS,COLS, CV_8UC3);                      // resized new frame.
    n_frame.copyTo(prev_frame);
    for(int i=0; i<6;++i) {n_frame.copyTo(frames[i]);}
    
    for (int i=0; i<3; ++i){
        transform[i] =0.f;
        rotation[i] =0.f;                                                                         // expected  pitch, yaw, roll in degrees
        for (int j=0; j<3; ++j){
            prev_transfrom[i][j] = 0.f;
            prev_rotation[i][j] =0.f;
    }}
    rot_x.eye(3,3,CV_32F);
    rot_y.eye(3,3,CV_32F);
    rot_z.eye(3,3,CV_32F);
    trans.zeros(3,1,CV_32F);
    
    // Lie SE(3) generators 1,2,3,4,5,6
    // static const Eigen::Matrix4f gen[6];
    // jacobian_transformation
    //const  
    gen[0]    =  (Eigen::Matrix4f() << 0,0,0,1,  0,0,0,0,  0,0,0,0,  0,0,0,0 ).finished() ;  //trans_x
    gen[1]    =  (Eigen::Matrix4f() << 0,0,0,0,  0,0,0,1,  0,0,0,0,  0,0,0,0 ).finished() ;  //trans_y
    gen[2]    =  (Eigen::Matrix4f() << 0,0,0,0,  0,0,0,0,  0,0,0,1,  0,0,0,0 ).finished() ;  //trans_z 
    // jacobian_rotation
    gen[3]    =  (Eigen::Matrix4f() << 0,0,0,0,  0,0,1,0,  0,-1,0,0,  0,0,0,0 ).finished() ;  //roll  
    gen[4]    =  (Eigen::Matrix4f() << 0,0,-1,0,  0,0,0,0,  1,0,0,0,  0,0,0,0 ).finished() ;  //pitch 
    gen[5]    =  (Eigen::Matrix4f() << 0,1,0,0,  -1,0,0,0,  0,0,0,0,  0,0,0,0 ).finished() ;  //yaw   
    
    
    //  for cv::Mat version of generators
    float gen_0[16]={0,0,0,1,  0,0,0,0,  0,0,0,0,  0,0,0,0 };
    float gen_1[16]={0,0,0,0,  0,0,0,1,  0,0,0,0,  0,0,0,0};
    float gen_2[16]={0,0,0,0,  0,0,0,0,  0,0,0,1,  0,0,0,0 };
    float gen_3[16]={0,0,0,0,  0,0,1,0,  0,-1,0,0,  0,0,0,0};
    float gen_4[16]={0,0,-1,0,  0,0,0,0,  1,0,0,0,  0,0,0,0};
    float gen_5[16]={0,1,0,0,  -1,0,0,0,  0,0,0,0,  0,0,0,0};
    
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
    
//    namedWindow(WIN_m_frame, WINDOW_AUTOSIZE );
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

void App::predict_next_frame() {   // called by  track_camera_pose()   
    // predict_transpose(), ... but could include vel_map & depth_map
    
    // create intial flow map, including independent motion, to be modified by rotate & transform, (which detect acceleration of the camera)
    // expected transpose
    int i = appFrameNum % 3;                                                         // mod 3 on frame num.
    for (int j=0; j<3; ++j){
        prev_rotation[i][j] = rotation[j];
        prev_transfrom[i][j] = transform[j];
        transform[j] += transform[j] - prev_transfrom[i-1][j];
        rotation[j] += rotation[j] - prev_rotation[i-1][j];
    }
}

Mat App::build_rotation_matrix(float rot[3]){ // called by   track_camera_pose()
     // build projection matrix
    rot_x = Mat::eye(3,3,CV_32F);
    rot_y = Mat::eye(3,3,CV_32F);
    rot_z = Mat::eye(3,3,CV_32F);
    trans = Mat::zeros(3,1,CV_32F);

    rot_x.at<float>(2,2)=cos(rot[0]);
    rot_x.at<float>(2,3)=-sin(rot[0]);
    rot_x.at<float>(3,2)=sin(rot[0]);
    rot_x.at<float>(3,3)=cos(rot[0]);
    
    rot_y.at<float>(1,1)=cos(rot[1]);
    rot_y.at<float>(1,3)=sin(rot[1]);
    rot_y.at<float>(3,1)=-sin(rot[1]);
    rot_y.at<float>(3,3)=cos(rot[1]);
   
    rot_z.at<float>(1,1)=cos(rot[2]);
    rot_z.at<float>(1,2)=-sin(rot[2]);
    rot_z.at<float>(2,1)=sin(rot[2]);
    rot_z.at<float>(2,2)=cos(rot[2]);
    
    return rot_xyz = rot_x * rot_y * rot_z ;
}

Mat App::build_extrinsic(float transfm[6]){ // called by   track_camera_pose()
     // build projection matrix
    rot_x.eye(3,3,CV_32F);
    rot_y.eye(3,3,CV_32F);
    rot_z.eye(3,3,CV_32F);
    trans.zeros(3,1,CV_32F);
    
    rot_x.at<float>(2,2)=cos(transfm[0]);
    rot_x.at<float>(2,3)=-sin(transfm[0]);
    rot_x.at<float>(3,2)=sin(transfm[0]);
    rot_x.at<float>(3,3)=cos(transfm[0]);
    
    rot_y.at<float>(1,1)=cos(transfm[1]);
    rot_y.at<float>(1,3)=sin(transfm[1]);
    rot_y.at<float>(3,1)=-sin(transfm[1]);
    rot_y.at<float>(3,3)=cos(transfm[1]);
   
    rot_z.at<float>(1,1)=cos(transfm[2]);
    rot_z.at<float>(1,2)=-sin(transfm[2]);
    rot_z.at<float>(2,1)=sin(transfm[2]);
    rot_z.at<float>(2,2)=cos(transfm[2]);
    
    trans.at<float>(1,1)=transfm[3];
    trans.at<float>(2,1)=transfm[4];
    trans.at<float>(3,1)=transfm[5];
    
    Mat extrinsic;
    hconcat( (rot_x * rot_y * rot_z),  trans, extrinsic );
    return extrinsic;
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

Mat pseudoInverseNx2(Mat jacobian){  //  will be needed by  'efficient 2nd order minimization'
    //Mat jacobian = Mat::zeros(ROWS,COLS, CV_32FC1);
    CV_DbgAssert( jacobian.cols == 2 );
    CV_DbgAssert( jacobian.type() == CV_32FC1 );
    int rows = jacobian.rows;
    float temp[2][2];
    // (J^T J)
   int a=0, b=0, c=0, d=0, e=0;
   for (int i=0; i<rows; i++){
       a+= jacobian.at<float>(i,1)* jacobian.at<float>(i,1);
       b+= jacobian.at<float>(i,1) * jacobian.at<float>(i,2);
       d+= jacobian.at<float>(i,2) * jacobian.at<float>(i,2);
   }
   c=b;
/*  inverse of 2x2 matrix
 * e = 1 / (ad-bc)
 * 
 * (a  b)^(-1)   =    e *   ( d  -a ) 
 * (c  d)                          ( -c   b )
 */
    e = 1/(a*d - b*b);
    temp[0][0]= d*e;
    temp[0][1]=-a*e;
    temp[1][0]=-b*e;
    temp[1][1]=b*e;
    
    Mat pinv = Mat(2,rows, CV_32FC1);  //same size as jacobian
    /* temp * J^T
     * For temp_ik * J^T_kj :    pinv_ij = (k=1->2) SUM Aik * Bkj     =      temp_i1 * J_j1  + tempi2 *J_j2
    */
    for (int i=0; i<rows; i++){
        pinv.at<float>(1, i) = temp[0][0] * jacobian.at<float>(i , 1 )  + temp[0][1] * jacobian.at<float>(i ,  2) ;
        pinv.at<float>(2, i) = temp[1][0] * jacobian.at<float>(i , 1 )  + temp[1][1] * jacobian.at<float>(i ,  2) ;
    }
    return pinv;
}


    

void App::prepFrame()   //  not currently called 
{
    cvtColor(m_frame,m_frame,COLOR_RGB2GRAY, 1);  
    remap(m_frame, n_frame, dewarp_map_xy, dewarp_map_interpol_tables, INTER_CUBIC); 
    // NB can adjust the scene 'intrinsic' to scale the image. This could eliminate Resize() below.
    resize(n_frame, n_frame,Size(COLS,ROWS), 0, 0, INTER_AREA);  
    //spatialGradient(n_frame, n_frame_dx, n_frame_dy);                  // compute where needed for edges - or for all pixels for depth reconstruction ?
}

void App::prepKeyFrame(){  //  not currently called 
    n_frame.copyTo(keyframe);
    spatialGradient(keyframe, keyframe_dx, keyframe_dy);
//    n_frame_dx.copyTo(keyframe_dx);
//    n_frame_dy.copyTo(keyframe_dy);                                             // nb would be more efficient to swap pointers.
    double threshold1= 1 ;                                                             // canny threshold  => vector of points for camera tracking   (for each layer and channel of pyramid) 
    double threshold2= threshold1 * 3 ;    
    keyframe_edges.zeros(keyframe.rows, keyframe.cols, CV_8U);
    Canny(keyframe_dx, keyframe_dy, keyframe_edges, threshold1, threshold2, false );
    edgepoints = Mat(0,0,CV_32F);                                                  // clear for Mat(CV_32FC2)
    edgepoints_dx = Mat(0,0,CV_32FC1);
    edgepoints_dy = Mat(0,0,CV_32FC1);
    edgepoints_values = Mat(0,0,CV_32FC1);
    int numPoints=0;
    for(int i=3; i<keyframe_edges.rows-2; i++){                            // use only points with safe margin from edge of image
        for(int j=3; j<keyframe_edges.cols-2; j++){
            if (keyframe_edges.at<uchar>(i,j) ) {
                numPoints++;
                Point3f p(i,j,1);                                                                 // 2D homogeneous coords
                edgepoints.push_back(p);
                edgepoints_dx.push_back(keyframe_dx.at<float>(i,j) );
                edgepoints_dy.push_back(keyframe_dy.at<float>(i,j) );
                edgepoints_values.push_back(keyframe.at<float>(i,j) );
            }
        }
    }
    edgepoints.resize(numPoints);
}

samples App::sample_curr_frame (Mat sampleAt){  // called by   track_camera_pose()
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
        ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > interpolator(grid);
        interpolator.Evaluate(row, col, data.f, data.dfdr, data.dfdc);    // nb gives value & grad at sampled point => don't need img grad of curr frame
        Samples.push_back(data);
    }
     return Samples;
}


void App::dwarp_dSE3( Vec4f edgePoint, float hom_pixel_z, Vec2f dw_dse3[6] ){   // called by    compute_next_step()
                                                                                                //  NB minimized calculation is less clear than matrix mul.
                                                                                                //  NB intrisic and extrinsic values could be fetched in a class and reused ?
                                                                                                // NB should use Lie Groups to work with ESM
    float A = intrinsic.at<float>(1,1) ;                                      // fx
    float C = intrinsic.at<float>(1,3) ;                                      // u0
    float E = intrinsic.at<float>(2,2) ;                                      // fy
    float F = intrinsic.at<float>(2,3) ;                                      // v0
    float I = intrinsic.at<float>(3,3) ;                                       //  1
    
    float b,c, e, g, i, j, x,y,z;            //  extrinsic transpose              intrinsic projection
    b = extrinsic.at<float>(1,2);      // (a, b, c, d)                             (A, B, C)
    c = extrinsic.at<float>(1,3);      // (e, f,  g, h)                             (D, E, F)
    e = extrinsic.at<float>(2,1);      // (i,  j,  k, l)                              (G, H, I )
    g = extrinsic.at<float>(2,3);      // (m, n, o, p)
    i = extrinsic.at<float>(3,1);
    j = extrinsic.at<float>(3,2);
    
    x = edgePoint[0];
    y = edgePoint[1];
    z = edgePoint[2];

    dw_dse3[0] = Vec2f( A / hom_pixel_z   ,  0                        )  ;  // trans_x  = (+fx/w, 0)  
    dw_dse3[1] = Vec2f( 0                          ,  E / hom_pixel_z )  ;  // trans_y  = (+fy/w, 0)
    dw_dse3[2] = Vec2f( C  / hom_pixel_z  ,  F / hom_pixel_z )   ;  // trans_z  = (/w, 0)

    dw_dse3[3] = Vec2f( -C*j*x /(hom_pixel_z - I*j*x)                      ,  ( -F*j*x +E*g*z)   /(hom_pixel_z - I*j*x)  ); 
    dw_dse3[4] = Vec2f( ( -A*c*z + C*i*x)/ (hom_pixel_z  - I*i*x)    ,    F*i*x/(hom_pixel_z  - I*i*x)                   ); 
    dw_dse3[5] = Vec2f( A*b*y/hom_pixel_z                                   ,    -E*e*x/hom_pixel_z                              );       
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

float App::photometricCost(samples _sampledData, Mat transformedEdgepoints){ // called by compute_next_step(...) below
    double f_sum=0, dfdr_sum=0, dfdc_sum=0;
    float photometricCost, photoDiff, e2 ;
    float sigma=2;                                                                             // Geman-McLure loss fn
    for (int i=0; i<_sampledData.size() ; i++){
        sample data = _sampledData[i]   ; 
        f_sum += data.f[0];
        dfdr_sum += data.dfdr[0];
        dfdc_sum += data.dfdc[0];
       
        photoDiff = data.f[0] -  edgepoints_values.at<float>(i);
        e2 = (photoDiff * photoDiff) ; 
        photometricCost += e2 /(sigma + e2);                                     // Geman-McLure loss fn
        
        float kf_dx = keyframe_dx.at<float>(i);
        float kf_dy = keyframe_dy.at<float>(i);
        float cf_dx = data.dfdc[0];
        float cf_dy = data.dfdr[0];
        //Eigen::Vector2f kf_grad(kf_dx, kf_dy);                                                    // keyframe gradient
        //Eigen::Vector2f cf_grad(cf_dx, cf_dy);                                                    // current frane gradient
        float dx = kf_dx + cf_dx;
        float dy = kf_dy + cf_dy;
        Eigen::Vector3f img_grad(dx, dy,  1.0) ;
        
        //vec3f px_coord(  sampleAt.at<float>(i,1)  , sampleAt.at<float>(i,1)  , 1   ) ; 
        Vec4f pt_coord = transformedEdgepoints.row(i)      ;
        
        Vec6f partials_SE3;
        for(int j=0; j<6; j++){
        partials_SE3[j] =    img_grad *  proj_mat * (pt_coord  *  gen[j]);    // gen[0] is generator for translation in x, an Eigen::Matrix4f
                                                                                              // img_grad is a Vec3f,  pt.coord is a Vec4f   .... ??
                                                                                                // transform point by generator, 
                                                                            // should this be here ? where should calculation of partials be ?
        }
    }
    
    vector<Vec6f> step;                                                                   // roll, pitch, yaw, x, y, z
    //step.[0] = photometric cost * dfdr_sum * drdroll  +  photometric cost * dfdc_sum * dcdroll ;
    
    //drdroll =   ; // partial diff of image_row by roll
    //dcdroll =   ; // partial diff of image_col by roll
    
    return photometricCost;
}

void App::compute_next_step( samples _sampledData,  Mat _edgepoints  , Mat transformedEdgepoints  ){ // called by   track_camera_pose()                                                                                        
    int rows = _edgepoints.rows;
    CV_Assert( (_edgepoints.cols==4) && (transformedEdgepoints.cols==3)  && ( transformedEdgepoints.rows==rows ) );
    Mat SumJacobians = Mat::zeros(rows, 2, CV_32F);
    float nextStep[6];  
    for(int i=0;i<6; i++){nextStep[i]=0;}
    
    for(int i=0; i<rows; i++){                                                                     // for each point 
        Vec4f edgePoint( _edgepoints.at<float>(i,1), _edgepoints.at<float>(i,2), _edgepoints.at<float>(i,3), _edgepoints.at<float>(i,4)  )  ;  
                                                                                                                  // original xyz coords of pixel
        float hom_pixel_z  =  transformedEdgepoints.at<float>(i,3)   ;      // homogenerous z coord of pixel
        Vec2f dw_dse3[6] ;
        dwarp_dSE3(edgePoint, hom_pixel_z , dw_dse3 ) ;                           // Vec4f edgePoint,  float hom_pixel_z,  Vec2f dw_dse3[6] 
        sample Sample;
        for(int j=0; j<6; j++  ){                                                                   // for each parameter, for img row and img col
            Sample = _sampledData.at(j);
            //  img_motion_grad wrt param  * (img_grad@curr_frame + img_grad@keyframe) * photometriccost 
                                                                                                                // for image u&v coords : img_grad * dwarp_dSE3
            float col_grad = dw_dse3[j][0] * ( Sample.dfdc[0]  +  keyframe_dx.at<float>(i) ) ;
            float row_grad =  dw_dse3[j][1] * ( Sample.dfdr[0]  +  keyframe_dy.at<float>(i) ) ;
            SumJacobians.at<float>(i,j) =  (col_grad + row_grad) ;               // the row of the jacobian for this pixel, where j is the param ID
        }
    }
    // NB Jacobian is (param x pixels) matrix
    
    // make partial wrt params
    Mat dwarp_dSE3_[6];
    for (int i=0; i<6; i++){ 
    dwarp_dSE3_[i] = intrinsic * gen_[i] * edgepoints ;                      //  for each param find movememt in image plane for edge points
    }
    
    // then effcient 2nd order minimization, ie find  pseudo inverse of the summed jacobians
    Mat delta_x =   SumJacobians.inv(DECOMP_SVD)  * photometricCost( _sampledData); 
                                                                            // nb delta_x is the warp params of next step.  Jacobian should be 2 column Mat
}

void App::track_camera_pose(){    // called by App::run()
    //  early version just use small image, no pyramid
    // Lie group of rotations & translations 
    // Lie algebra of rotations & translations
    // predict camera transform
    predict_next_frame();
    build_rotation_matrix(rotation);
    build_extrinsic(transform);
    build_intrinsic();
    invert_intrinsic();
                                                                                            // rotation only one iteration
    trans = intrinsic * rot_xyz *inverse_intrinsic;                   // all Mat(3, 3, CV_32FC1)
    transformedEdgepoints =  trans * edgepoints ; 
    samples sampledData = sample_curr_frame(transformedEdgepoints) ;  
                                                                                            // interpolatioin of value and gradient of curr frame at transformedEdgepoints.
    compute_next_step(sampledData, edgepoints , transformedEdgepoints); 

    int iter=0, max_iter=6;
    do {                                                                                // 6DoF transpose optimisation looop
        edgepoints  ;
        
        
    }while (iter < max_iter     );
}

int App::run() // called by main()
{
    if (0 != initVideoSource())  return -1;
    setRunning(true);                                                                // set running state until ESC pressed
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
            prepFrame();                                                                   //imshow(WIN_n_frame, n_frame);
            predict_next_frame();
            track_camera_pose();                                                      //transform_rpy_xyz();
            handleKey((char)waitKey(30));                                       // delay 0 => until keystroke  33ms => 30fps
            Iter++;
        }
        prepKeyFrame();                                                         // new keframe every 6 frames 
//   depth_mapping();
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
