#include <gsl/gsl>
#include <opencv2/core.hpp>         // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>   // Gaussian Blur
using namespace std;
using namespace cv;
//nb1 first do it in plain opencv3.2, then later make opencl versions.
//nb2 first do it with plain small images, later make custom logpolar remap for foveal sampling
#define WORKING_IMG_SIZE     426,240   // ie YouTube minimum size
#define WORKING_IMG_ROWS  240
#define WORKING_IMG_COLS    426
#define WORKING_IMG_PX_COUNT 426*240

class rendering
{
    array<UMat,4>::size_type  frameNum;
    array<UMat,4>frames;                                                   // (WORKING_IMG_SIZE, CV_32FC3)
    array<UMat,7>predictions;                                           // prediction + grad in rotation & translation // later other params
    float theta_x, theta_y, theta_z, trans_x, trans_y, trans_z, f_x, f_y, x_0, y_0, skew;
    struct camera { UMat distorsion_coeffs, camera_matrix, transpose;} cam;
    struct scene_model { UMat pixel_cloud, illum_map, vel_map, depth_map, refl_map;}scene;

    rendering( ){                                                                  // default constructor
        Mat temp;
        frameNum = 0;
        Size img_sz(WORKING_IMG_SIZE);
        for(auto&& frame : frames  )  {frame.zeros(WORKING_IMG_SIZE, CV_32FC3);};  
        for(auto&& prediction : predictions) {prediction.zeros(WORKING_IMG_SIZE, CV_32FC3);};
                                                                                            //distorsion_coeffs : (k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])
        float default_distorsion_coeffs[ 14 ] = {0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0 };
        temp = Mat(1,14, CV_32FC1, &default_distorsion_coeffs);
        cam.distorsion_coeffs=temp.getUMat(CV_32FC1);
                                                                                            //default intrinsic matrix of camera 
        float centre_x = WORKING_IMG_COLS/2;
        float centre_y = WORKING_IMG_ROWS/2;
        float defaultCamMatrix[9] = {centre_x, 0, centre_x,   0, centre_y, centre_y,   0, 0, 1};
        temp = Mat(3, 3, CV_32FC1, &defaultCamMatrix);
        cam.camera_matrix=temp.getUMat(CV_32FC1);
        
        float default_cam_transpose[ 16 ] = {0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0};
        temp = Mat(4, 4, CV_32FC1, &defaultCamMatrix);
        cam.transpose=temp.getUMat(CV_32FC1);
        
        temp = Mat::ones(WORKING_IMG_SIZE, CV_32FC1);
        temp *= 0.5;
        scene.depth_map = temp.getUMat(CV_32FC1);               // 1/z depth
        
        temp = Mat::zeros(WORKING_IMG_SIZE, CV_32FC3); 
        scene.pixel_cloud = temp.getUMat(CV_32FC3);                // rgb

        scene.vel_map = temp.getUMat(CV_32FC3);                    // xyz
/*  Illumination & reflectance - later versions       
 * 
 *      scene.illum_map = temp.getUMat(CV_32FC3);                 // pixelwise sine lobes 
        // Cooke-Torrance reflectance
        // r = ambient + sum(l){(n.l)*(k+(1-k)*r_s)
        // where r_s = cross(F,D,G) / pi*dot(n,l)*dot(n.v)
        // n = normal
        // l = lambertian reflectance 
        // k = fraction of light that is reflected diffusely
        // F = fresnel 
        // D = directional distribution of microfacets 
        // G = geometric attenuation 
        // v = viewing direction
        scene.refl_map = temp.getUMat(CV_32FC1);                    // painting by numbers, 

        vector<array<int,3> >materials; // (vector of materials in the scene, currently just rgb values)
        */
    }
    
    UMat make_projection_matrix(float theta_x, float theta_y, float theta_z, float trans_x, float trans_y, float trans_z, float f_x, float f_y, float x_0, float y_0, float skew  );
    
    Mat rendering::splat_points(Mat pointcloud , Mat srcImage , Mat targetImage = Mat::zeros(WORKING_IMG_SIZE, CV_32FC3) );
    // nb needs pixel access, hence Mat not UMat.
    
    Mat rendering::predict_image(float theta_x=0.0,float theta_y=0.0, float theta_z=0.0, float trans_x=0.0, float trans_y=0.0, float trans_z=0.0, float f_x=0.0, float f_y=0.0, float x_0=0.0, float y_0=0.0, float skew=0.0);
    
    void render_image();
    
     void track_pose();
     void fit_vel_map();
     void fit_depth();
     void fit_camera_params();
     void fit_distorsion();
//     void fit_reflectance(); //, shading & illumimation
//     void fit_light_sources();
     void predict_next_frame();
     void display_images(); 
     //void compare_prediction();                                                                    // to prediction and gradient
     
    void next_frame(UMat nxt_frame){
        auto iter = frameNum%frames.size();                              // modulo iterator to loop round ImgCols
        resize(nxt_frame,frames[iter],Size(WORKING_IMG_SIZE) );
        track_pose();
        fit_vel_map();
        fit_depth();
        fit_camera_params();
        fit_distorsion();
//        fit_reflectance(); //, shading & illumimation
//        fit_light_sources();
        predict_next_frame();
        display_images();         
    }
/* sampling and projection
//image sampling function  
//initially just reduce image size,  -  later use logpolar sampling and   foveal super resolution
//interpolation function - generates new image from point cloud - use bicubic interpolation to avoid artefacts
//projection function
//condenses camera matrix, rotation, translation and distorsion to a single matrix, then applies them to the pixel cloud

void 	cv::projectPoints (InputArray objectPoints, InputArray rvec, InputArray tvec, InputArray cameraMatrix, InputArray distCoeffs, OutputArray imagePoints, OutputArray jacobian=noArray(), double aspectRatio=0)
// 	Projects 3D points to an image plane. More...
// 	note rvec and tvec are the rotation and translation in 3D.
// 	for our purpose the "object points" are the pixels of the previous image
*/
/*cost functions

    //photometric cost function //application to tracking //application to depth 


    //TV_L1 cost function //for dense maps //for sparse clouds - log polar or arbitary //for networks (eg SPH)

    //anisotropy


//total cost
//must penalize complexity and reverberation between maps - and any map complexity not required by images
//
*/
/*fitting protocol
//live online and off line versions - pick efficient sequence of param samples for optimisation.
//sequence - rot, trans, velocity map, depth, camMtrx, 1stDistorsion, reflectance & illumimation, higher distorsion
*/
/*record and display
//////////////////////////////////
//composite image with labels

//display images 

//record image files to folder
*/
/* program flow

// process command
// open video file

//while vid open, [nb record each step to file. Make only one step optimisation each frame]
{
    // scale & dewarp new frame (240p = 426x240, 16:9 for youtube, 24/25/30/48/50/60fps)
    
    // fit rotation
    
    // fit rot & translation
    
    // fit vel map, [nb previous frames inertia, update previous vel estimate and accel. ]
    
    // fit depth
    
    // fit camera params
    
    // fit  distorsion
    
    // fit reflectance, shading & illumimation
    
    // fit light sources
    
    
    // predict next frame.
    
    
    // display images 

}
*/
};//end of class rendering

UMat rendering::make_projection_matrix(float theta_x, float theta_y, float theta_z, float trans_x, float trans_y, float trans_z, float f_x, float f_y, float x_0, float y_0, float skew  ){
    Matx33f rot_x(1,0,0, 0,cos(theta_x),-sin(theta_x),  0,sin(theta_x),cos(theta_x) );
    Matx33f rot_y(cos(theta_y), 0, sin(theta_y),  0,1,0,  -sin(theta_y),0,cos(theta_x) ); 
    Matx33f rot_z(cos(theta_z), -sin(theta_z),0,  sin(theta_z), cos(theta_z),0, 0,0,1);
    Matx33f rot = rot_x * rot_y * rot_z;
    Matx34f extrinsic(rot[0],rot[1],rot[2],trans_x,     rot[3],rot[4],rot[5],trans_y,       rot[6],rot[7],rot[8],trans_z);
    Matx33f intrinsic(f_x, skew, x_0,   0, f_y, y_0,  0,0,1  ); 
    Mat proj_mat =  Mat(intrinsic * extrinsic);
    proj_mat.resize((size_t)4, (const Scalar)0) ;   //cv::Mat::resize ( size_t sz (new num rows), const Scalar &  s (value of new elements) )
    proj_mat.at<float>(4,4)=1.0;
    return UMat proj_Umat =  proj_mat.getUMat(CV_32FC3);    // needs to be 4x4 matrix for opencv perspectiveTransform()
}

Mat rendering::splat_points(Mat pointcloud , Mat srcImage , Mat targetImage = Mat::zeros(WORKING_IMG_SIZE, CV_32FC3) ){
    pointcloud.reshape(3,1);/* as 3chan, 1row, 102240 col */   
    srcImage.reshape(3,1);
    Mat dst_px_invdepth(WORKING_IMG_SIZE, CV_32FC1, -1.f ); //z channel, initialize to nan ? // size to refimage
    Mat dst_px = Mat::zeros(WORKING_IMG_SIZE, CV_32FC3  );//rgb channels //   size to refimage, & set to zero.

    for(int px=0; px<3*WORKING_IMG_PX_COUNT; px+=3){ // move in steps of 3 floats   // ideally do this by transferring a UMat to OpenCL
        int dst_x = floor(pointcloud.at<float>(px) );
        if (dst_x < 0 || dst_x > WORKING_IMG_COLS) continue;
        int dst_y = floor(pointcloud.at<float>(px+1) );
        if (dst_y < 0 || dst_y > WORKING_IMG_ROWS) continue;
        float dst_z = pointcloud.at<float>(px+2);
        if (dst_px_invdepth.at<float>(px)>0 && dst_px_invdepth.at<float>(px)<dst_z){
            dst_px_invdepth.at<float>(px)=dst_z;                                    // copy depth
            dst_px.at<Vec3f>(dst_x,dst_y)=srcImage.at<Vec3f>(px) - targetImage.at<Vec3f>(dst_x,dst_y) ;  //diff from targetImage
        }
    }
    return dst_px;  //result  splatted image, or diff from targetImage
};

Mat rendering::predict_image(float theta_x=0.0,float theta_y=0.0, float theta_z=0.0, float trans_x=0.0, float trans_y=0.0, float trans_z=0.0, float f_x=0.0, float f_y=0.0, float x_0=0.0, float y_0=0.0, float skew=0.0){
    // make point cloud from pixels of depthmap
    UMat pointCloud = UMat::ones(1, WORKING_IMG_PX_COUNT, CV_32FC3); // 1 collumn 
    vector<UMat>pointCloudVec(3);
    pointCloudVec[0] = UMat::ones(WORKING_IMG_SIZE, CV_32FC1);
    pointCloudVec[1] = UMat::ones(WORKING_IMG_SIZE, CV_32FC1);
    pointCloudVec[2] =  this->scene.depth_map;
    for(int row=0; row<WORKING_IMG_ROWS  ; row++){
        for(int col=0;col<WORKING_IMG_COLS  ; col++){
            pointCloudVec[0].at<float>(i,j)= col;
            pointCloudVec[1].at<float>(i,j)= row;
        }
    }
    pointCloudVec[0].reshape(1,WORKING_IMG_PX_COUNT);
    pointCloudVec[1].reshape(1,WORKING_IMG_PX_COUNT);
    pointCloudVec[2].reshape(1,WORKING_IMG_PX_COUNT);
    merge(pointCloudVec, pointCloud);
    // predict deformation of depth map (apply vel map) : ie apply pixelwise velocity to each point
    // ... 
    //combine transpose and projection matricies ?
    UMat projection_mat = make_projection_matrix(theta_x,theta_y,theta_z,trans_x,trans_y,trans_z,f_x,f_y,x_0,y_0,skew );
    // transform points to new camera frame
    perspectiveTransform(pointCloud,pointCloud,projection_mat);
    
    Mat refImagemat = Mat::ones(3,3,CV_32FC1);
    Mat targetImage = Mat::ones(3,3,CV_32FC1);
    return  splat_points(pointCloud.getMat(cv::ACCESS_READ), refImagemat) ;
};



void rendering::track_pose(){
    camera cam_r = cam_p = cam_yaw = cam;

    
    //fit_rotation
    loop{
        cam_r.transpose  .roll +=1;
        cam_p.transpose .pitch +=1;
        cam_yaw.transpose .yaw +=1;
        cur_error = error(newframe - render(cam);  // sum(magnitude(difference)) 
        grad_r = curr_error - error(newframe - render(cam_r) ); 
        grad_p = 
        grad_yaw = 
        
        ?? can we make a more efficient gradient of 'param' function ??  with option for 2'order gradient
        levenberg-Marquard => choose next param values 
        Threshold how good a fit / near optmum ?
            break loop => new_rpy()
    }
    //fit 6D transpose  - again but 6D
    loop{
    render(predicted + 1 step 6D +2 steps 6D)
    new_6D()
    }
};

void rendering::fit_vel_map(){
    // threshold pixelwise fit ?,  with image gradient anisotropy
    
    // TV_L1 flow ? with xyz vel
    
};

void rendering::fit_depth(){
    render(vary depth_map)
    
    // TV_L1 depth => reduce entropy in vel map
    
    
};

void rendering::fit_camera_params(){
    // centre and focal length 
    
};

void rendering::fit_distorsion(){
    // progressively fit => reduce total model entropy
    
};

//void rendering::fit_reflectance(){}; //, shading & illumimation
//void rendering::fit_light_sources(){};

void rendering::predict_next_frame(){
    // given scene model => frame + gradient in rotation 
    
};

void rendering::display_images(){
    // composite tiles and text labels
    // return composite image to app
}; 

void rendering::compare(){
    // for each pixel find 
};   
