#include <gsl/gsl>
#include <opencv2/core.hpp>         // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>   // Gaussian Blur
using namespace std;
//using namespace cv;

//cv::matrix

/////////////////////////////////////////////////////////////////////////////////
//Operations on arrays

//cv::gemm - matrix multiplication
void 	cv::gemm (InputArray src1, InputArray src2, double alpha, InputArray src3, double beta, OutputArray dst, int flags=0)
The function cv::gemm performs generalized matrix multiplication similar to the gemm functions in BLAS level 3. 
For example, 
gemm(src1, src2, alpha, src3, beta, dst, GEMM_1_T + GEMM_3_T) 
corresponds to
dst = alpha*src1.t()*src2 + beta*src3.t();


void cv::transform 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	m 
	) 	
src	input array that must have as many channels (1 to 4) as m.cols or m.cols-1.
dst	output array of the same size and depth as src; it has as many channels as m.rows.
m	transformation 2x2 or 2x3 floating-point matrix.



/////////////////////////////////////////////////////////////////////////////////
//Geometric Image Transformations
//Image processing

void cv::perspectiveTransform 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	m 
	) 	
src	input two-channel or three-channel floating-point array; each element is a 2D/3D vector to be transformed.
dst	output array of the same size and type as src.
m	3x3 or 4x4 floating-point transformation matrix. 


void cv::warpPerspective 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	M,
		Size  	dsize,
		int  	flags = INTER_LINEAR,
		int  	borderMode = BORDER_CONSTANT,
		const Scalar &  	borderValue = Scalar() 
	) 	
Parameters
    src	input image.
    dst	output image that has the size dsize and the same type as src .
    M	3×3 transformation matrix.
    dsize	size of the output image.
    flags	combination of interpolation methods (INTER_LINEAR or INTER_NEAREST) and the optional flag WARP_INVERSE_MAP, that sets M as the inverse transformation ( dst→src ).
    borderMode	pixel extrapolation method (BORDER_CONSTANT or BORDER_REPLICATE).
    borderValue	value used in case of a constant border; by default, it equals 0.
    

void cv::remap 	( 	InputArray  	src,
		OutputArray  	dst,
		InputArray  	map1,
		InputArray  	map2,
		int  	interpolation,
		int  	borderMode = BORDER_CONSTANT,
		const Scalar &  	borderValue = Scalar() 
	) 	
Parameters
    src	Source image.
    dst	Destination image. It has the same size as map1 and the same type as src .
 
    map1	The first map of either (x,y) points or just x values having the type CV_16SC2 , CV_32FC1, or CV_32FC2. See convertMaps for details on converting a floating point representation to fixed-point for speed.
 
    map2	The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty map if map1 is (x,y) points), respectively.

    interpolation	Interpolation method (see cv::InterpolationFlags). The method INTER_AREA is not supported by this function.
 
    borderMode	Pixel extrapolation method (see cv::BorderTypes). When borderMode=BORDER_TRANSPARENT, it means that the pixels in the destination image that corresponds to the "outliers" in the source image are not modified by the function.

    borderValue	Value used in case of a constant border. By default, it is 0. 
	

void cv::resize 	( 	InputArray  	src,
		OutputArray  	dst,
		Size  	dsize,
		double  	fx = 0,
		double  	fy = 0,
		int  	interpolation = INTER_LINEAR 
	) 	
Parameters
    src	input image.
    dst	output image; it has the size dsize (when it is non-zero) or the size computed from src.size(), fx, and fy; the type of dst is the same as of src.
    dsize	output image size; if it equals zero, it is computed as:

    dsize = Size(round(fx*src.cols), round(fy*src.rows))
    Either dsize or both fx and fy must be non-zero.
    fx	scale factor along the horizontal axis; when it equals 0, it is computed as

    (double)dsize.width/src.cols
    fy	scale factor along the vertical axis; when it equals 0, it is computed as

    (double)dsize.height/src.rows
    interpolation	interpolation method, see cv::InterpolationFlags
    
    
enum cv::InterpolationFlags
INTER_NEAREST 	nearest neighbor interpolation
INTER_LINEAR 	bilinear interpolation
INTER_CUBIC 	bicubic interpolation
INTER_AREA 	resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
INTER_LANCZOS4 	Lanczos interpolation over 8x8 neighborhood
INTER_MAX 	mask for interpolation codes
WARP_FILL_OUTLIERS 	flag, fills all of the destination image pixels. If some of them correspond to outliers in the source image, they are set to zero
WARP_INVERSE_MAP 	flag, inverse transformation
For example, cv::linearPolar or cv::logPolar transforms:
    flag is not set: dst(ρ,ϕ)=src(x,y)
    flag is set: dst(x,y)=src(ρ,ϕ)


    
void cv::logPolar 	( 	InputArray  	src,
		OutputArray  	dst,
		Point2f  	center,
		double  	M,
		int  	flags 
	) 	
src	Source image
dst	Destination image. It will have same size and type as src.
center	The transformation center; where the output precision is maximal
M	Magnitude scale parameter. It determines the radius of the bounding circle to transform too.
flags	A combination of interpolation methods, see cv::InterpolationFlags
	
	
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Camera Calibration and 3D Reconstruction



double 	cv::calibrateCamera (InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, OutputArray stdDeviationsIntrinsics, OutputArray stdDeviationsExtrinsics, OutputArray perViewErrors, int flags=0, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON))
 	Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern. More...
 
double 	cv::calibrateCamera (InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags=0, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON))
 
void 	cv::calibrationMatrixValues (InputArray cameraMatrix, Size imageSize, double apertureWidth, double apertureHeight, double &fovx, double &fovy, double &focalLength, Point2d &principalPoint, double &aspectRatio)
 	Computes useful camera characteristics from the camera matrix. More...
 
void 	cv::composeRT (InputArray rvec1, InputArray tvec1, InputArray rvec2, InputArray tvec2, OutputArray rvec3, OutputArray tvec3, OutputArray dr3dr1=noArray(), OutputArray dr3dt1=noArray(), OutputArray dr3dr2=noArray(), OutputArray dr3dt2=noArray(), OutputArray dt3dr1=noArray(), OutputArray dt3dt1=noArray(), OutputArray dt3dr2=noArray(), OutputArray dt3dt2=noArray())
 	Combines two rotation-and-shift transformations. More...
 
void 	cv::computeCorrespondEpilines (InputArray points, int whichImage, InputArray F, OutputArray lines)
 	For points in an image of a stereo pair, computes the corresponding epilines in the other image. More...
 
void 	cv::convertPointsFromHomogeneous (InputArray src, OutputArray dst)
 	Converts points from homogeneous to Euclidean space. More...
 
void 	cv::convertPointsHomogeneous (InputArray src, OutputArray dst)
 	Converts points to/from homogeneous coordinates. More...
 
void 	cv::convertPointsToHomogeneous (InputArray src, OutputArray dst)
 	Converts points from Euclidean to homogeneous space. More...




int 	cv::recoverPose (InputArray E, InputArray points1, InputArray points2, InputArray cameraMatrix, OutputArray R, OutputArray t, InputOutputArray mask=noArray())
 	Recover relative camera rotation and translation from an estimated essential matrix and the corresponding points in two images, using cheirality check. Returns the number of inliers which pass the check. More...
 
int 	cv::recoverPose (InputArray E, InputArray points1, InputArray points2, OutputArray R, OutputArray t, double focal=1.0, Point2d pp=Point2d(0, 0), InputOutputArray mask=noArray())

void 	cv::reprojectImageTo3D (InputArray disparity, OutputArray _3dImage, InputArray Q, bool handleMissingValues=false, int ddepth=-1)
 	Reprojects a disparity image to 3D space. More...

	
Mat cv::findHomography 	( 	InputArray  	srcPoints,
		InputArray  	dstPoints,
		int  	method = 0,
		double  	ransacReprojThreshold = 3,
		OutputArray  	mask = noArray(),
		const int  	maxIters = 2000,
		const double  	confidence = 0.995 
	)
Finds a perspective transformation between two planes. 
Parameters
    srcPoints	Coordinates of the points in the original plane, a matrix of the type CV_32FC2 or vector<Point2f> .
    dstPoints	Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or a vector<Point2f> .
    method	Method used to computed a homography matrix. The following methods are possible:

        0 - a regular method using all the points
        RANSAC - RANSAC-based robust method
        LMEDS - Least-Median robust method
        RHO - PROSAC-based robust method

    ransacReprojThreshold	Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC and RHO methods only). That is, if

    ∥dstPointsi−convertPointsHomogeneous(H∗srcPointsi)∥>ransacReprojThreshold
    then the point i is considered an outlier. If srcPoints and dstPoints are measured in pixels, it usually makes sense to set this parameter somewhere in the range of 1 to 10.
    mask	Optional output mask set by a robust method ( RANSAC or LMEDS ). Note that the input mask values are ignored.
    maxIters	The maximum number of RANSAC iterations, 2000 is the maximum it can be.
    confidence	Confidence level, between 0 and 1.
	
	
//////////
//cv::fisheye
 	The methods in this namespace use a so-called fisheye camera model.
 	
void 	cv::fisheye::undistortImage (InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, InputArray Knew=cv::noArray(), const Size &new_size=Size())
 	Transforms an image to compensate for fisheye lens distortion. More...
 
void 	cv::fisheye::undistortPoints (InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, InputArray R=noArray(), InputArray P=noArray())
 	Undistorts 2D points using fisheye model. More...
	
	
	

/////////////////////////////////////////////////////////////////////////////////
//cv::MatOp Class Reference

virtual void cv::MatOp::matmul 	( 	const MatExpr &  	expr1,
		const MatExpr &  	expr2,
		MatExpr &  	res 
	) 		const

virtual void cv::MatOp::multiply 	( 	const MatExpr &  	expr1,
		const MatExpr &  	expr2,
		MatExpr &  	res,
		double  	scale = 1 
	) 		const
	
	
virtual void cv::MatOp::multiply 	( 	const MatExpr &  	expr1,
		double  	s,
		MatExpr &  	res 
	) 		const
	
//////////////////////////////////////////////////////////////////////////////////////
//cv::MatExpr Class Reference

Mat cv::MatExpr::cross 	( 	const Mat &  	m	) 	const

MatExpr cv::MatExpr::diag 	( 	int  	d = 0	) 	const

double cv::MatExpr::dot 	( 	const Mat &  	m	) 	const

MatExpr cv::MatExpr::inv 	( 	int  	method = DECOMP_LU	) 	const

MatExpr cv::MatExpr::mul 	( 	const MatExpr &  	e,
		double  	scale = 1 
	) 		const
	
MatExpr cv::MatExpr::mul 	( 	const Mat &  	m,
		double  	scale = 1 
	) 		const
	
///////////////////////////////////////////////////////////////////////////////////////
//cv::Matx< _Tp, m, n > Class Template Reference

Template class for small matrices whose type and size are known at compilation time. 
(m*n matrix with upto 22 elements)
Matx33f m(1, 2, 3,
          4, 5, 6,
          7, 8, 9);
cout << sum(Mat(m*m.t())) << endl;



//////////////////////////////////////////////////////////////////////////////////////


reshape()

t()  transpose the matrix 


//////////////////////////////////////////////////////////////////////////////////////
//cv::Affine3< T > Class Template Reference
//core/include/opencv2/core/affine.hpp

cv::Affine3< T >::Affine3 	( 	const Vec3 &  	rvec,
		const Vec3 &  	t = Vec3::all(0) 
	) 	
	
Affine3 cv::Affine3< T >::rotate 	( 	const Mat3 &  	R	) 	const
a.rotate(R) is equivalent to Affine(R, 0) * a;

void cv::Affine3< T >::rotation 	( 	const Mat3 &  	R	) 	
Rotation matrix.

void cv::Affine3< T >::rotation 	( 	const Vec3 &  	rvec	) 	
Rodrigues vector.

Vec3 cv::Affine3< T >::rvec 	( 		) 	const
Rodrigues vector.

Affine3 cv::Affine3< T >::translate 	( 	const Vec3 &  	t	) 	const
a.translate(t) is equivalent to Affine(E, t) * a;

void cv::Affine3< T >::translation 	( 	const Vec3 &  	t	) 	

Vec3 cv::Affine3< T >::translation 	( 		) 	const

Mat4 cv::Affine3< T >::matrix


///////////////////////////////////////////////////////////////////////////////////////
//cv::UMat Class Reference
core/include/opencv2/core/mat.hpp

UMat 	reshape (int cn, int rows=0) const
 	creates alternative matrix header for the same data, with different More...
 
UMat 	reshape (int cn, int newndims, const int *newsz) const

double cv::UMat::dot 	( 	InputArray  	m	) 	const

Mat cv::UMat::getMat 	( 	int  	flags	) 	const

