/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_TRACKING_HPP
#define OPENCV_TRACKING_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{

class CV_EXPORTS_W DenseOpticalFlow : public Algorithm
{
public:
    /** @brief Calculates an optical flow.

    @param I0 first 8-bit single-channel input image.
    @param I1 second input image of the same size and the same type as prev.
    @param flow computed flow image that has the same size as prev and type CV_32FC2.
     */
    CV_WRAP virtual void calc( InputArray I0, InputArray I1, InputOutputArray flow ) = 0;
    /** @brief Releases all inner buffers.
    */
    CV_WRAP virtual void collectGarbage() = 0;
};






/** @brief "Dual TV L1" Optical Flow Algorithm.

The class implements the "Dual TV L1" optical flow algorithm described in @cite Zach2007 and
@cite Javier2012 .
Here are important members of the class that control the algorithm, which you can set after
constructing the class instance:

-   member double tau
    Time step of the numerical scheme.

-   member double lambda
    Weight parameter for the data term, attachment parameter. This is the most relevant
    parameter, which determines the smoothness of the output. The smaller this parameter is,
    the smoother the solutions we obtain. It depends on the range of motions of the images, so
    its value should be adapted to each image sequence.

-   member double theta
    Weight parameter for (u - v)\^2, tightness parameter. It serves as a link between the
    attachment and the regularization terms. In theory, it should have a small value in order
    to maintain both parts in correspondence. The method is stable for a large range of values
    of this parameter.

-   member int nscales
    Number of scales used to create the pyramid of images.

-   member int warps
    Number of warpings per scale. Represents the number of times that I1(x+u0) and grad(
    I1(x+u0) ) are computed per scale. This is a parameter that assures the stability of the
    method. It also affects the running time, so it is a compromise between speed and
    accuracy.

-   member double epsilon
    Stopping criterion threshold used in the numerical scheme, which is a trade-off between
    precision and running time. A small value will yield more accurate solutions at the
    expense of a slower convergence.

-   member int iterations
    Stopping criterion iterations number used in the numerical scheme.

C. Zach, T. Pock and H. Bischof, "A Duality Based Approach for Realtime TV-L1 Optical Flow".
Javier Sanchez, Enric Meinhardt-Llopis and Gabriele Facciolo. "TV-L1 Optical Flow Estimation".
*/
class CV_EXPORTS_W DualTVL1OpticalFlow : public DenseOpticalFlow
{
public:
    //! @brief Time step of the numerical scheme
    /** @see setTau */
    CV_WRAP virtual double getTau() const = 0;
    /** @copybrief getTau @see getTau */
    CV_WRAP virtual void setTau(double val) = 0;
    //! @brief Weight parameter for the data term, attachment parameter
    /** @see setLambda */
    CV_WRAP virtual double getLambda() const = 0;
    /** @copybrief getLambda @see getLambda */
    CV_WRAP virtual void setLambda(double val) = 0;
    //! @brief Weight parameter for (u - v)^2, tightness parameter
    /** @see setTheta */
    CV_WRAP virtual double getTheta() const = 0;
    /** @copybrief getTheta @see getTheta */
    CV_WRAP virtual void setTheta(double val) = 0;
    //! @brief coefficient for additional illumination variation term
    /** @see setGamma */
    CV_WRAP virtual double getGamma() const = 0;
    /** @copybrief getGamma @see getGamma */
    CV_WRAP virtual void setGamma(double val) = 0;
    //! @brief Number of scales used to create the pyramid of images
    /** @see setScalesNumber */
    CV_WRAP virtual int getScalesNumber() const = 0;
    /** @copybrief getScalesNumber @see getScalesNumber */
    CV_WRAP virtual void setScalesNumber(int val) = 0;
    //! @brief Number of warpings per scale
    /** @see setWarpingsNumber */
    CV_WRAP virtual int getWarpingsNumber() const = 0;
    /** @copybrief getWarpingsNumber @see getWarpingsNumber */
    CV_WRAP virtual void setWarpingsNumber(int val) = 0;
    //! @brief Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and running time
    /** @see setEpsilon */
    CV_WRAP virtual double getEpsilon() const = 0;
    /** @copybrief getEpsilon @see getEpsilon */
    CV_WRAP virtual void setEpsilon(double val) = 0;
    //! @brief Inner iterations (between outlier filtering) used in the numerical scheme
    /** @see setInnerIterations */
    CV_WRAP virtual int getInnerIterations() const = 0;
    /** @copybrief getInnerIterations @see getInnerIterations */
    CV_WRAP virtual void setInnerIterations(int val) = 0;
    //! @brief Outer iterations (number of inner loops) used in the numerical scheme
    /** @see setOuterIterations */
    CV_WRAP virtual int getOuterIterations() const = 0;
    /** @copybrief getOuterIterations @see getOuterIterations */
    CV_WRAP virtual void setOuterIterations(int val) = 0;
    //! @brief Use initial flow
    /** @see setUseInitialFlow */
    CV_WRAP virtual bool getUseInitialFlow() const = 0;
    /** @copybrief getUseInitialFlow @see getUseInitialFlow */
    CV_WRAP virtual void setUseInitialFlow(bool val) = 0;
    //! @brief Step between scales (<1)
    /** @see setScaleStep */
    CV_WRAP virtual double getScaleStep() const = 0;
    /** @copybrief getScaleStep @see getScaleStep */
    CV_WRAP virtual void setScaleStep(double val) = 0;
    //! @brief Median filter kernel size (1 = no filter) (3 or 5)
    /** @see setMedianFiltering */
    CV_WRAP virtual int getMedianFiltering() const = 0;
    /** @copybrief getMedianFiltering @see getMedianFiltering */
    CV_WRAP virtual void setMedianFiltering(int val) = 0;

    /** @brief Creates instance of cv::DualTVL1OpticalFlow*/
    CV_WRAP static Ptr<DualTVL1OpticalFlow> create(
                                            double tau = 0.25,
                                            double lambda = 0.15,
                                            double theta = 0.3,
                                            int nscales = 5,
                                            int warps = 5,
                                            double epsilon = 0.01,
                                            int innnerIterations = 30,
                                            int outerIterations = 10,
                                            double scaleStep = 0.8,
                                            double gamma = 0.0,
                                            int medianFiltering = 5,
                                            bool useInitialFlow = false);
};

/** @brief Creates instance of cv::DenseOpticalFlow
*/
CV_EXPORTS_W Ptr<DualTVL1OpticalFlow> createOptFlow_DualTVL1();







}//namespace cv
