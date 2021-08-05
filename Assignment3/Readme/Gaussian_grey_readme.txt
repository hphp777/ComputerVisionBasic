- Purpose of this code
 This code is made to blur input gray scale image through Gaussian filter. It is one of the low pass filterings.

- How to run this code
 Through the Visual Studio, load 'Gaussian_grey.cpp' and run this with the 'lena.jpg'.

- How to adjust parameters (if any)
 Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);
 const Mat input: Give input Mat on which the uniform mean filter will be applied.
 int n: Give the size of kernel ( Kernel size: (2n+1)*(2n+1) )
 const char* opt: Choose between "zero-paddle" or "mirroring" or "adjustkernel" for the boundary processing.
 float sigmaS: This is the standard deviation for Gaussian filter. We can give appropriate value manually.
 float sigmaT: This is the standard deviation for Gaussian filter. We can give appropriate value manually.

- How to define default parameters
 Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);
 const Mat input: Give input Mat on which the uniform mean filter will be applied.
 int n: Give the size of kernel ( Kernel size: (2n+1)*(2n+1) )
 const char* opt: Choose between "zero-paddle" or "mirroring" or "adjustkernel" for the boundary processing.
 float sigmaS: This is the standard deviation for Gaussian filter. We can give appropriate value. In this code, I gave 1. 
 float sigmaT: This is the standard deviation for Gaussian filter. We can give appropriate value. In this code, I gave 1. 