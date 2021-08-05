- Purpose of this code
 This code is made to make RGB image sharper by boosting high-frequency components.

- How to run this code
 Through the Visual Studio, load 'Gaussian_grey.cpp' and run this with the 'lena.jpg'.

- How to adjust parameters (if any)

 Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);
 const Mat input: Give input Mat on which the uniform mean filter will be applied.
 int n: Give the size of kernel ( Kernel size: (2n+1)*(2n+1) )
 const char* opt: Choose between "zero-paddle" or "mirroring" or "adjustkernel" for the boundary processing.
 float sigmaS: This is the standard deviation for Gaussian filter. We can give appropriate value manually.
 float sigmaT: This is the standard deviation for Gaussian filter. We can give appropriate value manually.

 Mat unsharp_masking(Mat input, Mat A_G, float k);
 Mat input: Give original image which will be manipulated.
 Mat A_G: Give image on which Gaussian filter was applied. (Blured image)
 float k: Arbitrary parameter to scale overall intensity of image on which low-pass filtering was applied. It is a value between 0.0-1.0.

- How to define default parameters

 Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);
 const Mat input: Give input Mat on which the uniform mean filter will be applied.
 int n: Give the size of kernel ( Kernel size: (2n+1)*(2n+1) ). I gave 1 in this code. 
 const char* opt: Choose between "zero-paddle" or "mirroring" or "adjustkernel" for the boundary processing.
 float sigmaS: This is the standard deviation for Gaussian filter. We can give appropriate value. In this code, I gave 1. 
 float sigmaT: This is the standard deviation for Gaussian filter. We can give appropriate value. In this code, I gave 1.

 Mat unsharp_masking(Mat input, Mat A_G, float k);
 Mat input: Give original image which will be manipulated.
 Mat A_G: Give image on which Gaussian filter was applied. (Blured image)
 float k: Arbitrary parameter to scale overall intensity of image on which low-pass filtering was applied. It is a value between 0.0-1.0. In this code, I gave 0.2.