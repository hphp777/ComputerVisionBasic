- Purpose of this code
 This is the simplest low-pass filte for gray scale image. This algorithm makes image blurry than the original. 
 As the filter size increases, the image become more blurry.

- How to run this code
 Through the Visual Studio, Load "MeanFilterGray.cpp" and run this code with input image"lena.jpg".

- How to adjust parameters (if any)
 Mat meanfilter(const Mat input, int n, const char* opt);
 const Mat input: Give input Mat on which the uniform mean filter will be applied.
 int n: Give the size of kernel ( Kernel size: (2n+1)*(2n+1) )
 const char* opt: Choose between "zero-paddle" or "mirroring" or "adjustkernel" for the boundary processing.

- How to define default parameters
 Mat meanfilter(const Mat input, int n, const char* opt);
 const Mat input: Give input Mat on which the uniform mean filter will be applied.
 int n: Give proper value for the kernel. For example, it can be 3.
 const char* opt: Choose between "zero-paddle" or "mirroring" or "adjustkernel" for the boundary processing. We can choose any option among them.

 