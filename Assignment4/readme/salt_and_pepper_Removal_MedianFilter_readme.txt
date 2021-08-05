- Purpose of this code
 This code is made ro remove salt-and-pepper noise using median filter

- How to run this code
 Load 'Salt_and_Pepper_Removal_MedianFilter.cpp' in Visual Studio and run this with 'lena.jpg' 

- How to adjust parameters (if any)
 Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp);
 const Mat input: Give input image
 float ps: density of salt noise. Give value between 0-1.
 float pp: density of pepper noise. Give value between 0-1.

 Mat Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt);
 Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt);  
 const Mat input: Give input image
 int n: Give variable to decide kernel size. For example, if n is given, the kernel size will ne 2*n+1.
 const char *opt: Give string variable to process image boundary. It is a value among "zero-padding" or "adjustkernel" or "mirroring"

- How to define default parameters
 Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp);
 const Mat input: Give input image.
 float ps: density of salt noise. Give value between 0-1. In this code, set 0.1.
 float pp: density of pepper noise. Give value between 0-1. In this code, set 0.1.

 Mat Salt_pepper_noise_removal_Gray(const Mat input, int n, const char *opt);
 Mat Salt_pepper_noise_removal_RGB(const Mat input, int n, const char *opt);  
 const Mat input: Give input image. In this code, it is 'lena.jpg'.
 int n: Give variable to decide kernel size. For example, if n is given, the kernel size will ne 2*n+1. I gave 2 as a default value. 
 const char *opt: Give string variable to process image boundary. It is a value among "zero-padding" or "adjustkernel" or "mirroring". I can give whatever I want.

