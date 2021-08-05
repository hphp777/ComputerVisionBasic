- Purpose of this code
 This code is made to remove Gaussian noise using biliteral filter. 

- How to run this code
 Load 'Gaussian_noise_gaussian_filter.cpp' with an input image. 

- How to adjust parameters (if any)
Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
 const Mat input: Give input image to give Gaussian noise.
 double mean: Give mean value for the Gaussian Function(Probability of the noise).
 double sigma: Give sigma value for the Gaussian Function(Probability of the noise).
Mat BiliteralFilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r,  const char* opt);
Mat BiliteralFilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);
 const Mat input: Give Mat on which Gaussian filter will be applied.
 int n: Give kernel size. When the value is n, the kernel size is (2*n+1)*(2*n+1). In this code, give 3 and 7 to compare the results from these.
 double sigma_t: Give sigma value for the Gaussian filter
 double sigma_s: Give sigma value for the Gaussian filter
 const char *opt: give string variable to process boundary. "zero-padding", "mirroring", "adjustkernel"

- How to define default parameters
Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
 const Mat input: Give input image to give Gaussian noise. In this code, it is 'lena.jpg'.
 double mean: Give mean value for the Gaussian Function(Probability of the noise). In this code, I gave 0 to make zero-mean AWGN. 
 double sigma: Give sigma value for the Gaussian Function(Probability of the noise). In this code I gave 1.
Mat BiliteralFilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r,  const char* opt);
Mat BiliteralFilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);
 const Mat input: Give Mat on which Gaussian filter will be applied.
 int n: Give kernel size. When the value is n, the kernel size is (2*n+1)*(2*n+1). 
 double sigma_t: Give sigma value for the Gaussian filter
 double sigma_s: Give sigma value for the Gaussian filter
 const char *opt: give string variable to process boundary. "zero-padding", "mirroring", "adjustkernel"


