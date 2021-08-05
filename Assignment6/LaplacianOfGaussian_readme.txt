- Purpose of this code
 This code is made to detect edges in input images.

- How to run this code
 Load LaplacianOfGaussian.cpp and run this code with the input image "lena.jpg". 
 
- How to adjust parameters (if any)
Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
 int n: Give kernel size. When the value is n, the kernel size is (2*n+1)*(2*n+1). In this code, give 3 and 7 to compare the results from these.
 double sigma_t: Give sigma value for the Gaussian filter
 double sigma_s: Give sigma value for the Gaussian filter
 bool normalize: if the user want to normalize the kernel values, give true or false.

Mat get_Laplacian_Kernel();

Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s);
Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s);
 const Mat input: Give Mat on which Gaussian filter will be applied.
 int n: Give kernel size. When the value is n, the kernel size is (2*n+1)*(2*n+1). In this code, give 3 and 7 to compare the results from these.
 double sigma_t: Give sigma value for the Gaussian filter
 double sigma_s: Give sigma value for the Gaussian filter

Mat Laplacianfilter(const Mat input);
Mat Laplacianfilter_RGB(const Mat input);
 const Mat input: Give Mat on which Laplacian filter will be applied.

Mat Mirroring(const Mat input, int n);
Mat Mirroring_RGB(const Mat input, int n);
 const Mat input: Give Mat which will be mirrored.
 int n: Give amount which user want to mirror.


- How to define default parameters
Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
 int n: Give kernel size. When the value is n, the kernel size is (2*n+1)*(2*n+1). In this code, give 3 and 7 to compare the results from these.
 double sigma_t: Give sigma value for the Gaussian filter
 double sigma_s: Give sigma value for the Gaussian filter
 bool normalize: if the user want to normalize the kernel values, give true or false. In this code, I gave true.

Mat get_Laplacian_Kernel();

Mat Gaussianfilter(const Mat input, int n, double sigma_t, double sigma_s);
Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s);
 const Mat input: Give Mat on which Gaussian filter will be applied.
 int n: Give kernel size. When the value is n, the kernel size is (2*n+1)*(2*n+1). In this code, give 3 and 7 to compare the results from these.
 double sigma_t: Give sigma value for the Gaussian filter
 double sigma_s: Give sigma value for the Gaussian filter

Mat Laplacianfilter(const Mat input);
Mat Laplacianfilter_RGB(const Mat input);
 const Mat input: Give Mat on which Laplacian filter will be applied.

Mat Mirroring(const Mat input, int n);
Mat Mirroring_RGB(const Mat input, int n);
 const Mat input: Give Mat which will be mirrored.
 int n: Give amount which user want to mirror. In this code, I gave 2.

