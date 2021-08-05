#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat LaplacianFilter(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);
	output = LaplacianFilter(input); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);


	waitKey(0);

	return 0;
}


Mat LaplacianFilter(const Mat input) {

	Mat kernel = Mat::zeros(3, 3, input.type());

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N
	int tempa;
	int tempb;

	Mat output = Mat::zeros(row, col, input.type());

	// Initialiazing 2 Kernel Matrix with 3x3 size for Laplacian Filtering
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)

	kernel.at<G>(0, 0) = 0;
	kernel.at<G>(0, 1) = 1;
	kernel.at<G>(0, 2) = 0;
	kernel.at<G>(1, 0) = 1;
	kernel.at<G>(1, 1) = -4;
	kernel.at<G>(1, 2) = 1;
	kernel.at<G>(2, 0) = 0;
	kernel.at<G>(2, 1) = 1;
	kernel.at<G>(2, 2) = 0;


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
					float sum_r = 0.0;
					float sum_g = 0.0;
					float sum_b = 0.0;
					for (int a = -n; a <= n; a++) { // for each kernel window
						for (int b = -n; b <= n; b++) {

							if (i + a > row - 1) {  //mirroring for the border pixels
								tempa = i - a;
							}
							else if (i + a < 0) {
								tempa = -(i + a);
							}
							else {
								tempa = i + a;
							}
							if (j + b > col - 1) {
								tempb = j - b;
							}
							else if (j + b < 0) {
								tempb = -(j + b);
							}
							else {
								tempb = j + b;
							}
							sum_r += (float)kernel.at<G>(1 + a, 1 + b) * (float)(input.at<C>(tempa, tempb)[0]);
							sum_g += (float)kernel.at<G>(1 + a, 1 + b) * (float)(input.at<C>(tempa, tempb)[1]);
							sum_b += (float)kernel.at<G>(1 + a, 1 + b) * (float)(input.at<C>(tempa, tempb)[2]);
						}
					}
					output.at<C>(i, j)[0] = 0.5 * G(sum_r);
					output.at<C>(i, j)[1] = 0.5 * G(sum_g);
					output.at<C>(i, j)[2] = 0.5 * G(sum_b);

				}

			}
		}
	}
	return output;
}