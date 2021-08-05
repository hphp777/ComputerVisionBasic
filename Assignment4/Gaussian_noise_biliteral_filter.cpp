#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

using namespace cv;

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat BiliteralFilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r,  const char* opt);
Mat BiliteralFilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale

	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// Add noise to original image
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);

	// Denoise, using gaussian filter
	Mat Denoised_Gray = BiliteralFilter_Gray(noise_Gray, 5, 1, 1, 1,"zero-padding");
	Mat Denoised_RGB = BiliteralFilter_RGB(noise_RGB, 5, 1, 1, 1, "zero-padding");

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (Grayscale)", noise_Gray);

	namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised (Grayscale)", Denoised_Gray);

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised (RGB)", Denoised_RGB);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}

Mat BiliteralFilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) {


	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	Mat kernel;
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Gaussian Kernel Matrix
	// Fill code to initialize Gaussian filter kernel matrix
	kernel = Mat::zeros(kernel_size, kernel_size, CV_64F);
	denom = 0.0;
	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// 반복문을 돌때마다 denom을 꼭 초기화해주어야 한다.
			denom = 0.0;
			//kernel value가 input 이미지의 좌표에 따라 달라진다. 따라서 매번 
			//kernel value를 새롭게 계산해주어야 한다.
			for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
				for (int b = -n; b <= n; b++) {
					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
						//difference of distance
						double value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
						//difference of intensity
						value1 *= exp(-( pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2) / (2 * sigma_r * sigma_r) ));
						kernel.at<double>(a + n, b + n) = value1;

						denom += value1;
					}
				}
			}

			for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
				for (int b = -n; b <= n; b++) {
					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
						kernel.at<double>(a + n, b + n) /= denom;					
					}
				}
			}
			
			if (!strcmp(opt, "zero-padding")) {
				double sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1 += kernel.at<double>(n + a, n + b) * (double)(input.at<double>(i + a, j + b));
						}

					}
				}
				output.at<double>(i, j) = (double)sum1;
				//printf("%2f ", output.at<double>(i, j));
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "mirroring" boundary process:

						Fill the code:
						*/
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
						sum1 += kernel.at<double>(n + a, n + b) * (double)(input.at<double>(tempa, tempb));
					}
				}
				output.at<double>(i, j) = (double)sum1;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				double sum1 = 0.0;
				double sum2 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "adjustkernel" boundary process:

						Fill the code:
						*/
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += kernel.at<double>(n + a, n + b) * (double)(input.at<double>(i + a, j + b));
							sum2 += kernel.at<double>(n + a, n + b);
						}
					}
				}
				output.at<double>(i, j) = (double)(sum1 / sum2);

			}

		}
	}

	return output;
}

Mat BiliteralFilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char* opt) {

	Mat kernel;
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;

	// Initialiazing Gaussian Kernel Matrix
	// Fill code to initialize Gaussian filter kernel matrix

	kernel = Mat::zeros(kernel_size, kernel_size, CV_64FC3);
	denom = 0.0;

	
	Mat output = Mat::zeros(row, col, input.type());

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			// 반복문을 돌때마다 denom을 꼭 초기화해주어야 한다.
			denom = 0.0;
			//kernel value가 input 이미지의 좌표에 따라 달라진다. 따라서 매번 
			//kernel value를 새롭게 계산해주어야 한다.
			for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
				for (int b = -n; b <= n; b++) {
					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
						//결국 구현하는것 보다는 아래의 식을 구하는것이 연구의 핵심이구나.
						double value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
						//multiply the distance feature
						value1 *= exp(-(pow(input.at<double>(i, j) - input.at<double>(i + a, j + b), 2) / (2 * sigma_r * sigma_r)));
						kernel.at<double>(a + n, b + n) = value1;

						denom += value1;
					}
				}
			}

			for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
				for (int b = -n; b <= n; b++) {
					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
						kernel.at<double>(a + n, b + n) /= denom;

					}
				}
			}

			if (!strcmp(opt, "zero-padding")) {
				double sumR = 0.0;
				double sumG = 0.0;
				double sumB = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sumB += kernel.at<double>(n + a, n + b) * (input.at<Vec3d>(i + a, j + b)[0]);
							sumG += kernel.at<double>(n + a, n + b) * (input.at<Vec3d>(i + a, j + b)[1]);
							sumR += kernel.at<double>(n + a, n + b) * (input.at<Vec3d>(i + a, j + b)[2]);
						}

					}
				}
				output.at<Vec3d>(i, j)[0] = sumB;
				output.at<Vec3d>(i, j)[1] = sumG;
				output.at<Vec3d>(i, j)[2] = sumR;
			}

			else if (!strcmp(opt, "mirroring")) {
				double sumR = 0.0;
				double sumG = 0.0;
				double sumB = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "mirroring" boundary process:

						Fill the code:
						*/
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
						sumB += kernel.at<double>(n + a, n + b) * (input.at<Vec3d>(tempa, tempb)[0]);
						sumG += kernel.at<double>(n + a, n + b) * (input.at<Vec3d>(tempa, tempb)[1]);
						sumR += kernel.at<double>(n + a, n + b) * (input.at<Vec3d>(tempa, tempb)[2]);
					}
				}
				output.at<Vec3d>(i, j)[0] = sumB;
				output.at<Vec3d>(i, j)[1] = sumG;
				output.at<Vec3d>(i, j)[2] = sumR;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				double sum1R = 0.0;
				double sum1G = 0.0;
				double sum1B = 0.0;
				double sum2 = 0.0;
				for (int a = -n; a <= n; a++) { // for each kernel window
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "adjustkernel" boundary process:

						Fill the code:
						*/
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1B += kernel.at<double>(n + a, n + b) * (input.at<Vec3d>(i + a, j + b)[0]);
							sum1G += kernel.at<double>(n + a, n + b) * (input.at<Vec3d>(i + a, j + b)[1]);
							sum1R += kernel.at<double>(n + a, n + b) * (input.at<Vec3d>(i + a, j + b)[2]);
							sum2 += kernel.at<double>(n + a, n + b);
						}
					}
				}
				output.at<Vec3d>(i, j)[0] = sum1B / sum2;
				output.at<Vec3d>(i, j)[1] = sum1G / sum2;
				output.at<Vec3d>(i, j)[2.] = sum1R / sum2;

			}

		}
	}

	return output;
}