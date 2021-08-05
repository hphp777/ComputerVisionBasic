#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

//Gaussian filter seperation version.

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

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat output;

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Input_RGB", WINDOW_AUTOSIZE);
	imshow("Input_RGB", input);
	output = gaussianfilter(input, 1, 1, 1, "zero-paddle"); //Boundary process: zero-paddle, mirroring, adjustkernel

	namedWindow("Gaussian Filter", WINDOW_AUTOSIZE);
	imshow("Gaussian Filter", output);


	waitKey(0);

	return 0;
}


Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;
	float kernelvalue;
	float* Wt = new float[kernel_size];
	float* Ws = new float[kernel_size];

	// Initialiazing Kernel Matrix 
	// ���⼭���ʹ� Ŀ�� ������ �� �ٸ���.
	// kernelvalue = w(s,t)
	kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);
	denom = 0.0;

	// �ϴ� ������ ���鼭 �� ��ǥ�� kernel value�� �Ҵ��Ѵ�.
	// �״��� �������� ���Ѵ�.
	// �� ��ǥ���� �� �������� �������ش�.
	// ������ ��쿡�� 2�� �����ϸ� �ȴ�.
	// x��ǥ���� ����
	// �̸� ���� �迭�� �ʿ��ϴ�. 
	for (int b = -n; b <= n; b++) {
		float value1 = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2))));
		Wt[n + b] = value1;
		denom += value1;
	}

	for (int b = -n; b <= n; b++) {
		Wt[n + b] /= denom;
	}
	denom = 0.0;
	//�״��� y��ǥ ����
	for (int a = -n; a <= n; a++) {
		float value2 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
		Ws[n + a] = value2;
		denom += value2;
	}

	for (int a = -n; a <= n; a++) {
		Ws[n + a] /= denom;
	}

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {


			if (!strcmp(opt, "zero-paddle")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with Zero-paddle boundary process:

						Fill the code:
						*/
						//�׳� �ȿ��͸� �״�� ���ϸ� �ȴ�. ������ 0�̹Ƿ�
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1_r += Ws[n + a] * Wt[n + b] * (float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += Ws[n + a] * Wt[n + b] * (float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += Ws[n + a] * Wt[n + b] * (float)(input.at<C>(i + a, j + b)[2]);
						
						}
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "mirroring" process:

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
						sum1_r += Ws[n + a] * Wt[n + b] * (float)(input.at<C>(tempa, tempb)[0]);
						sum1_g += Ws[n + a] * Wt[n + b] * (float)(input.at<C>(tempa, tempb)[1]);
						sum1_b += Ws[n + a] * Wt[n + b] * (float)(input.at<C>(tempa, tempb)[2]);
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}


			else if (!strcmp(opt, "adjustkernel")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				float sum2_r = 0.0;
				float sum2_g = 0.0;
				float sum2_b = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "adjustkernel" process:

						Fill the code:
						*/
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1_r += Ws[n + a] * Wt[n + b] * (float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += Ws[n + a] * Wt[n + b] * (float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += Ws[n + a] * Wt[n + b] * (float)(input.at<C>(i + a, j + b)[2]);
							sum2_r += Ws[n + a] * Wt[n + b];
							sum2_g += Ws[n + a] * Wt[n + b];
							sum2_b += Ws[n + a] * Wt[n + b];
						}
					}
				}
				output.at<C>(i, j)[0] = (G)(sum1_r/sum2_r);
				output.at<C>(i, j)[1] = (G)(sum1_g/sum2_g);
				output.at<C>(i, j)[2] = (G)(sum1_b/sum2_b);
			}
		}
	}
	free(Wt);
	free(Ws);

	return output;
}