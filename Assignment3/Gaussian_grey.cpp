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

Mat gaussianfilter(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;


	cvtColor(input, input_gray, CV_RGB2GRAY);



	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);
	output = gaussianfilter(input_gray, 5, 7, 7, "adjustkernel"); //Boundary process: zero-paddle, mirroring, adjustkernel

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
		Wt[n+b] = value1;
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
				float sum1 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with Zero-paddle boundary process:

						Fill the code:
						*/
						//�׳� �ȿ��͸� �״�� ���ϸ� �ȴ�. ������ 0�̹Ƿ�
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1 += Ws[n+a] * Wt[n+b] * (float)(input.at<G>(i + a, j + b));
						}
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
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
						sum1 += Ws[n + a] * Wt[n + b] * (float)(input.at<G>(tempa, tempb));
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}


			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int a = -n; a <= n; a++) {
					for (int b = -n; b <= n; b++) {

						/* Gaussian filter with "adjustkernel" process:

						Fill the code:
						*/
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += Ws[n + a] * Wt[n + b] * (float)(input.at<G>(i + a, j + b));
							sum2 += Ws[n + a] * Wt[n + b];
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1 / sum2);
			}
		}
	}
	free(Wt);
	free(Ws);

	return output;
}