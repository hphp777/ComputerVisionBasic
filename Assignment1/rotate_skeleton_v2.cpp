#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
	Mat input, rotated;
	
	// Read each image
	input = imread("lena.jpg");

	// Check for invalid input
	if (!input.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	// original image
	namedWindow("image");
	imshow("image", input);

	rotated = myrotate<Vec3b>(input, 45, "nearest");

	// rotated image
	namedWindow("rotated");
	imshow("rotated", rotated);

	waitKey(0);

	return 0;
}

float dist(float xx, float yy, float x, float y) {
	float d1 = abs(xx - x);
	float d2 = abs(yy - y);
	return sqrt(d1 * d1 + d2 * d2);
}

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180;

	float sq_row = ceil(row * sin(radian) + col * cos(radian));
	float sq_col = ceil(col * sin(radian) + row * cos(radian));

	Mat output = Mat::zeros(sq_row, sq_col, input.type());

	for (int i = 0; i < sq_row; i++) {
		for (int j = 0; j < sq_col; j++) {
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {
				const int x1 = floor(x);
				const int x2 = ceil(x);
				const int y1 = floor(y);
				const int y2 = ceil(y);
				int xx[2] = { x1, x2 };
				int yy[2] = { y1, y2 };
				if (!strcmp(opt, "nearest")) {
					//인접한 4개의 점 중에서 가장 가까운 점의 intensity값을 대입.					
					float min = 100;
					int tx, ty;
					for (int m = 0; m < 2; m++) {
						for (int n = 0;n < 2;n++) {
							float distance = dist((float)xx[n], (float)yy[m], x, y);
							if (distance < min) {
								min = distance;
								ty = m;
								tx = n;
							}
						}
					}
					//가장 거리가 가까운 점의 intensity를 inverse 
					output.at<Vec3b>(i, j) = input.at<Vec3b>(yy[ty], xx[tx]);

				}
				else if (!strcmp(opt, "bilinear")) {
					//가까운 두 점을 이어서 방정식을 만들고 거기에 대입.
					//먼저 각 x'의 좌표를 구한다(2번)
					float d1 = abs(x - xx[0]);
					float d2 = abs(x - xx[1]);
					Vec3b f1 = d2 * input.at<Vec3b>(yy[0], xx[0]) + d1 * input.at<Vec3b>(yy[0], xx[1]);
					Vec3b f2 = d2 * input.at<Vec3b>(yy[1], xx[0]) + d1 * input.at<Vec3b>(yy[1], xx[1]);

					//새로구한 픽셀을 다시 한번
					float d3 = abs(y - yy[0]);
					float d4 = abs(y - yy[1]);
					Vec3b f = d4 * f1 + d3 * f2;

					output.at<Vec3b>(i, j) = f;

				}
			}
		}
	}

	return output;
}