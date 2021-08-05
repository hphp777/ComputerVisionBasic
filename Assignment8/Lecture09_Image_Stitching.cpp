#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

int init();
void find_correspondent_points();
void stitching();
void stitching_RANSAC(int k, float threshold, int loop);

template <typename T>
Mat cal_affine(vector<Point2f> img1, vector<Point2f> img2);
void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int diff_x, int diff_y, float alpha);
double euclidDistance(Mat& vec1, Mat& vec2);

int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
int nearestNeighbor_second(int exclude, Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);

Mat input1;
Mat input2;
Mat input1_gray, input2_gray;

// Compute keypoints and descriptor from the source image in advance
vector<KeyPoint> keypoints1;
Mat descriptors1;
vector<KeyPoint> keypoints2;
Mat descriptors2;

FeatureDetector* detector = new SiftFeatureDetector(
	0,		// nFeatures
	4,		// nOctaveLayers
	0.04,	// contrastThreshold
	10,		// edgeThreshold
	1.6		// sigma
);

DescriptorExtractor* extractor = new SiftDescriptorExtractor();

// Find nearest neighbor pairs
vector<Point2f> srcPoints;
vector<Point2f> dstPoints;

int main() {
	
	if (init() == -1) return -1;

	find_correspondent_points();

	//Using matched points, We have to calculate Affine Trnasformation Matrix

	//Case1. Use All the corresponding points, calculate transformation matrix without using RANSAC
	//stitching();

	//Case2. Use estimated transformation maxtrix using RANSAC
	stitching_RANSAC(3, 0.5, 10); // k=3
	//stitching_RANSAC(4, 0.5, 10); // k=4

	waitKey(0);

	return 0;
}

int init() { //read two images and convert into the grayscale images
	input1 = imread("input1.jpg", CV_LOAD_IMAGE_COLOR);
	input2 = imread("input2.jpg", CV_LOAD_IMAGE_COLOR);

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	//resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	//resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));

	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);
}

void find_correspondent_points() {

	// Create a image for displaying mathing keypoints
	Size size = input2.size();
	Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);

	// input2 -> input1
	input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height)));
	input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));

	detector->detect(input1_gray, keypoints1); //detect corner
	extractor->compute(input1_gray, keypoints1, descriptors1); //descripter: information around the corner.
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size());

	// Detect keypoints(corner points)
	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);

	printf("input2 : %zd keypoints are found.\n", keypoints2.size());

	//Use both of refinement algorithms
	bool crossCheck = true;
	bool ratio_threshold = true;

	//두 이미지속 코너포인트의 갯수가 동일할 필요는 없음
	findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);
	printf("%zd keypoints are matched.\n", srcPoints.size());

}

template <typename T>
Mat cal_affine(vector<Point2f> img1, vector<Point2f> img2) { //using randomly selected points

	int number_of_points = img1.size();

	Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
	Mat b(2 * number_of_points, 1, CV_32F);

	Mat M_trans, temp, affineM;

	// initialize matrix
	for (int i = 0; i < number_of_points; i++) {
		M.at<T>(2 * i, 0) = img1[i].x;
		M.at<T>(2 * i, 1) = img1[i].y;
		M.at<T>(2 * i, 2) = 1;
		M.at<T>(2 * i, 3) = M.at<T>(2 * i, 4) = M.at<T>(2 * i, 5) = 0;
		M.at<T>(2 * i + 1, 0) = M.at<T>(2 * i + 1, 1) = M.at<T>(2 * i + 1, 2) = 0;
		M.at<T>(2 * i + 1, 3) = img1[i].x;
		M.at<T>(2 * i + 1, 4) = img1[i].y;
		M.at<T>(2 * i + 1, 5) = 1;

		b.at<T>(2 * i) = img2[i].x;
		b.at<T>(2 * i + 1) = img2[i].y;
	}

	// (M^T * M)^(−1) * M^T * b ( * : Matrix multiplication)
	transpose(M, M_trans); 
	invert(M_trans * M, temp); 
	affineM = temp * M_trans * b; 

	return affineM;
}

//합쳐진 이미지의 각 픽셀의 intensity를 구하는 함수
void blend_stitching(const Mat I1, const Mat I2, Mat& I_f, int diff_x, int diff_y, float alpha) {

	int bound_x = I1.rows + diff_x;
	int bound_y = I1.cols + diff_y;

	int col = I_f.cols;
	int row = I_f.rows;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// for check validation of I1 & I2
			bool cond1 = (i < bound_x && i > diff_x) && (j < bound_y && j > diff_y) ? true : false;
			bool cond2 = I_f.at<Vec3b>(i, j) != Vec3b(0, 0, 0) ? true : false;

			// I2 is already in I_f by inverse warping
			// So, It is not necessary to check that only I2 is valid
			// if both are valid
			if (cond1 && cond2) {
				I_f.at<Vec3b>(i, j) = alpha * I1.at<Vec3b>(i - diff_x, j - diff_y) + (1 - alpha) * I_f.at<Vec3b>(i, j);
			}
			// only I1 is valid
			else if (cond1) {
				I_f.at<Vec3b>(i, j) = I1.at<Vec3b>(i - diff_x, j - diff_y);
			}
		}
	}
}

void stitching() { //code from lecture2

	// height(row), width(col) of each image(RGB)
	// I1_row == I2_row, I1_col == I2_col
	const float I1_row = input1_gray.rows;
	const float I1_col = input1_gray.cols;
	const float I2_row = input2_gray.rows;
	const float I2_col = input2_gray.cols;

	//calculate affine Matrix A12, A21
	//left -> right I2의 integer point의 intensity를 구할때 사용한다.
	Mat A12 = cal_affine<float>(srcPoints, dstPoints);
	//right -> left I2를 I1에 맞추어 변환할때 사용한다.
	Mat A21 = cal_affine<float>(dstPoints, srcPoints);

	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));
	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * I2_row + A21.at<float>(5));
	Point2f p3(A21.at<float>(0) * I2_col + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * I2_row + A21.at<float>(5));
	Point2f p4(A21.at<float>(0) * I2_col + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * 0 + A21.at<float>(5));

	int bound_u = (int)round(min(0.0f, min(p1.y, p4.y)));
	int bound_b = (int)round(max(I1_row - 1, max(p2.y, p3.y)));
	int bound_l = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_r = (int)round(max(I1_col - 1, max(p3.x, p4.x)));

	int diff_x = abs(bound_u); //y좌표의 어퍼바운드
	int diff_y = abs(bound_l); //x좌표의 어퍼바운드

	// initialize merged image
	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_8UC3, Scalar(0));

	// inverse warping with bilinear interplolation

	for (int i = bound_u; i <= bound_b; i++) {
		for (int j = bound_l; j <= bound_r; j++) {
			// Image2를 Affine transform하기 전의 원본 이미지로부터 intensity값을 추출
			float x = A12.at<float>(0) * j + A12.at<float>(1) * i + A12.at<float>(2) - bound_l;
			float y = A12.at<float>(3) * j + A12.at<float>(4) * i + A12.at<float>(5) - bound_u;

			//변환된 floating point의 주변 4개의 점을 구한다.
			float y1 = floor(y); //floor함수: 내림
			float y2 = ceil(y); //ceil함수: 올림
			float x1 = floor(x);
			float x2 = ceil(x);

			float mu = y - y1;
			float lambda = x - x1;

			// linear interpolation을 y축으로 한번, x축으로 한번 총 2번 수행한다.
			if (x1 >= 0 && x2 < I2_col && y1 >= 0 && y2 < I2_row)
				I_f.at<Vec3b>(i - bound_u, j - bound_l) = lambda * (mu * input2.at<Vec3b>(y2, x2) + (1 - mu) * input2.at<Vec3b>(y1, x2)) +
				(1 - lambda) * (mu * input2.at<Vec3b>(y2, x1) + (1 - mu) * input2.at<Vec3b>(y1, x1));
		}
	}

	//image stitching with blend
	blend_stitching(input1, input2, I_f, diff_x, diff_y, 0.5);

	
	::namedWindow("result");
	::imshow("result", I_f);

}

void stitching_RANSAC(int k, float threshold, int loop) {
	//We should sample random 3 or 4 data from srcPoints and dstPoint
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(0, srcPoints.size()-1);
	
	int max_inlier = 0;
	int max_index;

	vector<Point2f> src_inlier;
	vector<Point2f> dst_inlier;

	vector<vector<Point2f>> total_src_inlier;
	vector<vector<Point2f>> total_dst_inlier;


	Mat A12, A21;
	for (int s = 0; s < loop; s++) {
		
		vector<Point2f> srcSample;
		vector<Point2f> dstSample;
		int inlier_counter = 0;
		src_inlier.clear();
		dst_inlier.clear();

		for (int i = 0; i < k; i++) {
			int index = dis(gen);
			srcSample.push_back(srcPoints[index]);
			dstSample.push_back(dstPoints[index]);
		}
		//calculate affine Matrix A12, A21
		A12 = cal_affine<float>(srcSample, dstSample);
		A21 = cal_affine<float>(dstSample, srcSample);

		//Using this parameter, convert srcPoint to dstPoint.
		//Compare dstPoint with converted point.
		for (int i = 0; i < srcPoints.size(); i++) {

			Point2f converted(A12.at<float>(0) * srcPoints[i].y + A12.at<float>(1) * srcPoints[i].x + A12.at<float>(2),
				A12.at<float>(3) * srcPoints[i].y + A12.at<float>(4) * srcPoints[i].x + A12.at<float>(5));

			//Euclid distance
			float distance = sqrt(pow(converted.x - dstPoints[i].x, 2) + pow(converted.y - dstPoints[i].y, 2));
			if (distance < threshold) {
				inlier_counter++;
				src_inlier.push_back(srcPoints[i]);
				dst_inlier.push_back(dstPoints[i]);
			}
			
		}

		total_src_inlier.push_back(src_inlier);
		total_dst_inlier.push_back(dst_inlier);

		if (inlier_counter > max_inlier) {
			max_inlier = inlier_counter;
			max_index = s;
		}
	}

	//Using inliers, recompute transformation matrix 
	A12 = cal_affine<float>(total_src_inlier[max_index], total_dst_inlier[max_index]);
	A21 = cal_affine<float>(total_dst_inlier[max_index], total_src_inlier[max_index]);
	
	//Using the measure of euclidian distance
	//If that distance is lower than threshold, count up 1.
	//Through the iteration, find the maximum value and save the sample point and inlier.
		// height(row), width(col) of each image(RGB)
	// I1_row == I2_row, I1_col == I2_col
	const float I1_row = input1.rows;
	const float I1_col = input1.cols;
	const float I2_row = input2.rows;
	const float I2_col = input2.cols;

	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));
	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * 0 + A21.at<float>(4) * I2_row + A21.at<float>(5));
	Point2f p3(A21.at<float>(0) * I2_col + A21.at<float>(1) * I2_row + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * I2_row + A21.at<float>(5));
	Point2f p4(A21.at<float>(0) * I2_col + A21.at<float>(1) * 0 + A21.at<float>(2), A21.at<float>(3) * I2_col + A21.at<float>(4) * 0 + A21.at<float>(5));


	int bound_u = (int)round(min(0.0f, min(p1.y, p4.y)));
	int bound_b = (int)round(max(I1_row - 1, max(p2.y, p3.y)));
	int bound_l = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_r = (int)round(max(I1_col - 1, max(p3.x, p4.x)));

	int diff_x = abs(bound_u); //y좌표의 어퍼바운드
	int diff_y = abs(bound_l); //x좌표의 어퍼바운드

	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_8UC3, Scalar(0));

	for (int i = bound_u; i <= bound_b; i++) {
		for (int j = bound_l; j <= bound_r; j++) {
			// Image2를 Affine transform하기 전의 원본 이미지로부터 intensity값을 추출
			float x = A12.at<float>(0) * j + A12.at<float>(1) * i + A12.at<float>(2) - bound_l;
			float y = A12.at<float>(3) * j + A12.at<float>(4) * i + A12.at<float>(5) - bound_u;

			//변환된 floating point의 주변 4개의 점을 구한다.
			float y1 = floor(y); //floor함수: 내림
			float y2 = ceil(y); //ceil함수: 올림
			float x1 = floor(x);
			float x2 = ceil(x);

			float mu = y - y1;
			float lambda = x - x1;

			// linear interpolation을 y축으로 한번, x축으로 한번 총 2번 수행한다.
			if (x1 >= 0 && x2 < I2_col && y1 >= 0 && y2 < I2_row)
				I_f.at<Vec3b>(i - bound_u, j - bound_l) = lambda * (mu * input2.at<Vec3b>(y2, x2) + (1 - mu) * input2.at<Vec3b>(y1, x2)) +
				(1 - lambda) * (mu * input2.at<Vec3b>(y2, x1) + (1 - mu) * input2.at<Vec3b>(y1, x1));

			printf("%u\n", I_f.at<Vec3b>(i - bound_u, j - bound_l));
		}
	}
	//image stitching with blend
	//diff_x = 변환된 이미지의 upper bound
	//diff_y = 변환된 이미지의 left bound
	blend_stitching(input1, input2, I_f, diff_x, diff_y, 0.5);

	::namedWindow("result");
	::imshow("result", I_f);
	
}

double euclidDistance(Mat& vec1, Mat& vec2) { //Compute on row vector
	double sum = 0.0;
	int dim = vec1.cols;
	for (int i = 0; i < dim; i++) {
		sum += (vec1.at<float>(0, i) - vec2.at<float>(0, i)) * (vec1.at<float>(0, i) - vec2.at<float>(0, i));
	}

	return sqrt(sum);
}

/**
* Find the index of nearest neighbor point from keypoints.
*/
//1행짜리 Mat을 인수로 받음
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;

	//find minimum distance
	//compute 
	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor
		//Each discripter is row vector  
		// vec vs v
		double distance = euclidDistance(vec, v);
		if (distance < minDist) {
			neighbor = i;
			minDist = distance;
		}

	}

	return neighbor;
}
int nearestNeighbor_second(int exclude, Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;

	//find minimum distance
	//compute 
	for (int i = 0; i < descriptors.rows; i++) {
		if (i != exclude) {
			Mat v = descriptors.row(i);		// each row of descriptor
		//Each discripter is row vector  
		// vec vs v
			double distance = euclidDistance(vec, v);
			if (distance < minDist) {
				neighbor = i;
				minDist = distance;
			}

		}

	}

	return neighbor;
}

/**
* Find pairs of points with the smallest distace between them
*/
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i];
		Mat desc1 = descriptors1.row(i); //information around keypoint pt1 4x4 size

		//Compute all the points in keypoinsts2.
		int nn = nearestNeighbor(desc1, keypoints2, descriptors2);

		// Refine matching points using ratio_based thresholding
		if (ratio_threshold) {
			//We should find second nearest neighbor.

			int nn2 = nearestNeighbor_second(nn, desc1, keypoints2, descriptors2);

			Mat first = descriptors2.row(nn);
			double first_dist = euclidDistance(desc1, first);
			Mat second = descriptors2.row(nn2);
			double second_dist = euclidDistance(desc1, second);

			double ratio = first_dist / second_dist;

			if (ratio > 0.5) continue;
		}

		// Refine matching points using cross-checking
		if (crossCheck) {
			//From the nn, we should find nearest neighbor and it should be i.
			KeyPoint pt2 = keypoints2[nn];
			Mat desc2 = descriptors2.row(nn);
			int L = nearestNeighbor(desc2, keypoints1, descriptors1);
			if (L != i) continue;
		}
		//If nearest point is veryfied
		KeyPoint pt2 = keypoints2[nn];
		//Parralel array
		srcPoints.push_back(pt1.pt);
		dstPoints.push_back(pt2.pt);
	}

	int a = 0;
	int b = 0;
}