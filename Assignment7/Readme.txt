- Purpose of this code
 This code is made to compare two pictures through checking the corner points. 
 It the corner points are correspoding between two images, the inpit two images are corresponded.

- How to run this code
 Through loading 'FeatureMatchingUsingSIFT.cpp' in Visual Studio with the input two images.

- How to adjust parameters (if any)

 double euclidDistance(Mat& vec1, Mat& vec2);
 Give two 1 column Mat variables in which the same number of points are stored. Iterating these points, the function calculates the Euclidian distance.  

 int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
 Mat& vec: Give a descripter of keypoint.
 vector<KeyPoint>& keypoints: Give a vector which store key points in it. Among them, the nearest point will be found.
 Mat& descriptors: Give empty Mat variable to compute descripter of key points in each 'keypoints'.

 int nearestNeighbor_second(int exclude, Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
 int exclude: Give index to exclude when computing the distance between two descripters.
 Mat& vec: Give a descripter of keypoint.
 vector<KeyPoint>& keypoints: Give a vector which store key points in it. Among them, the nearest point will be found.
 Mat& descriptors: Give empty Mat variable to compute descripter of key points in each 'keypoints'.

 void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);
 vector<KeyPoint>& keypoints1: Extracted key points in image1.
 vector<KeyPoint>& keypoints2: Extracted key points in image2.
 Mat& descriptors1: Give descripters of keypoints in image1. The number of row of descripter is equal to the number of key points in image1. The column is composed of information around keypoint1 pt1 4x4 size
 Mat& descriptors2: Give descripters of keypoints in image2. The number of row of descripter is equal to the number of key points in image2. The column is composed of information around keypoint1 pt1 4x4 size
 vector<Point2f>& srcPoints, vector<Point2f>& dstPoints: empty vectors to push matched key points
 bool crossCheck: Give true if you want to crooCheck or falase
 bool ratio_threshold: Give true if you want to check ratio between the nearest point and 2nd nearest point or give false I you don't.

- How to define default parameters

 double euclidDistance(Mat& vec1, Mat& vec2);
 Give two 1 column Mat variables in which the same number of points are stored. Iterating these points, the function calculates the Euclidian distance.  

 int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
 Mat& vec: Give a descripter of keypoint.
 vector<KeyPoint>& keypoints: Give a vector which store key points in it. Among them, the nearest point will be found.
 Mat& descriptors: Give empty Mat variable to compute descripter of key points in each 'keypoints'.

 int nearestNeighbor_second(int exclude, Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
 int exclude: Give index to exclude when computing the distance between two descripters.
 Mat& vec: Give a descripter of keypoint.
 vector<KeyPoint>& keypoints: Give a vector which store key points in it. Among them, the nearest point will be found.
 Mat& descriptors: Give empty Mat variable to compute descripter of key points in each 'keypoints'.

 void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);
 vector<KeyPoint>& keypoints1: Give Extracted key points in image1.
 vector<KeyPoint>& keypoints2: Give Extracted key points in image2.
 Mat& descriptors1: Give descripters of keypoints in image1. The number of row of descripter is equal to the number of key points in image1. The column is composed of information around keypoint1 pt1 4x4 size
 Mat& descriptors2: Give descripters of keypoints in image2. The number of row of descripter is equal to the number of key points in image1. The column is composed of information around keypoint1 pt1 4x4 size
 vector<Point2f>& srcPoints, vector<Point2f>& dstPoints: empty vectors to push matched key points
 bool crossCheck: Give true if you want to crooCheck or falase. In this code, I gave true.
 bool ratio_threshold: Give true if you want to check ratio between the nearest point and 2nd nearest point or give false I you don't. In this code, I gave true.

