#include "hist_func.h"

/*
	grey scale image�� Histogram Matching ������� 
	�����ϴ� ����̴�. 
	���� ǳ���� �Կ��� �������� �̹����� 
	contranst�� ���� �Ǹ��� �̹����� �������� �ΰ�
	������ �̹������� �ִ��� �� �̹����� ����ϵ��� �����ϴ� �۾��̴�.
	�� ������ �����ϱ� ���� �̹����� transformation function�� �ΰ��� �ʿ��ϴ�.
	�ϳ��� ��ȯ�� �̹���, �׸��� �ϳ��� �� ���� �̹����� HE�� �����ϱ� ���� ���̴�.
	��ǲ �̹����� ���� T, G-1�� �ι��� �����ؼ� ����� ����Ѵ�.
	���� HE�� 2�� �����ϴ°������� �� ���������� �������� trasformation�Լ��� 
	���Լ��� ���ϴ� ������ ���ľ� �Ѵٴ� ���̴�.
	���Լ��� 1:1������ �ƴϸ� ���Ҽ� ���� ���� ��Ģ�ε�, �־��� 
	transformation function�� �׷��� ���� ��찡 �Ϲ����̴�.
	���� ���Լ��� ���� �Ŀ� �̸� 1:1�������� �ٽ� �������ִ� ������ �ʿ��ϴ�.
*/

void hist_match(Mat& input, Mat& equalized, G* trans_func, float* CDF);
void hist_match_final(Mat& input, Mat& equalized, G* trans_func, float* CDF);
Mat drawHist(Mat img);

int main() {
	

	//input, output �̹����� ������ Matrix�� �����Ѵ�.
	Mat input=imread("blurred_input.jpg", CV_LOAD_IMAGE_COLOR); //���⿡ �ٲٰ��� �ϴ� �̹����� �ִ´�.
	Mat input_grey;
	Mat refer = imread("input.jpg", CV_LOAD_IMAGE_COLOR); //������ �̹���
	Mat refer_grey;
	

	cvtColor(input, input_grey, CV_RGB2GRAY);
	cvtColor(refer, refer_grey, CV_RGB2GRAY);

	Mat output1 = input_grey.clone();
	Mat output2 = refer_grey.clone();
	
	// PDF or transfer function txt files
	FILE* f_PDF; //PDF of original Image
	FILE* f_equalized_PDF; //PDF of output image

	FILE* f_trans_func_input;
	FILE* f_trans_func_refer;
	FILE* f_trans_func_result;

	float *PDF_input = cal_PDF(input_grey);
	float *PDF_refer = cal_PDF(refer_grey);
	float *CDF_input = cal_CDF(input_grey);
	float *CDF_refer = cal_CDF(refer_grey);

	fopen_s(&f_PDF, "HM_f_PDF_grey.txt", "w+");
	fopen_s(&f_equalized_PDF, "HM_f_equalized_PDF.txt", "w+");

	fopen_s(&f_trans_func_input, "HM_f_trans_func_input.txt", "w+");
	fopen_s(&f_trans_func_refer, "HM_f_trans_func_refer.txt", "w+");
	fopen_s(&f_trans_func_result, "HM_f_trans_func_inverse.txt", "w+");

	// transfer function
	G trans_func_input[L] = { 0 };
	G trans_func_refer[L] = { 0 };
	G trans_func_refer_inverse[L] = { 0 }; //use inverse matrix

	// histogram equalization
	hist_match(input_grey, output1, trans_func_input, CDF_input); //use this output later.
	hist_match(refer_grey, output2, trans_func_refer, CDF_refer);

	Mat result = output1.clone();


	// equalized PDF(�߰����)
	float* equalized_PDF_input = cal_PDF(output1); // input 1����ȯ PDF
	float* equalized_PDF_refer = cal_PDF(output2); // reference�̹����� ���

	// Make inverse function
	G i = 0;
	for (int k = 0; k < L; k++) {
		trans_func_refer_inverse[k] = 0;
	}

	for (int k = 0; k < L; k++) {
		//index, data �ڹٲٱ�
		while (i <= trans_func_refer[k]) {
			if (trans_func_refer_inverse[i] == 0) {
				trans_func_refer_inverse[i] = k;
			}
			i++;
		}
	}
	//���������� ���Լ�, s(intermediate result)�� �̿��ؼ� result�� ����� ����
	float* PDF_output1 = cal_PDF(output1);
	float* CDF_output1 = cal_CDF(output1);

	hist_match_final(output1, result, trans_func_refer_inverse, CDF_output1);

	//equalized PDF(�������)
	float* equalized_PDF_result = cal_PDF(result);

	for (int i = 0; i < L; i++) {
		// write PDF
		fprintf(f_PDF, "%d\t%f\n", i, PDF_input[i]);
		fprintf(f_equalized_PDF, "%d\t%f\n", i, equalized_PDF_result[i]);

		// write transfer functions
		fprintf(f_trans_func_input, "%d\t%d\n", i, trans_func_input[i]);
		fprintf(f_trans_func_refer, "%d\t%d\n", i, trans_func_refer[i]);
		fprintf(f_trans_func_result, "%d\t%d\n", i, trans_func_refer_inverse[i]);
	}
	

	// memory release
	free(PDF_input);
	free(CDF_input);
	free(PDF_refer);
	free(CDF_refer);
	free(PDF_output1);
	free(CDF_output1);
	fclose(f_PDF);
	fclose(f_equalized_PDF);
	fclose(f_trans_func_input);
	fclose(f_trans_func_result);
	fclose(f_trans_func_result);

	////////////////////// Show each image ///////////////////////

	namedWindow("input_gs", WINDOW_AUTOSIZE);
	imshow("input_gs", input_grey);

	namedWindow("intermediate_gs", WINDOW_AUTOSIZE);
	imshow("intermediate_gs", output1);

	namedWindow("refer_gs", WINDOW_AUTOSIZE);
	imshow("refer_gs", refer_grey);

	namedWindow("Matched Image", WINDOW_AUTOSIZE);
	imshow("Matched Image", result);

	//������׷� ���
	namedWindow("Histogram_input", WINDOW_AUTOSIZE);
	imshow("Histogram_input", drawHist(input_grey));
	
	namedWindow("Histogram_result", WINDOW_AUTOSIZE);
	imshow("Histogram_result", drawHist(result));
	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

void hist_match(Mat& input, Mat& equalized, G* trans_func, float* CDF) {
	// compute transfer function
	for (int i = 0; i < L; i++)
		trans_func[i] = (G)((L - 1) * CDF[i]);

	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}
void hist_match_final(Mat& input, Mat& equalized, G* trans_func, float* CDF) {
	// perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			equalized.at<G>(i, j) = trans_func[input.at<G>(i, j)];
}

Mat drawHist(Mat img) {
	MatND histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;
	calcHist(&img, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//Plot the histogram
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImg(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImg.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < number_bins; i++) {
		line(histImg, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
	}

	return histImg;
}