#include "hist_func.h"

void hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat equalized_RGB = input.clone();

	// PDF or transfer function txt files
	FILE *f_equalized_PDF_RGB, *f_PDF_RGB;
	FILE *f_trans_func_eq_RGB;
	
	fopen_s(&f_PDF_RGB, "HERGB_PDF_RGB.txt", "w+");
	fopen_s(&f_equalized_PDF_RGB, "HERGB_equalized_PDF_RGB.txt", "w+");
	fopen_s(&f_trans_func_eq_RGB, "HERGB_trans_func_eq_RGB.txt", "w+");

	float **PDF_RGB = cal_PDF_RGB(input);	// PDF of Input image(RGB) : [L][3]
	float **CDF_RGB = cal_CDF_RGB(input);	// CDF of Input image(RGB) : [L][3]

	G trans_func_eq_RGB[L][3] = { 0 };		// transfer function

	// histogram equalization on RGB image
	hist_eq_Color(input, equalized_RGB, trans_func_eq_RGB, CDF_RGB);

	// equalized PDF (RGB)
	float** equalized_PDF_RGB = cal_PDF_RGB(equalized_RGB);
	
	for (int i = 0; i < L; i++) {
		// color image print
		for (int k = 0; k < 3; k++) {
			fprintf(f_PDF_RGB, "%d\t%f ", i, PDF_RGB[i][k]);
			fprintf(f_equalized_PDF_RGB, "%d\t%f ", i, equalized_PDF_RGB[i][k]);
			// write transfer functions
			fprintf(f_trans_func_eq_RGB, "%d\t%d\n", i, trans_func_eq_RGB[i][k]);
		}
		fprintf(f_PDF_RGB, "\n");
		fprintf(f_equalized_PDF_RGB, "\n");
		fprintf(f_trans_func_eq_RGB, "\n");
	}

	// memory release
	free(PDF_RGB);
	free(CDF_RGB);
	fclose(f_PDF_RGB);
	fclose(f_equalized_PDF_RGB);
	fclose(f_trans_func_eq_RGB);

	////////////////////// Show each image ///////////////////////

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Equalized_RGB", WINDOW_AUTOSIZE);
	imshow("Equalized_RGB", equalized_RGB);

	//히스토그램 출력
	Mat* hist = new Mat[3];
	hist = drawHist_RGB_PDF(input);

	namedWindow("histogram_input_B", WINDOW_AUTOSIZE);
	imshow("histogram_input_B", (Mat)hist[0]);
	namedWindow("histogram_input_G", WINDOW_AUTOSIZE);
	imshow("histogram_input_G", (Mat)hist[1]);
	namedWindow("histogram_input_R", WINDOW_AUTOSIZE);
	imshow("histogram_input_R", (Mat)hist[2]);

	hist = new Mat[3];
	hist = drawHist_RGB_PDF(equalized_RGB);

	namedWindow("histogram_equalized_B", WINDOW_AUTOSIZE);
	imshow("histogram_equalized_B", (Mat)hist[0]);
	namedWindow("histogram_equalized_G", WINDOW_AUTOSIZE);
	imshow("histogram_equalized_G", (Mat)hist[1]);
	namedWindow("histogram_equalized_R", WINDOW_AUTOSIZE);
	imshow("histogram_equalized_R", (Mat)hist[2]);

	//////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}

// histogram equalization on 3 channel image
void hist_eq_Color(Mat &input, Mat &equalized, G(*trans_func)[3], float **CDF) {

	////////////////////////////////////////////////
	//											  //
	// How to access multi channel matrix element //
	//											  //
	// if matrix A is CV_8UC3 type,				  //
	// A(i, j, k) -> A.at<Vec3b>(i, j)[k]		  //
	//											  //
	////////////////////////////////////////////////

	// compute transfer function
	for (int k = 0 ; k < 3 ; k++)
		for (int i = 0; i < L; i++)
			trans_func[k][i] = (G)((L - 1) * CDF[i][k]);

	// perform the transfer function
	for (int k = 0 ; k < 3 ; k++)
		for (int i = 0; i < input.rows; i++)
			for (int j = 0; j < input.cols; j++)
				equalized.at<Vec3b>(i, j)[k] = trans_func[k][input.at<Vec3b>(i, j)[k]];

}