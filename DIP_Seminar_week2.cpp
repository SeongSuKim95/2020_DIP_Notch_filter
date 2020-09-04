#include "opencv2/opencv.hpp"
#include <iostream>
#include <ctime>
#include <math.h>

#define RadToDeg 57.29577951f
#define DegToRad 0.017453293f

using namespace std;
using namespace cv;

void MAT_AT()
{
	Mat img(1000, 1500, CV_8UC3, Scalar(0));
	//
	clock_t start, end;
	start = clock();
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (j < 500) {
				img.at<Vec3b>(i, j)[0] = 255; //b
			}
			else if (j < 1000) {
				img.at<Vec3b>(i, j)[1] = 255; //g
			}
			else {
				img.at<Vec3b>(i, j)[2] = 255; //r
			}
		}
	}
	end = clock();
	printf("%.3lf", (double)(end - start));
	imshow("AT", img);
	waitKey(0);

}

void MAT_PTR()
{
	Mat img(1000, 1500, CV_8UC3, Scalar(0));
	clock_t start, end;
	start = clock();

	for (int i = 0; i < img.rows; i++) {
		uchar* ptr_pixel = img.ptr<uchar>(i);
		for (int j = 0; j < img.cols; j++) {
			if (j < 500) {
				ptr_pixel[j * 3 + 0] = 255; //B
			}
			else if (j < 1000) {
				ptr_pixel[j * 3 + 1] = 255; //G
			}
			else {
				ptr_pixel[j * 3 + 2] = 255;//R
			}
		}
	}
	end = clock();
	printf("%.3lf", (double)(end - start));
	imshow("PTR", img);
	waitKey(0);
}

void MAT_DATA()
{
	Mat img(1000, 1500, CV_8UC3, Scalar(0));
	uchar* img_data = img.data;

	clock_t start, end;
	start = clock();
	for (int i = 0; i < img.rows;i++) {
		for (int j = 0; j < img.cols;j++) {
			if (j < 500) {
				img_data[i * img.cols * 3 + j * 3 + 0] = 255;
			}
			else if (j < 1000) {
				img_data[i * img.cols * 3 + j * 3 + 1] = 255;
			}
			else {
				img_data[i * img.cols * 3 + j * 3 + 2] = 255;
			}
		}
	}
	end = clock();
	printf("%.3lf", (double)(end - start));
	imshow("DATA", img);
	waitKey(0);
}

void zero_padding(Mat img, Mat& dst, Size size)
{
	dst = Mat(size, CV_8U, Scalar(0));

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			int x = j;
			int y = i;
			dst.at<uchar>(y, x) = img.at<uchar>(i, j);
		}
	}

}

void scaling(Mat img, Mat& dst, Size size)
{
	dst = Mat(size, CV_8U, Scalar(0)); // destination 

	double ratioY = (double)size.height / img.rows;
	double ratioX = (double)size.width / img.cols;

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {

			int x = (int)(j * ratioX);
			int y = (int)(i * ratioY);
			dst.at<uchar>(y, x) = img.at<uchar>(i, j);
		}
	}

}

void scaling_nearest(Mat img, Mat& dst, Size size)
{
	dst = Mat(size, CV_8U, Scalar(0));
	double ratioY = (double)size.height / img.rows;
	double ratioX = (double)size.width / img.cols;

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {

			int x = (int)cvRound(j / ratioX);
			int y = (int)cvRound(i / ratioY);
			dst.at<uchar>(i, j) = img.at<uchar>(y, x);
		}
	}
}

uchar Bilinear_interpolation(Mat img, double x, double y)
{
	//Exception masking
	if (x >= img.cols - 1) x = x - 1;
	if (y >= img.rows - 1) y = y - 1;

	//
	Point pt((int)x, (int)y);

	int A = img.at<uchar>(pt);
	int B = img.at<uchar>(pt + Point(0, 1));
	int C = img.at<uchar>(pt + Point(1, 0));
	int D = img.at<uchar>(pt + Point(1, 1));

	double alpha = y - pt.y;
	double beta = x - pt.x;

	int M1 = A + (int)cvRound(alpha * (B - A));
	int M2 = C + (int)cvRound(alpha * (D - C));
	int P = M1 + (int)cvRound(beta * (M2 - M1));

	return saturate_cast<uchar>(P);
}

void scaling_bilinear(Mat img, Mat& dst, Size size)
{
	dst = Mat(size, img.type(), Scalar(0));
	double ratio_Y = (double)size.height / img.rows;
	double ratio_X = (double)size.width / img.cols;

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {

			double y = i / ratio_Y;
			double x = j / ratio_X;
			dst.at<uchar>(i, j) = Bilinear_interpolation(img, x, y);
		}
	}

}

void Translation(Mat img, Mat& dst, Point pt)
{
	Rect rect(Point(0, 0), img.size());
	dst = Mat(img.size(), img.type(), Scalar(0));

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {

			Point dst_pt(j, i);
			Point img_pt = dst_pt - pt;
			if (rect.contains(img_pt))
				dst.at<uchar>(dst_pt) = img.at<uchar>(img_pt);
		}
	}

}

void rotation_center(Mat img, Mat& dst, double degree, Point pt)
{
	double radian = degree * DegToRad;
	double sin_lookup = sin(radian);
	double cos_lookup = cos(radian);

	Rect rect(Point(0, 0), img.size());

	dst = Mat(img.size(), img.type(), Scalar(0));

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {

			int j_center = j - pt.x;
			int i_center = i - pt.y;

			double x = j_center * cos_lookup + i_center * sin_lookup + pt.x;
			double y = -(j_center)*sin_lookup + i_center * cos_lookup + pt.y;

			if (rect.contains(Point2d(x, y))) // Region Exception
			{
				dst.at<uchar>(i, j) = Bilinear_interpolation(img, x, y);
			}
		}
	}
}

int main() {

	Mat image = imread("C:/Users/user/Desktop/DIP_Seminar/Lenna_gray.tif", 0);
	CV_Assert(image.data);

	Mat dst1, dst2, dst3, dst4, dst5, dst6;

	scaling(image, dst1, Size(256, 256));
	scaling(image, dst2, Size(1024, 1024));

	imshow("image", image);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	zero_padding(image, dst1, Size(1000, 1000));
	Translation(dst1, dst2, Point(300, 0));

	//Affine TF Speed Test
	clock_t start, end;

	//Rotation after Translation
	start = clock();
	Translation(dst2, dst3, Point(100, 50));
	rotation_center(dst3, dst4, 30, Point(0, 0));
	printf("%.5f\n", (float)(clock() - start) / CLOCKS_PER_SEC);

	//imshow("image", dst3);
	//waitKey();
	//imshow("image", dst4);
	//waitKey();

	//imwrite("C:/Users/user/Desktop/DIP_Seminar/week1/test.bmp", dst2);
	//imwrite("C:/Users/user/Desktop/DIP_Seminar/week1/translation1.bmp", dst3);
	//imwrite("C:/Users/user/Desktop/DIP_Seminar/week1/rotation1.bmp", dst4);

	//Translation after Rotation
	start = clock();
	rotation_center(dst2, dst5, 30, Point(0, 0));
	Translation(dst5, dst6, Point(61, 93));
	printf("%.5f\n", (float)(clock() - start) / CLOCKS_PER_SEC);
	//imshow("image", dst5);
	//waitKey();
	//imshow("image", dst6);
	//waitKey();

	//imwrite("C:/Users/user/Desktop/DIP_Seminar/week1/translation2.bmp", dst5);
	//imwrite("C:/Users/user/Desktop/DIP_Seminar/week1/rotation2.bmp", dst6);
	waitKey();

	if (image.empty()) {
		cout << "Image does not exist." << endl;
		return 0;
	}


	return 0;
}
