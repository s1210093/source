#include<iostream>
#include<string.h>
#include<stdio.h>
#include<vector>
#include<fstream>
#include<time.h>
#include<windows.h>

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>

static const int src_img_rows = 600;
static const int src_img_cols = 600;

static const double R = 1;
static const double G = 1;
static const double B = 0;

static const int THRESH = 120; // しきい値

using namespace cv;
using namespace std;

void onTrackbarChanged(int thres, void*);
Point2i calculate_center(Mat);
void getCoordinates(int event, int x, int y, int flags, void* param);
Mat undist1(Mat);
Mat undist2(Mat); 
Mat undist3(Mat); 
Mat undist4(Mat);
double get_point_distance(Point2i, Point2i);
void colorExtraction(Mat* src,Mat* dst, int code, 
	int ch1Lower, int ch1Upper, 
	int ch2Lower, int ch2Upper,
	int ch3Lower,int chUpper
	);


Mat image1;
Mat synthesis,synthesis1,synthesis2,synthesis_S,synthesis_D;
Mat in_img1,in_img2,in_img3,in_img4;
Mat src_img1,src_img2,src_img3,src_img4;
ofstream fout("out");
Mat element = Mat::ones(3, 3, CV_8UC1); // 追加　3×3の行列ですべて1　dilate必要な行列
int Ax1, Ay1, Bx1, By1, Cx1, Cy1, Dx1, Dy1;
int Ax2, Ay2, Bx2, By2, Cx2, Cy2, Dx2, Dy2;
int Ax3, Ay3, Bx3, By3, Cx3, Cy3, Dx3, Dy3;
int Ax4, Ay4, Bx4, By4, Cx4, Cy4, Dx4, Dy4;
int Ax, Ay, Bx, By, Cx, Cy, Dx, Dy;
int Tr, Tg, Tb;
Point2i pre_point;


int main(int argc, char *argv[]){

	//データ取得テキストファイル定義
	string i, j;
	i = 540, j = 60;
	std::string filename1 = i+"-" +j+".txt";
	std::ofstream writing_file1;
	
	writing_file1.open(filename1, std::ios::out);
	
	//ＰＣ画像参照
	//Mat in_img = imread("./PICTURE/78.jpg");
	
	int count;
	//カメラ定義
	VideoCapture cap1(1);
	VideoCapture cap2(2); 
	VideoCapture cap3(3);
	VideoCapture cap4(4);
	//サイズ定義
	cap1.set(CV_CAP_PROP_FRAME_WIDTH, 2016);
	cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 1512);
	cap2.set(CV_CAP_PROP_FRAME_WIDTH, 2016);
	cap2.set(CV_CAP_PROP_FRAME_HEIGHT, 1512);
	cap3.set(CV_CAP_PROP_FRAME_WIDTH, 2016);
	cap3.set(CV_CAP_PROP_FRAME_HEIGHT, 1512);
	cap4.set(CV_CAP_PROP_FRAME_WIDTH, 2016);
	cap4.set(CV_CAP_PROP_FRAME_HEIGHT, 1512);

	//実験時の座標（固定）
	//Ax1 = 508, Ay1 = 571, Bx1 = 8, By1 = 294, Cx1 = 504, Cy1 = 268, Dx1 =981 , Dy1 = 361;
	//Ax2 = 995, Ay2 = 190, Bx2 = 453, By2 = 531, Cx2 = 48, Cy2 = 384, Dx2 = 498, Dy2 = 235;
	//Ax3 = 486, Ay3 = 170, Bx3 = 990, By3 = 174, Cx3 = 454, Cy3 = 464, Dx3 = 11, Dy3 = 285;
	//Ax4 = 12, Ay4 = 240, Bx4 = 490, By4 = 340, Cx4 = 937, Cy4 = 560, Dx4 = 462, Dy4 = 665;
	
	//ウインドウ生成
	namedWindow("roi1");
	namedWindow("roi2");
	namedWindow("roi3");
	namedWindow("roi4");
	namedWindow("kirihari");
	namedWindow("src1");
	namedWindow("src2");
	namedWindow("src3");
	namedWindow("src4");
	namedWindow("合成画像重心"); 
	namedWindow("距離による加重平均");
	namedWindow("dst1");
	namedWindow("dst2");
	namedWindow("dst3");
	namedWindow("dst4");
	namedWindow("合成+kirihari");
	namedWindow("glay scale");
	namedWindow("binari1");
	namedWindow("binari2");
	namedWindow("binari3");
	namedWindow("binari4");
	namedWindow("binariB");
	namedWindow("binariS");
	
	//カメラによる取得部
	for (int i = 1;i <100; i++){
		
		vector<int> v;
		clock_t loop_start = clock(); //タイマー計測(プログラムの速度計測)
		LARGE_INTEGER start_loop, end_camera1, end_camera2, end_camera3, end_camera4, end_loop, freq_pc, end_alg;
		QueryPerformanceFrequency(&freq_pc);
		QueryPerformanceCounter(&start_loop);
		//画像をリサイズ(大きすぎるとディスプレイに入りきらないため)

		cap1 >> in_img1;
		cap2 >> in_img2;
		cap3 >> in_img3;
		cap4 >> in_img4;

		cv::resize(in_img1, src_img1, cv::Size(), 0.5, 0.5);
		cv::resize(in_img2, src_img2, cv::Size(), 0.5, 0.5);
		cv::resize(in_img3, src_img3, cv::Size(), 0.5, 0.5);
		cv::resize(in_img4, src_img4, cv::Size(), 0.5, 0.5);

		src_img1 = undist1(src_img1);
		src_img2 = undist2(src_img2);
		src_img3 = undist3(src_img3);
		src_img4 = undist4(src_img4);
		
	//座標データポイント用
	/*	if (i == 1) {
			namedWindow("getCoordinates1");
			imshow("getCoordinates1", src_img1);
			cvSetMouseCallback("getCoordinates1", getCoordinates, NULL); //変換したい四角形の四隅の座標を取る(クリック)
			//帰ってきた値を代入　Ax1 =Ax......
			Ax1 = Ax; 	Ay1 = Ay;
			Bx1 = Bx;	By1 = By;
			Cx1 = Cx;	Cy1 = Cy;
			Dx1 = Dx;	Dy1 = Dy;
			waitKey(0);
			destroyAllWindows();

			namedWindow("getCoordinates2");
			imshow("getCoordinates2", src_img2);
			cvSetMouseCallback("getCoordinates2", getCoordinates, NULL);
			Ax2 = Ax; 	Ay2 = Ay;
			Bx2 = Bx;	By2 = By;
			Cx2 = Cx;	Cy2 = Cy;
			Dx2 = Dx;	Dy2 = Dy;
			waitKey(0);
			destroyAllWindows();


			namedWindow("getCoordinates3");
			imshow("getCoordinates3", src_img3);
			cvSetMouseCallback("getCoordinates3", getCoordinates, NULL);
			Ax3 = Ax; 	Ay3 = Ay;
			Bx3 = Bx;	By3 = By;
			Cx3 = Cx;	Cy3 = Cy;
			Dx3 = Dx;	Dy3 = Dy;
			waitKey(0);
			destroyAllWindows();

			namedWindow("getCoordinates4");
			imshow("getCoordinates4", src_img4);
			cvSetMouseCallback("getCoordinates4", getCoordinates, NULL);
			Ax4 = Ax; 	Ay4 = Ay;
			Bx4 = Bx;	By4 = By;
			Cx4 = Cx;	Cy4 = Cy;
			Dx4 = Dx;	Dy4 = Dy;
			waitKey(0);
			destroyAllWindows();

			waitKey(0);
			destroyAllWindows();
		}
	*/
		//変換後画像の定義
		Mat dst_img1, dst_img2, dst_img3, dst_img4, colorExtra1, colorExtra2, colorExtra3, colorExtra4, colorExtraB;

		ofstream ofs("out4.csv");
		int cam1lost = 0, cam2lost = 0, cam3lost = 0, cam4lost = 0;
		int camcount = 0;
		//-----------------透視変換-----------------------------------------------
		//1
		Point2f pts1s[] = { Point2f(Ax1, Ay1), Point2f(Bx1, By1), Point2f(Cx1, Cy1), Point2f(Dx1, Dy1) };


		Point2f pts1d[] = { Point2f(0, src_img_rows), Point2f(0, 0), Point2f(src_img_cols, 0), Point2f(src_img_cols, src_img_rows) };

		Mat perspective_matrix1 = getPerspectiveTransform(pts1s, pts1d);

		warpPerspective(src_img1, dst_img1, perspective_matrix1, Size(600, 600), INTER_LINEAR);

		colorExtraction(&dst_img1, &colorExtra1, CV_BGR2HSV, 150, 180, 120, 255, 70, 255);

		cvtColor(colorExtra1, colorExtra1, CV_BGR2GRAY);
		//--------------binari1
		Mat binari_1;
		threshold(colorExtra1, binari_1, 0, 255, THRESH_BINARY);
		dilate(binari_1, binari_1, element, Point(-1, -1), 3);
		//--------------centroid1
		Point2i centroid1 = calculate_center(binari_1);
		writing_file1 << centroid1.x << " " << centroid1.y << endl;
		if (centroid1.x != 0){
			int ypos = src_img_rows - (centroid1.y + 6 * ((1000 / centroid1.y) + 1));
			//writing_file1 << centroid1.x << " " << ypos << endl;
			ofs << centroid1.x << " " << ypos << endl;
		}
		if (centroid1.x == 0 && centroid1.y == 0){
			cam1lost = 1;
		}
		else camcount++;
		//clock_t centroid_1_end = clock(); // カメラ1の計算の時間
		QueryPerformanceCounter(&end_camera1);
		//writing_file << "time1 :" << (end_camera1.QuadPart - start_loop.QuadPart) / (double)freq_pc.QuadPart << "sec.\n" << endl;
		//writing_file << "time1 :" << (double)(centroid_1_end -loop_start)/CLOCKS_PER_SEC << "sec.\n";
		/*	if (!centroid1.y == 0){
			circle(dst_img1, Point(centroid1.x, centroid1.y + 6 * ((1000 / centroid1.y) + 1)),5,Scalar(200,0,0),-1,CV_AA);
			}*/

		//2
		Point2f pts2s[] = { Point2f(Ax2, Ay2), Point2f(Bx2, By2), Point2f(Cx2, Cy2), Point2f(Dx2, Dy2) };

		Point2f pts2d[] = { Point2f(0, src_img_rows), Point2f(0, 0), Point2f(src_img_cols, 0), Point2f(src_img_cols, src_img_rows) };


		Mat perspective_matrix2 = getPerspectiveTransform(pts2s, pts2d);

		warpPerspective(src_img2, dst_img2, perspective_matrix2, Size(600, 600), INTER_LINEAR);
		colorExtraction(&dst_img2, &colorExtra2, CV_BGR2HSV, 150, 180, 120, 255, 70, 255);
		cvtColor(colorExtra2, colorExtra2, CV_BGR2GRAY);
		//--------------binari2
		Mat binari_2;
		threshold(colorExtra2, binari_2, 0, 255, THRESH_BINARY);
		dilate(binari_2, binari_2, element, Point(-1, -1), 3);
		//--------------centroid2
		Point2i centroid2 = calculate_center(binari_2);
		//	clock_t centroid_2_end = clock(); // カメラ2の計算の時間
		QueryPerformanceCounter(&end_camera2);
		if (centroid2.x == 0 && centroid2.y == 0){
			cam2lost = 1;
		}
		else camcount++;
		//writing_file << "time2 :" << (end_camera2.QuadPart - start_loop.QuadPart) / (double)freq_pc.QuadPart << "sec.\n" << endl;
		writing_file1 << centroid2.x << " " << centroid2.y << endl;
		//	writing_file << "time2 :" << (double)(centroid_2_end - loop_start)/CLOCKS_PER_SEC <<"sec.\n";

		if (centroid2.x != 0){
			int ypos = src_img_rows - (centroid2.y + 6 * ((1000 / centroid2.y) + 1));
			//writing_file << centroid2.x << " " << ypos << endl;
			ofs << centroid2.x << " " << ypos << endl;
		}


		/*if (!centroid2.y == 0){
			circle(dst_img2, Point(centroid2.x, centroid2.y + 6 * ((1000 / centroid2.y) + 1)), 5, Scalar(200, 0, 0), -1, CV_AA);
			}*/
		//3
		Point2f pts3s[] = { Point2f(Ax3, Ay3), Point2f(Bx3, By3), Point2f(Cx3, Cy3), Point2f(Dx3, Dy3) };

		Point2f pts3d[] = { Point2f(0, src_img_rows), Point2f(0, 0), Point2f(src_img_cols, 0), Point2f(src_img_cols, src_img_rows) };


		Mat perspective_matrix3 = getPerspectiveTransform(pts3s, pts3d);

		warpPerspective(src_img3, dst_img3, perspective_matrix3, Size(600, 600), INTER_LINEAR);
		colorExtraction(&dst_img3, &colorExtra3, CV_BGR2HSV, 150, 180, 120, 255, 70, 255);
		cvtColor(colorExtra3, colorExtra3, CV_BGR2GRAY);
		//--------------binari3
		Mat binari_3;
		threshold(colorExtra3, binari_3, 0, 255, THRESH_BINARY);
		dilate(binari_3, binari_3, element, Point(-1, -1), 3);
		//--------------centroid3
		Point2i centroid3 = calculate_center(binari_3);
		writing_file1 << centroid3.x << " " << centroid3.y << endl;
		//clock_t centroid_3_end = clock(); // カメラ3の計算の時間
		QueryPerformanceCounter(&end_camera3);
		if (centroid3.x == 0 && centroid3.y == 0){
			cam3lost = 1;
		}
		else camcount++;
	//	writing_file << "time3 :" << (end_camera3.QuadPart - start_loop.QuadPart) / (double)freq_pc.QuadPart << "sec.\n" << endl;
		//writing_file << "time3 :" << (double)(centroid_3_end-loop_start)/CLOCKS_PER_SEC << "sec.\n";
		if (centroid3.x != 0){
			int ypos = src_img_rows - (centroid3.y + 6 * ((1000 / centroid3.y) + 1));
		//	writing_file << centroid3.x << " " << ypos << endl;
			ofs << centroid3.x << " " << ypos << endl;
		}
		/*if (!centroid3.y == 0){
			circle(dst_img3, Point(centroid3.x, centroid3.y + 6 * ((1000 / centroid3.y) + 1)), 5, Scalar(200, 0, 0), -1, CV_AA);
			}*/
		//4
		Point2f pts4s[] = { Point2f(Ax4, Ay4), Point2f(Bx4, By4), Point2f(Cx4, Cy4), Point2f(Dx4, Dy4) };

		Point2f pts4d[] = { Point2f(0, src_img_rows), Point2f(0, 0), Point2f(src_img_cols, 0), Point2f(src_img_cols, src_img_rows) };


		Mat perspective_matrix4 = getPerspectiveTransform(pts4s, pts4d);

		warpPerspective(src_img4, dst_img4, perspective_matrix4, Size(600, 600), INTER_LINEAR);
		colorExtraction(&dst_img4, &colorExtra4, CV_BGR2HSV, 150, 180, 120, 255, 70, 255);
		cvtColor(colorExtra4, colorExtra4, CV_BGR2GRAY);
		//--------------binari4
		Mat binari_4;
		threshold(colorExtra4, binari_4, 0, 255, THRESH_BINARY);
		dilate(binari_4, binari_4, element, Point(-1, -1), 3);
		//--------------centroid4
		Point2i centroid4 = calculate_center(binari_4);
		writing_file1 << centroid4.x << " " << centroid4.y << endl;
		if (centroid4.x == 0 && centroid4.y == 0){
			cam4lost = 1;
		}
		else camcount++;
		//clock_t centroid_4_end = clock(); // カメラ4の計算の時間
		QueryPerformanceCounter(&end_camera4);
		//writing_file << "time4 :" << (end_camera4.QuadPart - start_loop.QuadPart) / (double)freq_pc.QuadPart << "sec.\n" << endl;
		//writing_file << "time4 :" << (double)(centroid_4_end -loop_start)/CLOCKS_PER_SEC<< "sec.\n";
		if (centroid4.x != 0){
			int ypos = src_img_rows - (centroid4.y + 6 * ((1000 / centroid4.y) + 1));
			//writing_file << centroid4.x << " " << ypos << endl;
			ofs << centroid4.x << " " << ypos << endl;
		}
		/*if (!centroid4.y == 0){
			circle(dst_img4, Point(centroid4.x, centroid4.y + 6 * ((1000 / centroid4.y) + 1)), 5, Scalar(200, 0, 0), -1, CV_AA);
			}*/


		//変換前後の座標を描画

		line(src_img1, pts1s[0], pts1s[1], Scalar(225, 0, 225), 2, CV_AA);
		line(src_img1, pts1s[1], pts1s[2], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img1, pts1s[2], pts1s[3], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img1, pts1s[3], pts1s[0], Scalar(255, 255, 0), 2, CV_AA);
		/*line(src_img1, pts1d[0], pts1d[1], Scalar(255, 0, 255), 2, CV_AA);
		line(src_img1, pts1d[1], pts1d[2], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img1, pts1d[2], pts1d[3], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img1, pts1d[3], pts1d[0], Scalar(255, 255, 0), 2, CV_AA);*/

		line(src_img2, pts2s[0], pts2s[1], Scalar(225, 0, 225), 2, CV_AA);
		line(src_img2, pts2s[1], pts2s[2], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img2, pts2s[2], pts2s[3], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img2, pts2s[3], pts2s[0], Scalar(255, 255, 0), 2, CV_AA);
		/*line(src_img2, pts2d[0], pts2d[1], Scalar(255, 0, 255), 2, CV_AA);
		line(src_img2, pts2d[1], pts2d[2], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img2, pts2d[2], pts2d[3], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img2, pts2d[3], pts2d[0], Scalar(255, 255, 0), 2, CV_AA);
		*/
		line(src_img3, pts3s[0], pts3s[1], Scalar(225, 0, 225), 2, CV_AA);
		line(src_img3, pts3s[1], pts3s[2], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img3, pts3s[2], pts3s[3], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img3, pts3s[3], pts3s[0], Scalar(255, 255, 0), 2, CV_AA);
		/*line(src_img3, pts3d[0], pts3d[1], Scalar(255, 0, 255), 2, CV_AA);
		line(src_img3, pts3d[1], pts3d[2], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img3, pts3d[2], pts3d[3], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img3, pts3d[3], pts3d[0], Scalar(255, 255, 0), 2, CV_AA);*/

		line(src_img4, pts4s[0], pts4s[1], Scalar(225, 0, 225), 2, CV_AA);
		line(src_img4, pts4s[1], pts4s[2], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img4, pts4s[2], pts4s[3], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img4, pts4s[3], pts4s[0], Scalar(255, 255, 0), 2, CV_AA);
/*		line(src_img4, pts4d[0], pts4d[1], Scalar(255, 0, 255), 2, CV_AA);
		line(src_img4, pts4d[1], pts4d[2], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img4, pts4d[2], pts4d[3], Scalar(255, 255, 0), 2, CV_AA);
		line(src_img4, pts4d[3], pts4d[0], Scalar(255, 255, 0), 2, CV_AA);*/
		//重心重み付け等
		int centroid_sum_x, centroid_sum_y, centroid_sum_x_d, centroid_sum_y_d;
		double distance_1, distance_2, distance_3, distance_4, distance_sum;
		double w1, w2, w3, w4;
		centroid_sum_x = (centroid1.x + centroid2.x + centroid3.x + centroid4.x) / 4;
		centroid_sum_y = (centroid1.y + centroid2.y + centroid3.y + centroid4.y) / 4;
		//端からロボットの位置までの距離計算（重み付けのため）
		distance_4 = (centroid4.x)*(centroid4.x) + (centroid4.y)*(centroid4.y);
		distance_1 = (src_img_rows - centroid1.x)*(src_img_rows - centroid1.x) + (centroid1.y)*(centroid1.y);
		distance_2 = (src_img_rows - centroid2.x)*(src_img_rows - centroid2.x) + (src_img_cols - centroid2.y)*(src_img_cols - centroid2.y);
		distance_3 = (centroid3.x)*(centroid3.x) + (src_img_cols - centroid3.y)*(src_img_cols - centroid3.y);
		if (camcount==4){
			distance_sum = distance_1 + distance_2 + distance_3 + distance_4;
			w1 = distance_1 / distance_sum;
			w2 = distance_2 / distance_sum;
			w3 = distance_3 / distance_sum;
			w4 = distance_4 / distance_sum;
			centroid_sum_x_d = (centroid1.x * w1 + centroid2.x * w2 + centroid3.x * w3 + centroid4.x * w4);
			centroid_sum_y_d = (centroid1.y * w1 + centroid2.y * w2 + centroid3.y * w3 + centroid4.y * w4);
		}else if (camcount==3&&cam1lost==1){
			distance_sum = distance_2 + distance_3 + distance_4;
			w2 = distance_2 / distance_sum;
			w3 = distance_3 / distance_sum;
			w4 = distance_4 / distance_sum;
			centroid_sum_x_d = (centroid2.x * w2 + centroid3.x * w3 + centroid4.x * w4);
			centroid_sum_y_d = (centroid2.y * w2 + centroid3.y * w3 + centroid4.y * w4);
		}else if (camcount == 3 && cam2lost == 1){
			distance_sum = distance_1 + distance_3 + distance_4;
			w1 = distance_1 / distance_sum;
			w3 = distance_3 / distance_sum;
			w4 = distance_4 / distance_sum;
			centroid_sum_x_d = (centroid1.x * w1 + centroid3.x * w3 + centroid4.x * w4);
			centroid_sum_y_d = (centroid1.y * w1 + centroid3.y * w3 + centroid4.y * w4);
		}
		else if (camcount == 3 && cam3lost == 1){
			distance_sum = distance_2 + distance_3 + distance_4;
			w2 = distance_2 / distance_sum;
			w1 = distance_1 / distance_sum;
			w4 = distance_4 / distance_sum;
			centroid_sum_x_d = (centroid1.x * w1 + centroid2.x * w2 + centroid4.x * w4);
			centroid_sum_y_d = (centroid1.y * w1 + centroid2.y * w2 + centroid4.y * w4);
		}
		else if (camcount == 3 && cam4lost == 1){
			distance_sum = distance_2 + distance_3 + distance_1;
			w2 = distance_2 / distance_sum;
			w3 = distance_3 / distance_sum;
			w1 = distance_1 / distance_sum;
			centroid_sum_x_d = (centroid1.x * w1 + centroid2.x * w2 + centroid3.x * w3 );
			centroid_sum_y_d = (centroid1.y * w1 + centroid2.y * w2 + centroid3.y * w3 );
		}
		else if (camcount == 2 && cam1lost == 1 && cam2lost == 1){
			distance_sum = distance_3 + distance_4;
			w3 = distance_3 / distance_sum;
			w4 = distance_4 / distance_sum;
			centroid_sum_x_d = (centroid4.x * w4 + centroid3.x * w3);
			centroid_sum_y_d = (centroid4.y * w4 + centroid3.y * w3);
		}
		else if (camcount == 2 && cam1lost == 1 && cam3lost == 1){
			distance_sum = distance_2 + distance_4;
			w2 = distance_2 / distance_sum;
			w4 = distance_4 / distance_sum;
			centroid_sum_x_d = (centroid4.x * w4 + centroid2.x * w2);
			centroid_sum_y_d = (centroid4.y * w4 + centroid2.y * w2);
		}
		else if (camcount == 2 && cam1lost == 1 && cam4lost == 1){
			distance_sum = distance_2 + distance_3;
			w3 = distance_3 / distance_sum;
			w2 = distance_2 / distance_sum;
			centroid_sum_x_d = (centroid2.x * w2 + centroid3.x * w3);
			centroid_sum_y_d = (centroid2.y * w2 + centroid3.y * w3);
		}
		else if (camcount == 2 && cam2lost == 1 && cam3lost == 1){
			distance_sum = distance_1 + distance_4;
			w1 = distance_1 / distance_sum;
			w4 = distance_4 / distance_sum;
			centroid_sum_x_d = (centroid4.x * w4 + centroid1.x * w1);
			centroid_sum_y_d = (centroid4.y * w4 + centroid1.y * w1);
		}
		else if (camcount == 2 && cam2lost == 1 && cam4lost == 1){
			distance_sum = distance_1 + distance_3;
			w1 = distance_1 / distance_sum;
			w3 = distance_3 / distance_sum;
			centroid_sum_x_d = (centroid3.x * w3 + centroid1.x * w1);
			centroid_sum_y_d = (centroid3.y * w3 + centroid1.y * w1);
		}
		else if (camcount == 2 && cam3lost == 1 && cam4lost == 1){
			distance_sum = distance_1 + distance_2;
			w1 = distance_1 / distance_sum;
			w2 = distance_2 / distance_sum;
			centroid_sum_x_d = (centroid2.x * w2 + centroid1.x * w1);
			centroid_sum_y_d = (centroid2.y * w2 + centroid1.y * w1);
		}
		else if (camcount == 1 && cam1lost == 0){
			distance_sum = distance_1;
			w1 = distance_1 / distance_sum;
			centroid_sum_x_d = (centroid1.x * w1);
			centroid_sum_y_d = (centroid1.y * w1);
		}
		else if (camcount == 1 && cam2lost == 0){
			distance_sum = distance_2;
			w2 = distance_2 / distance_sum;
			centroid_sum_x_d = (centroid2.x * w2);
			centroid_sum_y_d = (centroid2.y * w2);
		}
		else if (camcount == 1 && cam3lost == 0){
			distance_sum = distance_3;
			w3 = distance_3 / distance_sum;
			centroid_sum_x_d = (centroid3.x * w3);
			centroid_sum_y_d = (centroid3.y * w3);
		}
		else if (camcount == 1 && cam4lost == 0){
			distance_sum = distance_4;
			w4 = distance_4 / distance_sum;
			centroid_sum_x_d = (centroid4.x * w4);
			centroid_sum_y_d = (centroid4.y * w4);
		}
		

		//カメラの配置を変の中央版
		/*distance_4 = ((src_img_rows/2) - centroid4.x)*((src_img_rows /2)-centroid4.x) + (centroid4.y)*(centroid4.y);
		distance_1 = (src_img_rows - centroid1.x)*(src_img_rows - centroid1.x) + (centroid1.y- (src_img_cols/2))*(centroid1.y - (src_img_cols/2));
		distance_2 = ((src_img_rows/2) - centroid2.x)*((src_img_rows/2) - centroid2.x) + (src_img_cols - centroid2.y)*(src_img_cols - centroid2.y);
		distance_3 = (centroid3.x)*(centroid3.x) + ((src_img_cols/2) - centroid3.y)*((src_img_cols/2) - centroid3.y);
		distance_sum = distance_1 + distance_2 + distance_3 + distance_4;
		w1 = distance_1 / distance_sum;
		w2 = distance_2 / distance_sum;
		w3 = distance_3 / distance_sum;
		w4 = distance_4 / distance_sum;
		centroid_sum_x_d = (centroid1.x * w1 + centroid2.x * w2 + centroid3.x * w3 + centroid4.x * w4);
		centroid_sum_y_d = (centroid1.y * w1 + centroid2.y * w2 + centroid3.y * w3 + centroid4.y * w4);
		*/
		QueryPerformanceCounter(&end_alg);
		cout << "count :" << i << endl;

		// 画像、円の中心座標、半径、色、線太さ、種類(-1, CV_AAは塗りつぶし)
		//	circle(dst_img1, Point(centroid1.x, centroid1.y), 5, Scalar(0, 255, 0), -1, CV_AA);
		//	circle(dst_img2, Point(centroid2.x, centroid2.y), 5, Scalar(0, 255, 0), -1, CV_AA);
		//	circle(dst_img3, Point(centroid3.x, centroid3.y), 5, Scalar(0, 255, 0), -1, CV_AA);
		//	circle(dst_img4, Point(centroid4.x, centroid4.y), 5, Scalar(0, 255, 0), -1, CV_AA);
		//-----------------表示部分---------------------------------
		Mat base(src_img_rows, src_img_cols, CV_8UC3);
	
		Mat dst_roi1(dst_img2, Rect(0, 0, src_img_cols / 2, src_img_rows / 2));
		Mat roi1(base, Rect(0, 0, src_img_cols/2, src_img_rows/2));
		dst_roi1.copyTo(roi1);
		imshow("roi1", roi1);

		Mat dst_roi2(dst_img3, Rect(src_img_cols / 2, 0, src_img_cols / 2, src_img_rows / 2));
		Mat roi2(base, Rect(src_img_cols/2, 0, src_img_cols/2, src_img_rows/2));
		dst_roi2.copyTo(roi2);
		imshow("roi2", roi2);

		Mat dst_roi3(dst_img1, Rect(0, src_img_rows/2, src_img_cols / 2, src_img_rows / 2));
		Mat roi3(base, Rect(0, src_img_rows/2, src_img_cols/2, src_img_rows / 2));
		dst_roi3.copyTo(roi3);
		imshow("roi3", roi3);
		
		Mat dst_roi4(dst_img4, Rect(src_img_cols / 2, src_img_rows / 2, src_img_cols / 2, src_img_rows / 2));
		Mat roi4(base, Rect(src_img_cols/2, src_img_rows/2, src_img_cols/2, src_img_rows / 2));
		dst_roi4.copyTo(roi4);
		imshow("roi4", roi4);

		colorExtraction(&base, &colorExtraB, CV_BGR2HSV, 150, 180, 120, 255, 70, 255);
		cvtColor(colorExtraB, colorExtraB, CV_BGR2GRAY);
		//--------------binaribase
		Mat binari_B;
		threshold(colorExtraB, binari_B, 0, 255, THRESH_BINARY);
		dilate(binari_B, binari_B, element, Point(-1, -1), 3);
		//--------------centroidbase
		Point2i centroidB = calculate_center(binari_B);
		writing_file1 << centroidB.x << " " << centroidB.y << endl;
		circle(base, Point(centroidB.x, centroidB.y), 5, Scalar(255, 255, 255), -1, CV_AA);
		imshow("kirihari",base);

		addWeighted(dst_img1, 0.5, dst_img2, 0.5, 0, synthesis1);
		addWeighted(dst_img3, 0.5, dst_img4, 0.5, 0, synthesis2);
		addWeighted(synthesis1, 0.5, synthesis2, 0.5, 0, synthesis);
		
		Mat colorExtraS;
		colorExtraction(&synthesis, &colorExtraS, CV_BGR2HSV, 150, 180, 70, 255, 70, 255);
		cvtColor(colorExtraS, colorExtraS, CV_BGR2GRAY);
		//--------------binariS
		Mat binari_S;
		threshold(colorExtraS, binari_S, 0, 255, THRESH_BINARY);
		dilate(binari_S, binari_S, element, Point(-1, -1), 3);
		Point2i centroid_S = calculate_center(binari_S);
		synthesis_S = synthesis.clone();
		synthesis_D = synthesis.clone();
		circle(synthesis_S, Point(centroid_S.x, centroid_S.y), 5, Scalar(255, 255, 0), -1, CV_AA);
		circle(synthesis_D, Point(centroid_sum_x_d, centroid_sum_y_d), 5, Scalar(0, 255, 255), -1, CV_AA);
		writing_file1 << centroid_S.x << " " << centroid_S.y << endl;
		//writing_file<< "position_s2 :" << centroid_sum_x << " " << centroid_sum_y << endl;
		writing_file1 << centroid_sum_x_d << " " << centroid_sum_y_d << endl;
		Mat synthesis_kirihari;
		addWeighted(synthesis, 0.5,base,0.5,0,synthesis_kirihari );
		imshow("src1", src_img1);
		imshow("src2", src_img2);
		imshow("src3", src_img3);
		imshow("src4", src_img4);
		imshow("合成画像重心", synthesis_S); 
		imshow("距離による加重平均", synthesis_D);
	//	circle(dst_img1, Point(centroid1.x, centroid1.y), 5, Scalar(0, 255, 0), -1, CV_AA);
	//	circle(dst_img2, Point(centroid2.x, centroid2.y), 5, Scalar(0, 255, 0), -1, CV_AA);
	//	circle(dst_img3, Point(centroid3.x, centroid3.y), 5, Scalar(0, 255, 0), -1, CV_AA);
	//	circle(dst_img4, Point(centroid4.x, centroid4.y), 5, Scalar(0, 255, 0), -1, CV_AA);

		imshow("dst1", dst_img1);
		imshow("dst2", dst_img2);
		imshow("dst3", dst_img3);
		imshow("dst4", dst_img4);
		imshow("合成+kirihari", synthesis_kirihari);
		imshow("glay scale", colorExtra1);
		imshow("binari1", binari_1);
		imshow("binari2", binari_2);
		imshow("binari3", binari_3);
		imshow("binari4", binari_4);
		imshow("binariB", binari_B);
		imshow("binariS", binari_S);
		
		//clock_t end = clock();
		//writing_file << "one_loop_time :" << (double)(end - loop_start)/ CLOCKS_PER_SEC <<"sec.\n"<< endl;
		
		QueryPerformanceCounter(&end_loop);
		//writing_file << "count" << i <<endl;
		int key = waitKey(1);
		if (key == 113)break;
		/*	namedWindow("gray");
			imshow("gray", image1);

			namedWindow("binari_2");
			imshow("binari_2", binari_2);*/
	}
			waitKey(0);
			destroyAllWindows();
		
		//重心位置確認用
		//namedWindow("dst") ;
		//imshow("dst", dst_img) ;
		//waitKey(0);
		//destroyAllwindows();
	
	fout.close();
	//fout2.close() ;
	getchar();
	exit(1);
}

double get_points_distance(Point2i point, Point2i pre_point){

	return sqrt((point.x - pre_point.x)*(point.x - pre_point.x)
		+ (point.y - pre_point.y) * (point.y - pre_point.y));
}
void onTrackbarChanged(int thres, void*)
{

	Mat image2;
	threshold(image1, image2, thres, 255, THRESH_BINARY);

	imshow("binari", image2);
}

Point2i calculate_center(Mat gray)
{
	Point2i center = Point2i(0, 0);
	Moments moment = moments(gray, true);

	if (moment.m00 != 0)
	{
		center.x = (int)(moment.m10 / moment.m00);
		center.y = (int)(moment.m01 / moment.m00);
	}

	return center;
}

void getCoordinates(int event, int x, int y, int flags, void* param)
{
	static int count = 0;
	switch (event){
	case CV_EVENT_LBUTTONDOWN:

		if (count == 0){
			Ax1= x, Ay1 = y;
			cout << "Ax1 :" << x << ", Ay1 : " << y << endl;
		}
		else if (count == 1){
			Bx1 = x, By1 = y;
			cout << "Bx1 :" << x << ", By1 :" << y << endl;
		}
		else if (count == 2){
			Cx1 = x, Cy1 = y;
			cout << "Cx1 :" << x << ", Cy1 :" << y << endl;
		}
		else if (count == 3){
			Dx1 = x, Dy1 = y;
			cout << "Dx1 :" << x << ", Dy1 :" << y << endl;
		}
		else if (count == 4){
			Ax2 = x, Ay2 = y;
			cout << "Ax2 :" << x << ", Ay2 : " << y << endl;
		}
		else if (count == 5){
			Bx2 = x, By2 = y;
			cout << "Bx2 :" << x << ", By2 :" << y << endl;
		}
		else if (count == 6){
			Cx2 = x, Cy2 = y;
			cout << "Cx2 :" << x << ", Cy2 :" << y << endl;
		}
		else if (count == 7){
			Dx2 = x, Dy2 = y;
			cout << "Dx2 :" << x << ", Dy2 :" << y << endl;
		}
		else if (count == 8){
			Ax3 = x, Ay3 = y;
			cout << "Ax3 :" << x << ", Ay3 : " << y << endl;
		}
		else if (count == 9){
			Bx3 = x, By3 = y;
			cout << "Bx3 :" << x << ", By3:" << y << endl;
		}
		else if (count == 10){
			Cx3 = x, Cy3 = y;
			cout << "Cx3 :" << x << ", Cy3:" << y << endl;
		}
		else if (count == 11){
			Dx3 = x, Dy3 = y;
			cout << "Dx3 :" << x << ", Dy3" << y << endl;
		}
		else if (count == 12){
			Ax4 = x, Ay4 = y;
			cout << "Ax4 :" << x << ", Ay4: " << y << endl;
		}
		else if (count == 13){
			Bx4 = x, By4 = y;
			cout << "Bx4 :" << x << ", By4:" << y << endl;
		}
		else if (count == 14){
			Cx4 = x, Cy4 = y;
			cout << "Cx4 :" << x << ", Cy4:" << y << endl;
		}
		else if (count == 15){
			Dx4 = x, Dy4 = y;
			cout << "Dx4 :" << x << ", Dy4" << y << endl;
		}
		else{
			break;
		}
		count++;
		break;

	default:
		break;
	}
}


Mat undist1(Mat src_img)
{
	Mat dst_img;

	//カメラマトリックス
	Mat cameraMatrix = (Mat_<double>(3, 3) << 7.5858056570897111e+02, 0, 5.1866705643539865e+02, 0, 7.5797757138944689e+02, 3.4132731608800276e+02, 0, 0, 1);
	//歪み行列
	Mat distcoeffs = (Mat_<double>(1, 5) << 8.9403017238394653e-02, -1.3137759201202207e-01, 2.1963127176430172e-03, -7.4668286503898774e-04, -3.5364253310066922e-02);

	undistort(src_img, dst_img, cameraMatrix, distcoeffs);

	return dst_img;
}

Mat undist2(Mat src_img)
{
	Mat dst_img;

	//カメラマトリックス
	Mat cameraMatrix = (Mat_<double>(3, 3) << 1.5025815571285352e+03 ,0 ,9.1485767778503396e+02, 0, 1.5031505471476949e+03, 8.1124951197355801e+02, 0, 0, 1);
	//歪み行列
	Mat distcoeffs = (Mat_<double>(1, 5) << 6.0631623384908698e-02 ,- 1.0752408016529373e-01 ,3.3679487496303686e-03, - 3.1406031253158667e-03 ,- 2.8522445629677842e-02);

	undistort(src_img, dst_img, cameraMatrix, distcoeffs);

	return dst_img;
}

Mat undist3(Mat src_img)
{
	Mat dst_img;

	//カメラマトリックス
	Mat cameraMatrix = (Mat_<double>(3, 3) << 7.7140328294208598e+02, 0, 4.8583699835138520e+02, 0, 7.7098809246564019e+02, 3.7794046314033523e+02, 0, 0, 1);
	//歪み行列
	Mat distcoeffs = (Mat_<double>(1, 5) << 8.3131169184355577e-02, -5.5660571751010197e-02, 1.6352528561686152e-04, -1.9751465776663090e-03, -1.3585275314413905e-01);

	undistort(src_img, dst_img, cameraMatrix, distcoeffs);

	return dst_img;
}
Mat undist4(Mat src_img)
{
	Mat dst_img;

	//カメラマトリックス
	Mat cameraMatrix = (Mat_<double>(3, 3) << 7.7089662359448016e+02, 0, 4.8177269308105713e+02, 0, 7.7087218947063400e+02, 3.8580478709997357e+02, 0, 0, 1);
	//歪み行列
	Mat distcoeffs = (Mat_<double>(1, 5) << 8.6187003547312138e-02, -8.3941796833625681e-02, 6.4471991170733699e-03, 2.0845536793389482e-03, -1.0390462509808196e-01);

	undistort(src_img, dst_img, cameraMatrix, distcoeffs);

	return dst_img;
}


void colorExtraction(cv::Mat* src, cv::Mat* dst,
	int code,
	int ch1Lower, int ch1Upper, //@comment H(色相)　最小、最大
	int ch2Lower, int ch2Upper, //@comment S(彩度)　最小、最大
	int ch3Lower, int ch3Upper  //@comment V(明度)　最小、最大
	)
{
	cv::Mat colorImage;
	int lower[3];
	int upper[3];

	cv::Mat lut = cv::Mat(256, 1, CV_8UC3);

	cv::cvtColor(*src, colorImage, code);

	lower[0] = ch1Lower;
	lower[1] = ch2Lower;
	lower[2] = ch3Lower;

	upper[0] = ch1Upper;
	upper[1] = ch2Upper;
	upper[2] = ch3Upper;

	for (int i = 0; i < 256; i++){
		for (int k = 0; k < 3; k++){
			if (lower[k] <= upper[k]){
				if ((lower[k] <= i) && (i <= upper[k])){
					lut.data[i*lut.step + k] = 255;
				}
				else{
					lut.data[i*lut.step + k] = 0;
				}
			}
			else{
				if ((i <= upper[k]) || (lower[k] <= i)){
					lut.data[i*lut.step + k] = 255;
				}
				else{
					lut.data[i*lut.step + k] = 0;
				}
			}
		}
	}
	//@comment LUTを使用して二値化
	cv::LUT(colorImage, lut, colorImage);

	//@comment Channel毎に分解
	std::vector<cv::Mat> planes;
	cv::split(colorImage, planes);

	//@comment マスクを作成
	cv::Mat maskImage;
	cv::bitwise_and(planes[0], planes[1], maskImage);
	cv::bitwise_and(maskImage, planes[2], maskImage);

	//@comemnt 出力
	cv::Mat maskedImage;
	src->copyTo(maskedImage, maskImage);
	*dst = maskedImage;

}