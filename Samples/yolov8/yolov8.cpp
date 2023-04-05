#include <SoyNet.h>
#include <yolo.h>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <random>
#include <stdlib.h>
#include <stdio.h>
using namespace std;
using namespace cv;
using namespace chrono;

static char soynet_home[] = "..";

static vector<string> coco_label = { //"BG",
	"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};
static vector<uchar> colors = {
	0,252,255,0,118,255,0,70,255,0,67,255,0,10,255,0,6,255,0,0,255,0,204,255,0,185,255,0,143,255,0,134,255,0,214,255,0,45,255,0,131,255,0,220,255,0,207,255,0,112,255,0,77,255,0,194,255,0,175,255,0,38,255,0,64,255,0,3,255,0,48,255,0,54,255,0,163,255,0,41,255,0,153,255,0,172,255,0,102,255,0,239,255,0,223,255,0,51,255,0,99,255,0,182,255,0,61,255,0,89,255,0,198,255,0,191,255,0,92,255,0,179,255,0,140,255,0,86,255,0,188,255,0,150,255,0,105,255,0,159,255,0,83,255,0,226,255,0,115,255,0,166,255,0,108,255,0,242,255,0,35,255,0,96,255,0,13,255,0,16,255,0,156,255,0,32,255,0,26,255,0,201,255,0,73,255,0,230,255,0,128,255,0,245,255,0,137,255,0,147,255,0,29,255,0,19,255,0,121,255,0,236,255,0,249,255,0,169,255,0,233,255,0,124,255,0,217,255,0,22,255,0,57,255,0,210,255,0,80,255,
};



void yolov8_img(int org_height, int org_width, vector<string>& params, char model_name[])
{
	int region_count = 2000;
	int count_per_class = 10; // 한장의 이미지안에  class 당 최대 몇개인지..., 사람이 많아도 이놈이 한계! 
	int batch_size = params.size();
	int make_engine = 1;
	int class_count = coco_label.size();
	int result_count = batch_size * count_per_class * class_count; // 모든 batch를 통털어 전체 출력 갯수
	float conf_thres = 0.25f;
	float iou_thres = 0.7f;
	char cfg_file[256];	sprintf(cfg_file, "../mgmt/configs/%s.cfg", model_name);
	char engine_file[256];	sprintf(engine_file, "../mgmt/engines/%s.bin", model_name);
	char weight_file[256];	sprintf(weight_file, "../mgmt/weights/%s.weights", model_name);
	char log_dir[] = "../mgmt/logs";
	char extend_param[2000];
	//char img_size[] = "640,640";
	int new_height = 640, new_width = 640;
	int stride = 32;

	YOLO_SIZE s;
	int is_resize = calc_resize_yolo( &s, new_height, new_width, org_height, org_width, stride	);

	//vector<uchar> colors(class_count * 3); // 마스크로 사용할 색상 table 저장소
	//makeColors(class_count, colors.data(), "bgr"); // mask로 사용할 색상 Table을 미리 만들어 놓는다.
	//for (int idx = 0; idx < colors.size(); idx++) {
	//	printf("%d,", colors[idx]);
	//}
	
	sprintf(extend_param,
		"BATCH_SIZE=%d SOYNET_HOME=%s MODEL_NAME=%s MAKE_ENGINE=%d ENGINE_FILE=%s CLASS_COUNT=%d CONF_THRES=%f IOU_THRES=%f REGION_COUNT=%d COUNT_PER_CLASS=%d RESULT_COUNT=%d WEIGHT_FILE=%s RE_SIZE=%d,%d ORG_SIZE=%d,%d TOP=%d BOTTOM=%d LEFT=%d RIGHT=%d",
		batch_size, soynet_home, model_name, make_engine, engine_file, class_count, conf_thres, iou_thres, region_count, count_per_class, result_count, weight_file, s.unpad_height, s.unpad_width, org_height, org_width, s.pad_top, s.pad_bottom, s.pad_left, s.pad_right);

	void* handle = initSoyNet(cfg_file, extend_param);
	int re_map_size = s.unpad_height * s.unpad_width * 3;

	vector<uchar> input(batch_size * re_map_size);

	
	int ITER = 100;
	int total_usec = 0;
	for (int iter = 0; iter < ITER; iter++) {
		long long start_usec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
		for (int idx = 0; idx < batch_size; idx++) {
			Mat img = cv::imread(params[idx]);
			if (is_resize) {
				resize(img, img, Size(s.unpad_width, s.unpad_height));
			}
			memcpy(input.data() + idx * re_map_size, img.data, re_map_size);
		}
		feedData(handle, 0, input.data());
		inference(handle);
		vector<YOLO_RESULT> output(result_count);
		getOutput(handle, 0, output.data());
		long long end_usec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
		int frame_usec = int(end_usec - start_usec);
		total_usec += frame_usec;
		float fps = 1000000.f / frame_usec;
		printf("%03d %.2f fps\n", iter, fps);
		if (1) {
			// 예시로 batch_idx == 0인 경우를 보여줌
			YOLO_RESULT* res = (YOLO_RESULT*)output.data();
			Mat img = cv::imread(params[0]);

			for (int idx = 0; idx < result_count && res[idx].conf>0.f; idx++) {
				YOLO_RESULT& r = res[idx];
				if (r.batch_idx != 0) continue; // 예시로 batch_idx == 0인 경우를 보여줌
				Rect rect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
				Scalar color(colors[r.id * 3], colors[r.id * 3 + 1], colors[r.id * 3 + 2]);
				int thickness = 2, lineType = 8, shift = 0;
				rectangle(img, rect, color, thickness, lineType, shift);

				if (1) {
					Point org(r.x1, r.y1 - 3);
					string text = coco_label[r.id] + " " + to_string(r.conf);
					putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
				}
				printf("%2d %4d %4d %4d %4d %2d %15s %.5f\n", idx, r.x1, r.y1, r.x2, r.y2, r.id, coco_label[r.id].c_str(), r.conf);
			}
			string bps_text = string("fps : ") + to_string(fps);
			putText(img, bps_text, Point(5, org_height - 6), cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
			imshow(model_name, img);
			int key = waitKey(1);
			if (key == 32) {
				waitKey(0);
			}
			else if (key == 27 || key == 'q' || key == 'Q') {
				break;
			}

		}

	}
	float avg_fps = 1000000.f / (total_usec / ITER);
	printf("average %.2f fps\n", avg_fps);


	freeSoyNet(handle);
}

int yolov8()
{


	char model_name[] = "yolov8n";

	// ### 완료
	//char model_name[] = "yolov8s";
	//char model_name[] = "yolov8m";
	//char model_name[] = "yolov8l";

	if (1) {

		string source_input = "../data/zidane.jpg";
		Mat img = cv::imread(source_input);
		int input_width = img.cols;
		int input_height = img.rows;
		vector<string> params = { source_input, };

		yolov8_img(input_height, input_width, params, model_name);
	}

	return 0;
}
/* nms 결과
tensor([[6.01854e+01, 1.10896e+02, 5.60192e+02, 3.67768e+02, 9.45480e-01, 0.00000e+00],
		[3.72996e+02, 3.25501e+01, 5.69566e+02, 3.67511e+02, 9.15290e-01, 0.00000e+00],
		[2.17575e+02, 2.30737e+02, 2.62292e+02, 3.70741e+02, 8.29365e-01, 2.70000e+01],
		[4.94870e+02, 1.70218e+02, 5.65998e+02, 3.62642e+02, 3.03374e-01, 2.70000e+01],
		[5.49785e+02, 2.09722e+02, 6.39624e+02, 3.69661e+02, 3.02512e-01, 0.00000e+00],
		[1.53511e-01, 3.10174e+02, 6.19860e+01, 3.71774e+02, 2.61197e-01, 0.00000e+00]], device='cuda:0')
*/
/* yolov8
zidane.jpg
 0  120  198 1120  712  0          person 0.94548
 1  746   41 1139  711  0          person 0.91529
 2  435  437  525  717 27             tie 0.82936
 3  990  316 1132  701 27             tie 0.30337
 4 1100  395 1279  715  0          person 0.30251
 5    0  596  124  720  0          person 0.26119
*/

/* yolov5m6r62
zidane.jpg
 0  121.000  202.000 1120.000  713.000  0          person 0.930953
 1  747.000   38.000 1146.000  714.000  0          person 0.909793
 2  375.000  437.000  524.000  719.000 27             tie 0.732315
*/




//yolov8l
//tensor([[1.21000e+02, 2.00000e+02, 1.11500e+03, 7.12000e+02, 9.40848e-01, 0.00000e+00],
//	[7.48000e+02, 4.10000e+01, 1.14700e+03, 7.10000e+02, 9.22650e-01, 0.00000e+00],
//	[4.35000e+02, 4.37000e+02, 5.26000e+02, 7.19000e+02, 9.00297e-01, 2.70000e+01]], device = 'cuda:0')]

//0  121  200 1115  712  0          person 0.94085
//1  748   41 1147  710  0          person 0.92265
//2  435  437  526  719 27             tie 0.90045




// yolov8s
//tensor([[7.45000e+02, 4.10000e+01, 1.13600e+03, 7.14000e+02, 8.94017e-01, 0.00000e+00],
//	[1.33000e+02, 2.00000e+02, 1.12700e+03, 7.14000e+02, 8.87118e-01, 0.00000e+00],
//	[4.38000e+02, 4.34000e+02, 5.31000e+02, 7.18000e+02, 7.40183e-01, 2.70000e+01],
//	[3.54000e+02, 4.35000e+02, 5.32000e+02, 7.18000e+02, 2.51489e-01, 2.70000e+01]], device = 'cuda:0')]

//0  745   41 1136  714  0          person 0.89399
//1  133  200 1127  714  0          person 0.88718
//2  438  434  531  718 27             tie 0.74015
//3  354  435  532  718 27             tie 0.25165



//yolo8n
//tensor([[1.23000e+02, 1.97000e+02, 1.11100e+03, 7.11000e+02, 8.05567e-01, 0.00000e+00],
//	[7.47000e+02, 4.10000e+01, 1.14200e+03, 7.12000e+02, 7.93672e-01, 0.00000e+00],
//	[4.37000e+02, 4.37000e+02, 5.24000e+02, 7.14000e+02, 3.70219e-01, 2.70000e+01]], device = 'cuda:0')]

//0  123  197 1111  711  0          person 0.80568
//1  747   41 1142  712  0          person 0.79341
//2  437  437  524  714 27             tie 0.36965
