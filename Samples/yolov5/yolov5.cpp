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
#include <algorithm>

using namespace std;
using namespace cv;
using namespace chrono;

#pragma pack(push, 1)
//typedef struct { float x1; float y1; float x2; float y2; float conf;  float obj_id; } BBox;
#pragma pack(pop)

static char soynet_home[] = "..";


static vector<uchar> colors = {
	0,252,255,0,118,255,0,70,255,0,67,255,0,10,255,0,6,255,0,0,255,0,204,255,0,185,255,0,143,255,0,134,255,0,214,255,0,45,255,0,131,255,0,220,255,0,207,255,0,112,255,0,77,255,0,194,255,0,175,255,0,38,255,0,64,255,0,3,255,0,48,255,0,54,255,0,163,255,0,41,255,0,153,255,0,172,255,0,102,255,0,239,255,0,223,255,0,51,255,0,99,255,0,182,255,0,61,255,0,89,255,0,198,255,0,191,255,0,92,255,0,179,255,0,140,255,0,86,255,0,188,255,0,150,255,0,105,255,0,159,255,0,83,255,0,226,255,0,115,255,0,166,255,0,108,255,0,242,255,0,35,255,0,96,255,0,13,255,0,16,255,0,156,255,0,32,255,0,26,255,0,201,255,0,73,255,0,230,255,0,128,255,0,245,255,0,137,255,0,147,255,0,29,255,0,19,255,0,121,255,0,236,255,0,249,255,0,169,255,0,233,255,0,124,255,0,217,255,0,22,255,0,57,255,0,210,255,0,80,255,
};


static vector<string> coco_label = {
	//"BG",
	"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

static void yolov5_img(int org_height, int org_width, vector<string>& params, char model_name[], int new_height, int new_width)
{


	const char* plugin_dir;

#ifdef NDEBUG
	plugin_dir = "../lib/plugins/";
#else
	plugin_dir = "../lib_debug/plugins/";
#endif

	int make_engine = 1;

	int region_count = 1000;
	int nms_count = 50; // 화면에 표시할 최대 객체의 수 <= 모델에서 정의한 최종 객체의 수
	int count_per_class = 10; // 한장의 이미지안에  class 당 최대 몇개인지..., 사람이 많아도 이놈이 한계! 
	float iou_thres = 0.45f;
	int batch_size = params.size();
	int class_count = coco_label.size();
	int result_count = batch_size * count_per_class * class_count; // 모든 batch를 통털어 전체 출력 갯수
	float conf_thres = 0.25f;
	//float conf_thres = 0.5f;
	char cfg_file[256];	sprintf(cfg_file, "../mgmt/configs/%s.cfg", model_name);
	//char cfg_file[256];	sprintf(cfg_file, "../mgmt/configs/%sr62.cfg", model_name);

	char license_file[] = "../mgmt/licenses/license_trial.key";
	char weight_file[256]; sprintf(weight_file, "../mgmt/weights/%s.weights", model_name);
	char engine_file[256]; sprintf(engine_file, "../mgmt/engines/%s.bin", model_name);
	char dict_file[] = "../layer_dict_V5.1.0.dct";
	char cache_file[256]; sprintf(cache_file, "../mgmt/engines/%s.cache", model_name);
	char log_dir[] = "../mgmt/logs";

	char extend_param[2000];

	int re_height, re_width; // resize가 필요한 경우 이 크기로...
	int top, bottom, left, right; // pad 크기
	int stride = 64;
	//int is_resize = calc_resize(re_height, re_width, new_height, new_width, org_height, org_width, stride, top, bottom, left, right);
	YOLO_SIZE s;
	int is_resize = calc_resize_yolo(&s, new_height, new_width, org_height, org_width, stride);

	sprintf(extend_param,
		"MODEL_CODE=%s SOYNET_HOME=%s BATCH_SIZE=%d MAKE_ENGINE=%d STRIDE=%d CLASS_COUNT=%d CONF_THRES=%f NMS_COUNT=%d REGION_COUNT=%d COUNT_PER_CLASS=%d RESULT_COUNT=%d IOU_THRES=%f RE_SIZE=%d,%d NEW_SIZE=%d,%d ORG_SIZE=%d,%d TOP=%d BOTTOM=%d LEFT=%d RIGHT=%d LICENSE_FILE=%s LOG_DIR=%s PLUGIN_DIR=%s WEIGHT_FILE=%s ENGINE_FILE=%s DICT_FILE=%s CACHE_FILE=%s",
		model_name, soynet_home, batch_size, make_engine, stride, class_count, conf_thres, nms_count, region_count, count_per_class, result_count, iou_thres, s.unpad_height, s.unpad_width, new_height, new_width, org_height, org_width, s.pad_top, s.pad_bottom, s.pad_left, s.pad_right, license_file, log_dir, plugin_dir, weight_file, engine_file, dict_file, cache_file);
	void* handle = initSoyNet(cfg_file, extend_param);

	//int re_map_size = re_height * re_width  * 3;
	int re_map_size = s.unpad_height * s.unpad_width * 3;
	vector<uchar> input(batch_size * re_map_size);


	int ITER = 50;
	int total_usec = 0;
	//vector<float> output(batch_size* nms_count * 6);

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
		if (1) {
			int output_count = getOutputCount(handle);
			for (int output_idx = 0; output_idx < output_count; output_idx++) {
				int elt_count = getOutputEltSize(handle, output_idx);
				int output_shape[10] = { 0 }; // 넉넉하게
				int dim_len;
				getOutputShape(handle, output_idx, output_shape, &dim_len);
				vector<int> ibuffer(elt_count);
				getOutput(handle, output_idx, ibuffer.data());

				vector<float> buffer(elt_count);
				getOutput(handle, output_idx, buffer.data());
				{
					std::ofstream ofs("../TEMP/R", std::ios::binary);
					ofs.write((char*)buffer.data(), buffer.size() * sizeof(float));
				}
				int jj = 0; 
			}
		}
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

static void yolo5_video(vector<string>& params, string source_type, char model_name[], int new_height, int new_width)
{
	const char* plugin_dir;

#ifdef NDEBUG
	plugin_dir = "../lib/plugins/";
#else
	plugin_dir = "../lib_debug/plugins/";
#endif


	int batch_size = params.size();
	vector<cv::VideoCapture> vcap;
	for (int vidx = 0; vidx < batch_size; vidx++) {
		if (source_type == "video") {
			vcap.emplace_back(VideoCapture(params[vidx]));
			if (!vcap[vidx].isOpened()) {
				printf("Error, Can't open video [%s]\n", params[vidx].c_str());
				exit(-1);
			}
		}
		else if (source_type == "cam") {
			vcap.emplace_back(VideoCapture(vidx));
			if (!vcap[vidx].isOpened()) {
				printf("Error, Can't open cam id [%d]", vidx);
				exit(-1);
			}
		}
		else {
			printf("Error, Not supprote source_type [%s], should be [video or cam]\n", source_type.c_str());
			exit(-1);
		}
	}
	int org_height = vcap[0].get(4);
	int org_width = vcap[0].get(3);

	int make_engine = 0;

	int region_count = 1000;
	int nms_count = 50; // 화면에 표시할 최대 객체의 수 <= 모델에서 정의한 최종 객체의 수
	int count_per_class = 10; // 한장의 이미지안에  class 당 최대 몇개인지..., 사람이 많아도 이놈이 한계! 
	float iou_thres = 0.45f;
	int class_count = coco_label.size();
	int result_count = batch_size * count_per_class * class_count; // 모든 batch를 통털어 전체 출력 갯수
	float conf_thres = 0.25f;
	//float conf_thres = 0.5f;
	char cfg_file[256];	sprintf(cfg_file, "../mgmt/configs/%s.cfg", model_name);
	//char cfg_file[256];	sprintf(cfg_file, "../mgmt/configs/%sr62.cfg", model_name);

	char license_file[] = "../mgmt/licenses/license_trial.key";
	char weight_file[256]; sprintf(weight_file, "../mgmt/weights/%s.weights", model_name);
	char engine_file[256]; sprintf(engine_file, "../mgmt/engines/%s.bin", model_name);
	char dict_file[] = "../layer_dict_V5.1.0.dct";
	char cache_file[256]; sprintf(cache_file, "../mgmt/engines/%s.cache", model_name);
	char log_dir[] = "../mgmt/logs";

	char extend_param[2000];

	int re_height, re_width; // resize가 필요한 경우 이 크기로...
	int top, bottom, left, right; // pad 크기
	int stride = 64;
	//int is_resize = calc_resize(re_height, re_width, new_height, new_width, org_height, org_width, stride, top, bottom, left, right);
	YOLO_SIZE s;
	int is_resize = calc_resize_yolo(&s, new_height, new_width, org_height, org_width, stride);


	sprintf(extend_param,
		"MODEL_CODE=%s SOYNET_HOME=%s BATCH_SIZE=%d MAKE_ENGINE=%d STRIDE=%d CLASS_COUNT=%d CONF_THRES=%f NMS_COUNT=%d REGION_COUNT=%d COUNT_PER_CLASS=%d RESULT_COUNT=%d IOU_THRES=%f RE_SIZE=%d,%d NEW_SIZE=%d,%d ORG_SIZE=%d,%d TOP=%d BOTTOM=%d LEFT=%d RIGHT=%d LICENSE_FILE=%s LOG_DIR=%s PLUGIN_DIR=%s WEIGHT_FILE=%s ENGINE_FILE=%s DICT_FILE=%s CACHE_FILE=%s",
		model_name, soynet_home, batch_size, make_engine, stride, class_count, conf_thres, nms_count, region_count, count_per_class, result_count, iou_thres, s.unpad_height, s.unpad_width, new_height, new_width, org_height, org_width, s.pad_top, s.pad_bottom, s.pad_left, s.pad_right, license_file, log_dir, plugin_dir, weight_file, engine_file, dict_file, cache_file);
	void* handle = initSoyNet(cfg_file, extend_param);

	uint64_t dur_microsec = 0;
	uint64_t count = 0;

	int is_break = 0;
	int is_bbox = 1;
	int is_text = 1;
	int is_fps = 1;
	int is_objInfo = 0;

	vector<Mat> img(batch_size);
	Mat img_resize;
	int re_map_size = s.unpad_height * s.unpad_width * 3;


	while (1) {


		//void* handle = initSoyNet(cfg_file, extend_param);


		/* #include <thread>
		this_thread::sleep_for(chrono::milliseconds(50));*/

		vector<uchar> input(batch_size * re_map_size);
		int loop = 1;
		for (int vidx = 0; vidx < batch_size; vidx++) {
			vcap[vidx] >> img[vidx];
			if (img[vidx].empty()) {
				loop = 0;
				break;
			}
			if (is_resize) {
				resize(img[vidx], img_resize, Size(s.unpad_width, s.unpad_height));
			}
			memcpy(&input[vidx * re_map_size], img_resize.data, re_map_size);
		}
		if (loop == 0) {
			break;
		}
		vector<YOLO_RESULT> output(result_count);


		long long start_usec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
		feedData(handle, 0, input.data());
		inference(handle);
		getOutput(handle, 0, output.data());
		long long end_usec = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
		uint64_t dur = end_usec - start_usec;
		dur_microsec += dur;
		for (int vidx = 0; vidx < batch_size; vidx++) {
			if (1) {
				YOLO_RESULT* res = &output[vidx * result_count];
				for (int idx = 0; idx < result_count && res[idx].conf>0.f; idx++) {
					YOLO_RESULT& r = res[idx];
					if (r.batch_idx != 0) continue; // 예시로 batch_idx == 0인 경우를 보여줌
					Rect rect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
					Scalar color(colors[r.id * 3], colors[r.id * 3 + 1], colors[r.id * 3 + 2]);
					
					rectangle(img[vidx], rect, color, 2, 8, 0);

					if (is_text) {
						Point org(r.x1, r.y1 - 3);
						//string prob_s = to_string(roundf(bbox[ridx].prob * 100) / 100).erase(4, 8); // 소수점 3번째 자리에서 반올림 후 string으로 변환
						string text = to_string(idx) + " " + coco_label[r.id] + " " + to_string(r.conf);
						putText(img[vidx], text, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
					}


					if (is_fps == 1) {// fps
						Point org2(15, org_height - 20);
						string text2 = "fps = " + to_string(1000000. / dur);
						putText(img[vidx], text2, org2, cv::FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255, 255, 255));
					}
				}
				String name = model_name + to_string(vidx);
				imshow(name, img[vidx]);

			}
		}
		printf("%lld time=%.1f msec fps=%.2f\n", count, dur / 1000., 1000000. / dur);

		int ret = waitKey(1);
		if (ret == 27 || ret == 'q' || ret == 'Q') {
			is_break = 1;
		}
		else if (ret == ' ') {
			waitKey(0);
		}
		else if (ret == 'b' || ret == 'B') {
			is_bbox ^= 1;
		}
		else if (ret == 't' || ret == 'T') {
			is_text ^= 1;
		}
		else if (ret == 'f' || ret == 'F') {
			is_fps ^= 1;
		}
		else if (ret == 'i' || ret == 'I') {
			is_objInfo ^= 1;
		}
		if (is_break == 1) break;
		count++;

	}
	printf("count=%lld total_time=%.1f msec avg_time=%.1f msec fps=%.2f\n",
		count, dur_microsec / 1000., dur_microsec / 1000. / count, count * 1000000. / dur_microsec);

	freeSoyNet(handle);

}



int yolov5()
{


	/*
	YOLOv5 c++코드 및 config에서 height, width 종류
	org    : 1080,810 jpeg 원본 크기
	new    :  640,640 "이정도" 크기로 resize해서 모델에 집어 넣어야지...
	resize :  640,480 원본의 가로/세로 비율을 유지하면서 new에 맞춘 놈
	model  :  640,512 resize에 pad를 붙여서 stride에 맞춘크기
	---------------------------------------------------------------------
	preproc : 정합성을 위해 정수 resize해서 feedData 수행
			  (resize -> new) -> model
	postproc : (resize, new, orig)
	*/


	//char model_name[] = "yolov5n6r62";
	char model_name[] = "yolov5l6r62";
	//char model_name[] = "yolov5s6r62";
	//char model_name[] = "yolov5m6r62";

	if (1) {
		
		string source_input = "../data/zidane.jpg"; //
		//string source_input = "../data/bus.jpg";


		vector<string> params = { source_input
			//, source_input
			//, source_input , source_input 
		};

		Mat img = cv::imread(source_input);
		//int new_height = 512-64, new_width = 512-64;
		int new_height = 640, new_width = 640;
		//int new_height = 1280, new_width = 1280;
		int org_height = img.rows;
		int org_width = img.cols;

		yolov5_img(org_height, org_width, params, model_name, new_height, new_width);

	}
	else {
		string source_type = "video";

		vector<string> params = { 
			"../data/video.mp4",// "../data/video.mkv", "../data/video.mkv", "../data/video.mkv", 
			//"../data/video.mkv", "../data/video.mkv","../data/video.mkv", "../data/video.mkv",
			//"../data/video.mkv", "../data/video.mkv","../data/video.mkv", "../data/video.mkv",
			//"../data/video.mkv", "../data/video.mkv","../data/video.mkv", "../data/video.mkv",
		};

		int new_height = 640, new_width = 640;

		yolo5_video( params, source_type, model_name, new_height, new_width);

	}
	
	return 0;
}

/*
zidane.jpg
 0  121.000  202.000 1120.000  713.000  0          person 0.930953
 1  747.000   38.000 1146.000  714.000  0          person 0.909793
 2  375.000  437.000  524.000  719.000 27             tie 0.732315
*/