#pragma once

#ifdef _WIN32
#ifdef __SAMPLES__
#define SAMPLES_DLL __declspec(dllexport)
#else
#define SAMPLES_DLL __declspec(dllimport)
#endif
#else
#define SAMPLES_DLL 
#endif

#ifdef __cplusplus
extern "C" {
#endif
	SAMPLES_DLL void efficientnet(const void* handle, void* output, int* time,
		char* model_code, char* image_path, int batch_size);
	SAMPLES_DLL void yolo(const void* handle, void* output, int* time,
		char* image_path, int batch_size);
	SAMPLES_DLL void inception_resnet_v2(const void* handle, void* output, int* time,
		char* image_path, int batch_size);
	SAMPLES_DLL void vgg(const void* handle, void* output, int* time,
		char* image_path, int batch_size);
	SAMPLES_DLL void bert(const void* handle, void* output, int* outputSize, int* time,
		char* token_data, int batch_size);
	//SAMPLES_DLL void palm(const void* handle, void* output, int* outputSize, int* time,
	//	char* token_path, int batch_size);
	SAMPLES_DLL void resnet(const void* handle, void* output, int* time,
		char* image_path, int batch_size);
	//SAMPLES_DLL void yolov4(const void* handle, void* output, int* time,
	//	char* image_path, int batch_size);
	SAMPLES_DLL void gleanX8(const void* handle, void* output, int* time,
		char* image_path, int batch_size);
	SAMPLES_DLL void maskrcnn(const void* handle, void* output1, float* contours, int* point_num, int* time,
		char* image_path, int batch_size);
	SAMPLES_DLL void tips_yolo_test(const void* handle, void* output, int* time, char* path,
		int save_result, char* result_img_path, int video_test);

#ifdef __cplusplus
}
#endif
