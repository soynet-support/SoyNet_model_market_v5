#pragma once

#ifdef _WIN32
#ifdef __SOYNET__
#define SOYNET_DLL __declspec(dllexport)
#else
#define SOYNET_DLL __declspec(dllimport)
#endif
#else
#define SOYNET_DLL 
#endif

#pragma pack(push, 1)
struct YOLO_RESULT { int batch_idx, x1, y1, x2, y2, id; float conf; };
struct YOLOX_RESULT { int  batch_idx; float x1, y1, x2, y2; int id; float conf; };
struct YOLO_SIZE {
	int unpad_height, unpad_width;
	int pad_top, pad_bottom, pad_left, pad_right;
};
#pragma pack(pop)

#ifdef __cplusplus
extern "C" {
#endif

	SOYNET_DLL int calc_resize_yolo( YOLO_SIZE* ys, int nH, int nW, int H, int W, int stride);
	SOYNET_DLL int calc_resize_yolox(YOLO_SIZE* ys, int nH, int nW, int H, int W);

#ifdef __cplusplus
}
#endif
