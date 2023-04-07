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

enum TrackerType { BOT_SORT=0, };
enum BBoxType { XYXY = 0, XYWH = 1 };
enum BBoxDType { F32 = 0, U32 = 1 };
enum DetType { YOLOX = 0, YOLO=1};


#pragma pack(push, 1)
#pragma pack(pop)

#ifdef __cplusplus
extern "C" {
#endif

	SOYNET_DLL void* initByteTrack(float track_thres);
	SOYNET_DLL int doByteTrack(void* handle, void* det_infos, int det_count);
	SOYNET_DLL void freeByteTrack(void* handle);

#ifdef __cplusplus
}
#endif
