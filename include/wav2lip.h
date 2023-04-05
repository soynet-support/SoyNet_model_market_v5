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

struct BOX { int x1, y1, x2, y2; };

#ifdef __cplusplus
extern "C" {
#endif

	SOYNET_DLL void getOutputWav2LipFaceDetect(const void* handle, BOX* output, int batch_size, int count, int height, int width, int py1, int py2, int px1, int px2);
	// handle : SoyNet 객체의 포인터(핸들)
	// output : BOX{int x1,y1,x2,y2}의 배열를 출력하는 저장공간
	// count : batch_size=128개 중에서 유효한 숫자
	//         ex. 총 207개인 경우 batch_size=128이면 
	//             batch_idx=0인 경우 count=128개가 유효
	//             batch_idx=1인 경우 count=79개만 유효 
	// height,width : detection된 얼굴의 위치를 원본 이미지 (height,width) 기준으로 환산하기 위한 것
	SOYNET_DLL void smoothFaceBBox(BOX* bboxes, int count, int T);
	// bboxes : BOX{int x1,y1,x2,y2}의 배열를 입/출력하는 저장공간
	// count : BOX의 갯수
	// T : smoothing 처리 window 크기

#ifdef __cplusplus
}
#endif
