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

#pragma pack(pop)

#ifdef __cplusplus
extern "C" {
#endif

	SOYNET_DLL int regulate_len_for_fastspeech(const void* handle, int output_idx, int* reg_len);
	// handle : SoyNet 객체의 포인터(핸들)
	// input : output_idx 모델에서 출력으로 설정된 순번, config 파일의 [output] refname=... 참조
	// return : reg_len의 길이
	// output : reg_len (충분한 공간이 확보되어 있어야 함, ex MEL_MAX_LEN=10000

#ifdef __cplusplus
}
#endif
