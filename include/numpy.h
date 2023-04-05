#pragma once

#include <cstdlib>

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

	SOYNET_DLL void npy_load(void* data, size_t* byte_size, const char* npy_name);

#ifdef __cplusplus
}
#endif


