#pragma once

#include <stdio.h>

#ifdef _WIN32
#ifdef __SOYNET__
#define SOYNET_DLL __declspec(dllexport)
#else
#define SOYNET_DLL __declspec(dllimport)
#endif
#else
#define SOYNET_DLL 
#endif

#ifdef __cplusplus
extern "C" {
#endif
	SOYNET_DLL void* initSoyNet(const char* cfg_file_name, const char* extend_param); // SoyNet handle을 return한다.
	SOYNET_DLL void* initSoyNetV(const char* cfg_file_name, const char* extend_param, void* handle); // SoyNet handle을 return한다.
	SOYNET_DLL void feedData(const void * handle, int idx, const void* data);
	SOYNET_DLL void feedDataV(const void* handle, int idx, const void* data, const int* shape);
	SOYNET_DLL void inference(const void * handle);
	SOYNET_DLL void getOutput(const void * handle, int idx, void * output);
	SOYNET_DLL int getOutputCount(const void* handle);
	SOYNET_DLL void getOutputShape(const void* handle, int idx, int* dims, int* dims_len);
	SOYNET_DLL size_t getOutputEltSize(const void* handle, int idx);
	SOYNET_DLL size_t getOutputByteSize(const void* handle, int idx);
	SOYNET_DLL void freeSoyNet(const void* handle);

#ifdef __cplusplus
}
#endif


#ifdef __SOYNET_JAVA__

#include "jni.h"

#ifdef __cplusplus
extern "C" {
#endif

	JNIEXPORT jlong JNICALL Java_SoyNet_initSoyNet(JNIEnv *, jclass, jstring, jstring);
	JNIEXPORT void JNICALL Java_SoyNet_feedData(JNIEnv *, jclass, jlong, jbyteArray);
	JNIEXPORT void JNICALL Java_SoyNet_inference(JNIEnv *, jclass, jlong);
	JNIEXPORT void JNICALL Java_SoyNet_getOutput(JNIEnv *, jclass, jlong, jfloatArray);
	JNIEXPORT void JNICALL Java_SoyNet_freeSoyNet(JNIEnv *, jclass, jlong);

#ifdef __cplusplus
}
#endif
#endif