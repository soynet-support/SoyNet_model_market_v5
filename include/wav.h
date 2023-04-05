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

#ifdef __cplusplus
extern "C" {
#endif

    SOYNET_DLL void save_pcm_to_wav(const char* wav_file_name, char* pcm_data, int pcm_byte_size, int sampling_rate);

#ifdef __cplusplus
}
#endif
