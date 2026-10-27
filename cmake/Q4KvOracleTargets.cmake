# These tools require a matching managed CUDA engine and CUDA SDK. Keeping the
# option off leaves CPU, ROCm, Jetson and normal package builds SDK-independent.
if(NOT EXISTS "${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/src/ggml-cuda/fattn-q4-kv-policy.h" OR
   NOT EXISTS "${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml-cuda.so")
    message(FATAL_ERROR "Q4 KV GPU oracles require a managed CUDA llama.cpp source/build")
endif()
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "Q4 KV GPU oracles require a CUDA compiler")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_executable(test_q4_kv_loader EXCLUDE_FROM_ALL tests/test_q4_kv_loader.cu)
set_target_properties(test_q4_kv_loader PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES 89)
# Match the managed ggml-cuda converter/loader's CUDA math flags.
target_compile_options(test_q4_kv_loader PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--extended-lambda>")
target_include_directories(test_q4_kv_loader PRIVATE
    ${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/include
    ${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/src
    ${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/src/ggml-cuda)
target_link_libraries(test_q4_kv_loader PRIVATE nlohmann_json::nlohmann_json CUDA::cudart
    ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml-cuda.so
    ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml-base.so)

add_executable(test_q4_kv_attention_operator EXCLUDE_FROM_ALL tests/test_q4_kv_attention_operator.cpp)
target_include_directories(test_q4_kv_attention_operator PRIVATE
    ${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/include
    ${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/src/ggml-cuda)
target_link_libraries(test_q4_kv_attention_operator PRIVATE nlohmann_json::nlohmann_json CUDA::cudart ${CMAKE_DL_LIBS}
    ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml-cuda.so
    ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml.so
    ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml-base.so)
add_custom_target(q4_kv_gpu_oracles
    DEPENDS test_q4_kv_loader test_q4_kv_attention_operator)
