cmake_minimum_required(VERSION 3.16)

project(ezlocalai
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The Compose profile gate needs only the Docker client and nlohmann_json, so
# it can run on hosts without the native inference dependencies:
#   cmake -S . -B build-compose -DEZLOCALAI_COMPOSE_GATE_ONLY=ON
option(EZLOCALAI_COMPOSE_GATE_ONLY "Configure only the host Compose profile gate" OFF)
if(EZLOCALAI_COMPOSE_GATE_ONLY)
    enable_testing()
    include(cmake/ComposeGate.cmake)
    return()
endif()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/licenses/SciPy.txt" DESTINATION share/ezlocalai/licenses)

# Docker excludes Git metadata. Identify the actual first-party runtime sources,
# including uncommitted changes, without requiring Git or Python at runtime.
file(GLOB_RECURSE EZLOCALAI_VERSION_SOURCES CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "src/*" "include/*" "engine/*" "patches/*" "cmake/*")
list(APPEND EZLOCALAI_VERSION_SOURCES CMakeLists.txt cuda.Dockerfile)
list(SORT EZLOCALAI_VERSION_SOURCES)
set(EZLOCALAI_VERSION_DIGEST_INPUT "")
foreach(EZLOCALAI_VERSION_SOURCE IN LISTS EZLOCALAI_VERSION_SOURCES)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${EZLOCALAI_VERSION_SOURCE}")
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${EZLOCALAI_VERSION_SOURCE}" EZLOCALAI_VERSION_HASH)
    string(APPEND EZLOCALAI_VERSION_DIGEST_INPUT "${EZLOCALAI_VERSION_SOURCE}:${EZLOCALAI_VERSION_HASH}\n")
endforeach()
string(SHA256 EZLOCALAI_VERSION_HASH "${EZLOCALAI_VERSION_DIGEST_INPUT}")
string(SUBSTRING "${EZLOCALAI_VERSION_HASH}" 0 12 EZLOCALAI_VERSION_HASH)
if(NOT DEFINED EZLOCALAI_BUILD_VERSION)
    set(EZLOCALAI_BUILD_VERSION "ezlocalai-${PROJECT_VERSION}+${EZLOCALAI_VERSION_HASH}")
endif()
get_filename_component(EZLOCALAI_RUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
configure_file(cmake/runtime_build_version.hpp.in generated/runtime_build_version.hpp @ONLY)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc videoio)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(WEBP REQUIRED IMPORTED_TARGET libwebp libwebpdemux)
pkg_check_modules(BROTLI REQUIRED IMPORTED_TARGET libbrotlidec)
pkg_check_modules(WHISPER_FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswresample)
pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)

set(EZLOCALAI_SNDFILE_LIBRARY "" CACHE FILEPATH "Optional exact libsndfile build used by the source soundfile package")
if(EZLOCALAI_SNDFILE_LIBRARY)
    if(NOT EXISTS "${EZLOCALAI_SNDFILE_LIBRARY}")
        message(FATAL_ERROR "EZLOCALAI_SNDFILE_LIBRARY must name an existing native shared library")
    endif()
    set_source_files_properties(src/ctts_audio.cpp src/llama_tts.cpp PROPERTIES COMPILE_DEFINITIONS "EZLOCALAI_SNDFILE_LIBRARY=\"${EZLOCALAI_SNDFILE_LIBRARY}\"")
endif()

add_library(ezlocalai_core
    src/core.cpp
    src/dotenv.cpp
    src/tts_text.cpp
    src/configuration.cpp
    src/request_queue.cpp
    src/cuda_mps.cpp
)
add_library(ezlocalai_c ALIAS ezlocalai_core)
target_include_directories(ezlocalai_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(ezlocalai_core
    PUBLIC
        Threads::Threads
)

add_library(ezlocalai_native
    src/native_server.cpp
    src/worker_models.cpp
    src/runtime_version.cpp
    src/worker_heartbeat.cpp
    src/tunnel_websocket.cpp
    src/tunnel_connection.cpp
    src/voice_conversion.cpp
    src/realtime_voice.cpp
    src/realtime.cpp
    src/wakeword_paths.cpp
    src/wakeword_samples.cpp
    src/wakeword_tts.cpp
    src/wakeword_jobs.cpp
    src/edge_tts.cpp
    src/edge_tts_transport.cpp
    src/torch_checkpoint.cpp
    src/http_deadline.cpp
    src/http_body.cpp
    src/request_gzip.cpp
    src/http_url.cpp
    src/http_cookies.cpp
    src/music_runtime.cpp
    src/music_video.cpp
    src/ltx_video.cpp
    src/media_engine_process.cpp
    src/ltx_runtime.cpp
    src/hub_assets.cpp
    src/hub_local.cpp
    src/hub_snapshot.cpp
    src/whisper_model.cpp
    src/model_download.cpp
    src/wan_video.cpp
    src/qwen_image.cpp
    src/image_processing.cpp
    src/llm_video.cpp
    src/whisper_features.cpp
    src/whisper_vad.cpp
    src/whisper_tokenizer.cpp
    src/whisper_transcription.cpp
    src/stt_diarization.cpp
    src/stt_runtime.cpp
    src/stt_pool.cpp
    src/stt_service.cpp
    src/llama_tts.cpp
    src/ctts_runtime.cpp
    src/ctts_audio.cpp
    src/ctts_paths.cpp
    src/ctts_cache.cpp
    src/ctts.cpp
    src/ctts_stream_bridge.cpp
    src/tts_pool.cpp
    src/tts_service.cpp
    src/tts_routing.cpp
    src/tts_websocket.cpp
    src/tts_pipeline.cpp
    src/gtts.cpp
    src/gtts_transport.cpp
    src/subtitles.cpp
    src/voice_slots.cpp
    src/stt_audio.cpp
    src/json_validation.cpp
    src/video_runtime.cpp
    src/resource_manager.cpp
    src/model_lifecycle.cpp
    src/llm_load.cpp
    src/llm_batch.cpp
    src/llm_cache.cpp
    src/llm_speculative.cpp
    src/llm_parameters.cpp
    src/llm_context.cpp
    src/llm_catalog.cpp
    src/llm_slots.cpp
    src/llm_residency.cpp
    src/llm_stream.cpp
    src/embedding_runtime.cpp
    src/embedding_placement.cpp
)
target_include_directories(ezlocalai_native
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(ezlocalai_native PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(ezlocalai_native
    PUBLIC
        ezlocalai_core
        Threads::Threads
        OpenSSL::SSL
        OpenSSL::Crypto
        ZLIB::ZLIB
        ICU::uc
        PkgConfig::BROTLI
        PNG::PNG
        JPEG::JPEG
        PkgConfig::WEBP
        ${OpenCV_LIBS}
        PkgConfig::WHISPER_FFMPEG
        ${CMAKE_DL_LIBS}
    PRIVATE
        nlohmann_json::nlohmann_json
)
target_include_directories(ezlocalai_native PRIVATE ${OPENBLAS_INCLUDE_DIRS})
option(EZLOCALAI_USE_HIP "Use AMD HIP for device discovery and memory probes" OFF)
if(EZLOCALAI_USE_HIP)
    target_compile_definitions(ezlocalai_native PRIVATE EZLOCALAI_USE_HIP=1)
    find_path(EZLOCALAI_HIP_INCLUDE_DIR hip/hip_runtime_api.h HINTS /opt/rocm/include)
    if(NOT EZLOCALAI_HIP_INCLUDE_DIR)
        message(FATAL_ERROR "HIP builds require SDK headers for the runtime ABI gate")
    endif()
endif()
list(GET OPENBLAS_LINK_LIBRARIES 0 EZLOCALAI_WHISPER_OPENBLAS_LIBRARY)
set_source_files_properties(src/whisper_features.cpp PROPERTIES
    COMPILE_DEFINITIONS "EZLOCALAI_OPENBLAS_LIBRARY=\"${EZLOCALAI_WHISPER_OPENBLAS_LIBRARY}\"")

add_executable(ezlocalai-native
    src/main.cpp
)
target_link_libraries(ezlocalai-native
    PRIVATE
        ezlocalai_native
)

add_executable(parity_report
    tools/parity_report.cpp
)

add_executable(llama_cpp_tool tools/llama_cpp.cpp)
target_link_libraries(llama_cpp_tool PRIVATE ezlocalai_native)
target_include_directories(llama_cpp_tool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(llama_cpp_tool PRIVATE EZLOCALAI_NATIVE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(ezlocalai-native-entrypoint src/docker_entrypoint.cpp)
target_link_libraries(ezlocalai-native-entrypoint PRIVATE ezlocalai_core)
target_include_directories(ezlocalai-native-entrypoint PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_cuda_mps tests/test_cuda_mps.cpp src/cuda_mps.cpp)
target_include_directories(test_cuda_mps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_mps PRIVATE Threads::Threads)
add_test(NAME cuda_mps COMMAND test_cuda_mps)
set_tests_properties(cuda_mps PROPERTIES TIMEOUT 60)

add_custom_target(llama_cpp_update
    COMMAND llama_cpp_tool update
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Updating ezlocalai-managed llama.cpp checkout"
)

add_custom_target(llama_cpp_build
    COMMAND llama_cpp_tool build
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building ezlocalai-managed llama-server"
)

add_custom_target(llama_cpp_print_bin
    COMMAND llama_cpp_tool print-bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Printing ezlocalai-managed llama-server path"
)

add_custom_target(llama_cpp_version
    COMMAND llama_cpp_tool version
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Printing ezlocalai-managed llama.cpp version"
)

add_custom_target(llama_cpp_status
    COMMAND llama_cpp_tool status
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Printing ezlocalai-managed llama.cpp status"
)

add_custom_target(llama_cpp_model_status
    COMMAND llama_cpp_tool model-status
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Printing ezlocalai model resolution status for llama.cpp"
)

enable_testing()

add_executable(test_core
    tests/test_core.cpp
)
target_link_libraries(test_core
    PRIVATE
        ezlocalai_core
)
add_test(NAME core COMMAND test_core)
add_executable(test_http_stream_socket tests/test_http_stream_socket.cpp)
target_include_directories(test_http_stream_socket PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_http_stream_socket PRIVATE Threads::Threads)
add_test(NAME http_stream_socket COMMAND test_http_stream_socket)
set_tests_properties(http_stream_socket PROPERTIES TIMEOUT 15)
add_executable(test_request_gzip_codec tests/test_request_gzip_codec.cpp src/request_gzip.cpp)
target_include_directories(test_request_gzip_codec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_request_gzip_codec PRIVATE ZLIB::ZLIB)
add_test(NAME request_gzip_codec COMMAND test_request_gzip_codec)
set_tests_properties(request_gzip_codec PROPERTIES TIMEOUT 20)
add_executable(test_request_gzip_http tests/test_request_gzip_http.cpp)
target_link_libraries(test_request_gzip_http PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME request_gzip_http COMMAND test_request_gzip_http)
set_tests_properties(request_gzip_http PROPERTIES TIMEOUT 60)
add_executable(bench_stream_socket EXCLUDE_FROM_ALL benchmarks/bench_stream_socket.cpp)
target_include_directories(bench_stream_socket PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bench_stream_socket PRIVATE Threads::Threads)

# Explicit live-model acceptance runner; deployment lifecycle belongs to the caller.
add_executable(test_speculative_model tests/test_speculative_model.cpp)
target_include_directories(test_speculative_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_speculative_model PRIVATE nlohmann_json::nlohmann_json OpenSSL::Crypto Threads::Threads)

add_executable(test_dotenv tests/test_dotenv.cpp)
target_link_libraries(test_dotenv PRIVATE ezlocalai_core nlohmann_json::nlohmann_json)
add_test(NAME dotenv_reference COMMAND test_dotenv ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/dotenv_cases.json)

add_executable(test_dotenv_startup tests/test_dotenv_startup.cpp)
target_link_libraries(test_dotenv_startup PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
target_include_directories(test_dotenv_startup PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_dependencies(test_dotenv_startup ezlocalai-native)
add_test(NAME dotenv_startup COMMAND test_dotenv_startup $<TARGET_FILE:ezlocalai-native>)
set_tests_properties(dotenv_startup PROPERTIES TIMEOUT 60)

add_executable(test_queue_history tests/test_queue_history.cpp)
target_link_libraries(test_queue_history PRIVATE ezlocalai_core)
add_test(NAME queue_history COMMAND test_queue_history)
set_tests_properties(queue_history PROPERTIES TIMEOUT 60 ENVIRONMENT "NATIVE_QUEUE_RESULT_BYTES=65536")

add_executable(test_queue_http tests/test_queue_http.cpp)
target_link_libraries(test_queue_http PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME queue_http COMMAND test_queue_http)
set_tests_properties(queue_http PROPERTIES TIMEOUT 60 ENVIRONMENT "NATIVE_QUEUE_RESULT_BYTES=65536")

add_executable(test_native_http
    tests/test_native_http.cpp
)
add_executable(test_native_fixture tests/process_fixture.cpp)
target_compile_definitions(test_native_http PRIVATE TEST_NATIVE_FIXTURE_BIN="$<TARGET_FILE:test_native_fixture>")
add_dependencies(test_native_http test_native_fixture)
target_link_libraries(test_native_http
    PRIVATE
        ezlocalai_native
)
add_test(NAME native_http COMMAND test_native_http)
add_executable(test_worker_heartbeat tests/test_worker_heartbeat.cpp src/worker_heartbeat.cpp)
target_include_directories(test_worker_heartbeat PRIVATE include)
target_link_libraries(test_worker_heartbeat PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
add_test(NAME worker_heartbeat_reference COMMAND test_worker_heartbeat ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/worker_heartbeat_cases.json)
add_executable(test_worker_heartbeat_http tests/test_worker_heartbeat_http.cpp)
target_link_libraries(test_worker_heartbeat_http PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME worker_heartbeat_http COMMAND test_worker_heartbeat_http ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/worker_heartbeat_http_cases.json)
set_tests_properties(worker_heartbeat_http PROPERTIES TIMEOUT 60)
add_test(NAME worker_heartbeat_shutdown COMMAND test_worker_heartbeat_http
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/worker_heartbeat_http_cases.json
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/worker_heartbeat_shutdown_cases.json)
set_tests_properties(worker_heartbeat_shutdown PROPERTIES TIMEOUT 45)
add_executable(test_openapi tests/test_openapi.cpp)
target_link_libraries(test_openapi PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME openapi_reference COMMAND test_openapi ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/openapi_cases.json)
set_tests_properties(openapi_reference PROPERTIES TIMEOUT 30)
add_executable(test_worker_tunnel tests/test_worker_tunnel.cpp)
target_link_libraries(test_worker_tunnel PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME worker_tunnel COMMAND test_worker_tunnel)
set_tests_properties(worker_tunnel PROPERTIES TIMEOUT 60)
add_executable(test_stt_diarization tests/test_stt_diarization.cpp)
target_link_libraries(test_stt_diarization PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME stt_diarization_reference COMMAND test_stt_diarization ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_diarization_cases.json ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_diarization_assets)
set_tests_properties(stt_diarization_reference PROPERTIES TIMEOUT 90)

add_executable(test_tunnel_idle tests/test_tunnel_idle.cpp)
target_link_libraries(test_tunnel_idle PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tunnel_idle_reference COMMAND test_tunnel_idle ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tunnel_deadlines_cases.json)
set_tests_properties(tunnel_idle_reference PROPERTIES TIMEOUT 90)

add_executable(test_tunnel_transport tests/test_tunnel_transport.cpp)
target_link_libraries(test_tunnel_transport PRIVATE ezlocalai_native nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
add_test(NAME tunnel_transport_reference COMMAND test_tunnel_transport ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tunnel_deadlines_cases.json)
set_tests_properties(tunnel_transport_reference PROPERTIES TIMEOUT 45)

option(EZLOCALAI_REQUIRE_MEANVC2 "Require the packaged native MeanVC2 engine" OFF)
find_program(EZLOCALAI_MEANVC2_BIN NAMES ezlocalai-meanvc2)
if(EZLOCALAI_REQUIRE_MEANVC2 AND NOT EZLOCALAI_MEANVC2_BIN)
    message(FATAL_ERROR "The packaged MeanVC2 executable is missing")
endif()
if(EZLOCALAI_MEANVC2_BIN)
    add_custom_target(meanvc2_engine_check COMMAND ${EZLOCALAI_MEANVC2_BIN} --version)
    add_test(NAME voice_conversion_engine_version COMMAND ${EZLOCALAI_MEANVC2_BIN} --version)
    set_tests_properties(voice_conversion_engine_version PROPERTIES TIMEOUT 15
        PASS_REGULAR_EXPRESSION "cd770076152085af5b166521185bec76f9d47808.*reference-v1")
endif()
add_executable(test_voice_conversion tests/test_voice_conversion.cpp)
target_link_libraries(test_voice_conversion PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME voice_conversion COMMAND test_voice_conversion)
set_tests_properties(voice_conversion PROPERTIES TIMEOUT 45)
add_executable(test_realtime tests/test_realtime.cpp)
target_link_libraries(test_realtime PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME realtime COMMAND test_realtime)
set_tests_properties(realtime PROPERTIES TIMEOUT 45)
add_executable(test_tunnel_connection tests/test_tunnel_connection.cpp)
target_link_libraries(test_tunnel_connection PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tunnel_connection_reference COMMAND test_tunnel_connection ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tunnel_connection_cases.json)
add_executable(test_tunnel_redirects tests/test_tunnel_redirects.cpp)
target_link_libraries(test_tunnel_redirects PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tunnel_redirect_reference COMMAND test_tunnel_redirects ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tunnel_redirect_cases.json)
set_tests_properties(tunnel_redirect_reference PROPERTIES TIMEOUT 30)
add_executable(test_tunnel_framing tests/test_tunnel_framing.cpp src/tunnel_websocket.cpp)
target_include_directories(test_tunnel_framing PRIVATE include)
target_link_libraries(test_tunnel_framing PRIVATE nlohmann_json::nlohmann_json OpenSSL::Crypto ICU::uc)
add_test(NAME tunnel_framing_reference COMMAND test_tunnel_framing ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tunnel_framing_cases.json)
set_tests_properties(tunnel_framing_reference PROPERTIES TIMEOUT 30)
add_executable(test_service_clients tests/test_service_clients.cpp)
target_link_libraries(test_service_clients PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME service_clients_reference COMMAND test_service_clients ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/service_client_cases.json ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/service_response_cases.json)
set_tests_properties(service_clients_reference PROPERTIES TIMEOUT 60)
add_executable(test_worker_metadata tests/test_worker_metadata.cpp)
add_executable(test_file_response tests/test_file_response.cpp)
target_link_libraries(test_file_response PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME file_response_reference COMMAND test_file_response ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/file_response_cases.json)
target_link_libraries(test_worker_metadata PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME worker_metadata_reference COMMAND test_worker_metadata ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/worker_extra_cases.json ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/worker_model_reporting_cases.json)
set_tests_properties(worker_metadata_reference PROPERTIES TIMEOUT 90)
set(EZLOCALAI_WORKER_ROUTER_BIN "" CACHE FILEPATH "Rust WorkerRouter executable for worker protocol integration tests")
add_executable(test_cuda_architectures tests/test_cuda_architectures.cpp)
target_include_directories(test_cuda_architectures PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_cuda_architectures PRIVATE Threads::Threads)
add_test(NAME cuda_architecture_policy COMMAND test_cuda_architectures --self-test)
set(EZLOCALAI_CUDA_FLEET_PROFILE "" CACHE STRING "Packaged CUDA target inventory: cuda, gb10 or jetson; empty for ordinary builds")
if(EZLOCALAI_CUDA_FLEET_PROFILE)
    if(NOT EZLOCALAI_CUDA_FLEET_PROFILE MATCHES "^(cuda|gb10|jetson)$")
        message(FATAL_ERROR "Unknown CUDA fleet profile: ${EZLOCALAI_CUDA_FLEET_PROFILE}")
    endif()
    find_program(EZLOCALAI_CUOBJDUMP cuobjdump HINTS /usr/local/cuda/bin REQUIRED)
    add_test(NAME cuda_package_architectures COMMAND test_cuda_architectures
        ${EZLOCALAI_CUDA_FLEET_PROFILE} ${EZLOCALAI_CUOBJDUMP}
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml-cuda.so
        ${EZLOCALAI_SDCPP_BUILD}/bin/sd-cli
        /opt/acestep.cpp/build/libggml-cuda.so
        ${EZLOCALAI_CTRANSLATE2_LIBRARY}
        ${EZLOCALAI_LIBTORCH_ROOT}/lib/libtorch_cuda.so
        $<TARGET_FILE:ezlocalai-native> $<TARGET_FILE:ezlocalai-tts>)
    set_tests_properties(cuda_package_architectures PROPERTIES TIMEOUT 1800)
endif()
if(EZLOCALAI_WORKER_ROUTER_BIN)
    add_executable(test_worker_router tests/test_worker_router.cpp)
    target_include_directories(test_worker_router PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(test_worker_router PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
    add_dependencies(test_worker_router ezlocalai-native)
    target_compile_definitions(test_worker_router PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
    add_dependencies(test_worker_router test_embedding_fixture)
    add_test(NAME worker_router_integration COMMAND test_worker_router $<TARGET_FILE:ezlocalai-native> ${EZLOCALAI_WORKER_ROUTER_BIN})
    set_tests_properties(worker_router_integration PROPERTIES TIMEOUT 90)
endif()
set_tests_properties(native_http PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 180
)

add_test(
    NAME parity_manifest
    COMMAND parity_report ${CMAKE_CURRENT_SOURCE_DIR}/parity_manifest.tsv
)

add_executable(test_llama_cpp_tool tests/test_llama_cpp_tool.cpp)
target_include_directories(test_llama_cpp_tool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME llama_cpp_tool COMMAND test_llama_cpp_tool $<TARGET_FILE:llama_cpp_tool>)

add_executable(test_parity_report tests/test_parity_report.cpp)
target_include_directories(test_parity_report PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME parity_report_gate COMMAND test_parity_report $<TARGET_FILE:parity_report>)

set(EZLOCALAI_REFERENCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../ezlocalai" CACHE PATH "Reference checkout for differential parity tests")
add_executable(test_configuration tests/test_configuration.cpp)
add_executable(test_embedding_discovery tests/test_embedding_discovery.cpp)
target_link_libraries(test_embedding_discovery PRIVATE ezlocalai_core)
add_test(NAME embedding_discovery COMMAND test_embedding_discovery)
target_link_libraries(test_configuration PRIVATE ezlocalai_core OpenSSL::Crypto)
add_test(NAME configuration_reference COMMAND test_configuration ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/configuration_cases.tsv)

add_executable(test_embedding tests/test_embedding.cpp)
target_link_libraries(test_embedding PRIVATE ezlocalai_native)
add_test(NAME embedding_reference COMMAND test_embedding ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/embedding_cases.tsv)

add_executable(test_router_maintenance tests/test_router_maintenance.cpp)
target_link_libraries(test_router_maintenance PRIVATE ezlocalai_native)
add_test(NAME router_maintenance_reference COMMAND test_router_maintenance ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/router_maintenance_cases.tsv)
set_tests_properties(router_maintenance_reference PROPERTIES TIMEOUT 30)

add_executable(test_music tests/test_music.cpp)
target_link_libraries(test_music PRIVATE ezlocalai_native)
add_test(NAME music_reference COMMAND test_music ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/music_cases.tsv)
set_tests_properties(music_reference PROPERTIES TIMEOUT 90)

add_executable(test_music_runtime tests/test_music_runtime.cpp)
target_link_libraries(test_music_runtime PRIVATE ezlocalai_native)
target_compile_definitions(test_music_runtime PRIVATE TEST_NATIVE_FIXTURE_BIN="$<TARGET_FILE:test_native_fixture>")
add_dependencies(test_music_runtime test_native_fixture)
add_test(NAME music_runtime_reference COMMAND test_music_runtime ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/music_runtime_cases.tsv ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/music_precache_cases.json)
set_tests_properties(music_runtime_reference PROPERTIES TIMEOUT 90)

add_executable(test_music_deployment tests/test_music_deployment.cpp)
add_executable(test_music_video tests/test_music_video.cpp)
target_link_libraries(test_music_video PRIVATE ezlocalai_native)
add_test(NAME music_video_reference COMMAND test_music_video ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/music_video_cases.json)
set_tests_properties(music_video_reference PROPERTIES TIMEOUT 90)
set(EZLOCALAI_STORYBOARD_DEPS "" CACHE PATH "Pinned FreeType/Raqm headers and native libraries for storyboards")
if(EZLOCALAI_STORYBOARD_DEPS)
    if(EXISTS "${EZLOCALAI_STORYBOARD_DEPS}/include/freetype2/ft2build.h")
        set(FREETYPE_INCLUDE_DIRS "${EZLOCALAI_STORYBOARD_DEPS}/include/freetype2")
    else()
        find_package(Freetype REQUIRED)
    endif()
    find_package(ICU REQUIRED COMPONENTS i18n)
    file(GLOB STORYBOARD_FREETYPE "${EZLOCALAI_STORYBOARD_DEPS}/lib/libfreetype-*.so.*")
    list(LENGTH STORYBOARD_FREETYPE STORYBOARD_FREETYPE_COUNT)
    if(NOT STORYBOARD_FREETYPE_COUNT EQUAL 1 OR NOT EXISTS "${EZLOCALAI_STORYBOARD_DEPS}/lib/libraqm.so.0")
        message(FATAL_ERROR "Storyboard dependencies require one pinned FreeType library and libraqm.so.0")
    endif()
    target_sources(ezlocalai_native PRIVATE src/pillow_canvas.cpp src/pillow_drawing.cpp src/music_storyboard.cpp)
    target_include_directories(ezlocalai_native PRIVATE ${FREETYPE_INCLUDE_DIRS} "${EZLOCALAI_STORYBOARD_DEPS}/include")
    target_link_libraries(ezlocalai_native PUBLIC ICU::i18n ${STORYBOARD_FREETYPE} "${EZLOCALAI_STORYBOARD_DEPS}/lib/libraqm.so.0")
    target_compile_definitions(ezlocalai_native PUBLIC EZLOCALAI_STORYBOARD=1)
    add_executable(test_storyboard tests/test_storyboard.cpp)
    target_link_libraries(test_storyboard PRIVATE ezlocalai_native nlohmann_json::nlohmann_json OpenSSL::Crypto)
    add_test(NAME storyboard_raster_reference COMMAND test_storyboard ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/storyboard_raster_cases.json)
    add_test(NAME storyboard_composition_reference COMMAND test_storyboard ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/storyboard_composition_cases.json)
    set_tests_properties(storyboard_raster_reference storyboard_composition_reference PROPERTIES TIMEOUT 120)
endif()
add_executable(test_ltx_policy tests/test_ltx_policy.cpp)
add_executable(test_wakeword_paths tests/test_wakeword_paths.cpp)
add_executable(test_wakeword_cache tests/test_wakeword_cache.cpp)
add_executable(test_wakeword_samples tests/test_wakeword_samples.cpp)
add_executable(test_wakeword_tts tests/test_wakeword_tts.cpp)
add_executable(test_wakeword_qwen tests/test_wakeword_qwen.cpp)
add_executable(test_wakeword_jobs tests/test_wakeword_jobs.cpp)
target_link_libraries(test_wakeword_jobs PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME wakeword_jobs COMMAND test_wakeword_jobs)
set_tests_properties(wakeword_jobs PROPERTIES TIMEOUT 60)
target_link_libraries(test_wakeword_qwen PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME wakeword_qwen COMMAND test_wakeword_qwen)
set_tests_properties(wakeword_qwen PROPERTIES TIMEOUT 60)
add_executable(test_edge_tts tests/test_edge_tts.cpp)
add_executable(test_edge_tts_network tests/test_edge_tts_network.cpp)
target_link_libraries(test_edge_tts_network PRIVATE ezlocalai_native)
add_test(NAME edge_tts_network COMMAND test_edge_tts_network)
set_tests_properties(edge_tts_network PROPERTIES TIMEOUT 60)
target_link_libraries(test_edge_tts PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_link_libraries(test_wakeword_tts PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_link_libraries(test_wakeword_samples PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
set(EZLOCALAI_WAKEWORD_REFERENCE_DIR "" CACHE PATH "Python-generated wake-word checkpoint and audio evidence")
if(EXISTS "${EZLOCALAI_WAKEWORD_REFERENCE_DIR}/edge/reference.json")
    add_test(NAME edge_tts_reference COMMAND test_edge_tts ${EZLOCALAI_WAKEWORD_REFERENCE_DIR})
    set_tests_properties(edge_tts_reference PROPERTIES TIMEOUT 120)
endif()
if(EXISTS "${EZLOCALAI_WAKEWORD_REFERENCE_DIR}/tts/reference.json")
    add_test(NAME wakeword_tts_reference COMMAND test_wakeword_tts ${EZLOCALAI_WAKEWORD_REFERENCE_DIR})
    set_tests_properties(wakeword_tts_reference PROPERTIES TIMEOUT 120)
endif()
if(EXISTS "${EZLOCALAI_WAKEWORD_REFERENCE_DIR}/samples/reference.json")
    add_test(NAME wakeword_samples_reference COMMAND test_wakeword_samples ${EZLOCALAI_WAKEWORD_REFERENCE_DIR})
    set_tests_properties(wakeword_samples_reference PROPERTIES TIMEOUT 60)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/wakeword_samples.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
target_link_libraries(test_wakeword_cache PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME wakeword_cache_reference COMMAND test_wakeword_cache ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/wakeword_cache_cases.json)
add_executable(test_torch_checkpoint tests/test_torch_checkpoint.cpp src/torch_checkpoint.cpp)
target_include_directories(test_torch_checkpoint PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_torch_checkpoint PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME torch_checkpoint_reference COMMAND test_torch_checkpoint ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/torch_checkpoint_cases.json)
target_link_libraries(test_wakeword_paths PRIVATE ezlocalai_native)
add_test(NAME wakeword_paths_reference COMMAND test_wakeword_paths
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/wakeword_paths_cases.json ${EZLOCALAI_REFERENCE_ROOT})
target_link_libraries(test_ltx_policy PRIVATE ezlocalai_native)
add_test(NAME ltx_policy_reference COMMAND test_ltx_policy ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ltx_policy_cases.json)

add_executable(test_nf4 tests/test_nf4.cpp)
target_include_directories(test_nf4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_nf4 PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME nf4_reference COMMAND test_nf4 ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nf4_cases.json)
add_test(NAME nf4_f16_reference COMMAND test_nf4 ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nf4_f16_cases.json)
add_executable(test_nf4_mean tests/test_nf4_mean.cpp)
target_include_directories(test_nf4_mean PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(test_nf4_mean PRIVATE nlohmann_json::nlohmann_json ZLIB::ZLIB)
add_test(NAME nf4_cuda_mean_reference COMMAND test_nf4_mean ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nf4_mean_cases.json)

set(EZLOCALAI_SDCPP_SOURCE "" CACHE PATH "Pinned stable-diffusion.cpp source with patches/sdcpp-ltx-conditions.patch applied")
set(EZLOCALAI_SDCPP_BUILD "" CACHE PATH "Optional patched stable-diffusion.cpp build for GGML-backed LTX tests")
if(EZLOCALAI_SDCPP_SOURCE)
    add_executable(test_ltx_conditions tests/test_ltx_conditions.cpp ${EZLOCALAI_SDCPP_SOURCE}/src/core/parallel.cpp)
    target_include_directories(test_ltx_conditions PRIVATE ${EZLOCALAI_SDCPP_SOURCE}/src ${EZLOCALAI_SDCPP_SOURCE}/include)
    target_link_libraries(test_ltx_conditions PRIVATE Threads::Threads)
    add_test(NAME ltx_conditions_reference COMMAND test_ltx_conditions ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ltx_condition_cases.json)
    add_executable(test_ltx_rng tests/test_ltx_rng.cpp)
    target_include_directories(test_ltx_rng PRIVATE ${EZLOCALAI_SDCPP_SOURCE}/src)
    target_link_libraries(test_ltx_rng PRIVATE nlohmann_json::nlohmann_json)
    add_test(NAME ltx_rng_reference COMMAND test_ltx_rng ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ltx_rng_cases.json)
    if(EXISTS "${EZLOCALAI_SDCPP_SOURCE}/src/ltx_diffusers.hpp")
        if(EXISTS "${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-cpu.a")
            add_executable(test_ltx_name_concurrency tests/test_ltx_name_concurrency.cpp
                ${EZLOCALAI_SDCPP_SOURCE}/src/name_conversion.cpp)
            target_include_directories(test_ltx_name_concurrency PRIVATE
                ${EZLOCALAI_SDCPP_SOURCE}/src ${EZLOCALAI_SDCPP_SOURCE}/include
                ${EZLOCALAI_SDCPP_SOURCE}/thirdparty ${EZLOCALAI_SDCPP_SOURCE}/ggml/include)
            target_compile_options(test_ltx_name_concurrency PRIVATE -ffunction-sections -fdata-sections)
            target_link_options(test_ltx_name_concurrency PRIVATE -Wl,--gc-sections)
            target_link_libraries(test_ltx_name_concurrency PRIVATE Threads::Threads)
            add_test(NAME ltx_name_concurrency COMMAND test_ltx_name_concurrency)
            set_tests_properties(ltx_name_concurrency PROPERTIES TIMEOUT 30)
            find_package(OpenMP REQUIRED)
            add_executable(test_ltx_schedule tests/test_ltx_schedule.cpp
                ${EZLOCALAI_SDCPP_SOURCE}/src/core/util.cpp)
            target_include_directories(test_ltx_schedule PRIVATE include
                ${EZLOCALAI_SDCPP_SOURCE}/src ${EZLOCALAI_SDCPP_SOURCE}/include
                ${EZLOCALAI_SDCPP_SOURCE}/thirdparty ${EZLOCALAI_SDCPP_SOURCE}/ggml/include)
            target_compile_options(test_ltx_schedule PRIVATE -ffunction-sections -fdata-sections)
            target_compile_definitions(test_ltx_schedule PRIVATE GGML_MAX_NAME=160)
            target_link_options(test_ltx_schedule PRIVATE -Wl,--gc-sections)
            target_link_libraries(test_ltx_schedule PRIVATE nlohmann_json::nlohmann_json
                ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-base.a Threads::Threads ${CMAKE_DL_LIBS})
            add_test(NAME ltx_schedule_reference COMMAND test_ltx_schedule ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ltx_schedule_cases.json)
            add_executable(test_ltx_norm tests/test_ltx_norm.cpp)
            target_include_directories(test_ltx_norm PRIVATE
                ${EZLOCALAI_SDCPP_SOURCE}/src ${EZLOCALAI_SDCPP_SOURCE}/include
                ${EZLOCALAI_SDCPP_SOURCE}/thirdparty ${EZLOCALAI_SDCPP_SOURCE}/ggml/include
                ${EZLOCALAI_SDCPP_SOURCE}/ggml/src)
            target_compile_definitions(test_ltx_norm PRIVATE GGML_MAX_NAME=160)
            target_compile_options(test_ltx_norm PRIVATE -ffunction-sections -fdata-sections)
            target_link_options(test_ltx_norm PRIVATE -Wl,--gc-sections)
            target_link_libraries(test_ltx_norm PRIVATE nlohmann_json::nlohmann_json
                "-Wl,--start-group" ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-cpu.a
                ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-base.a "-Wl,--end-group"
                OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
            add_test(NAME ltx_norm_reference COMMAND test_ltx_norm ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ltx_gemma_norm_cases.json)
            set(nf4_storage_targets test_nf4_storage)
            if(EXISTS "${EZLOCALAI_SDCPP_BUILD}/ggml/src/ggml-cuda/libggml-cuda.a")
                list(APPEND nf4_storage_targets test_nf4_storage_cuda)
            endif()
            foreach(nf4_target IN LISTS nf4_storage_targets)
                add_executable(${nf4_target} tests/test_nf4_storage.cpp)
                target_include_directories(${nf4_target} PRIVATE include engine ${EZLOCALAI_SDCPP_SOURCE}/ggml/include)
                target_compile_definitions(${nf4_target} PRIVATE GGML_MAX_NAME=160)
                target_link_libraries(${nf4_target} PRIVATE nlohmann_json::nlohmann_json
                    "-Wl,--start-group" ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-cpu.a
                    ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-base.a "-Wl,--end-group"
                    OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
            endforeach()
            if(TARGET test_nf4_storage_cuda)
                find_package(CUDAToolkit REQUIRED)
                target_compile_definitions(test_nf4_storage_cuda PRIVATE EZLOCALAI_NF4_CUDA=1)
                target_link_libraries(test_nf4_storage_cuda PRIVATE
                    ${EZLOCALAI_SDCPP_BUILD}/ggml/src/ggml-cuda/libggml-cuda.a
                    ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-base.a
                    CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)
                if(EXISTS "${EZLOCALAI_SDCPP_BUILD}/libstable-diffusion.a")
                    foreach(ltx_engine_test IN ITEMS test_ltx_gemma_residency_cuda test_ltx_connectors)
                        if(ltx_engine_test STREQUAL "test_ltx_gemma_residency_cuda")
                            set(ltx_engine_test_source tests/test_ltx_gemma_residency.cpp)
                        else()
                            set(ltx_engine_test_source tests/test_ltx_connectors.cpp)
                        endif()
                        add_executable(${ltx_engine_test} ${ltx_engine_test_source}
                            ${EZLOCALAI_SDCPP_BUILD}/thirdparty/CMakeFiles/zip.dir/zip.c.o)
                        target_include_directories(${ltx_engine_test} PRIVATE include
                            ${EZLOCALAI_SDCPP_SOURCE}/src ${EZLOCALAI_SDCPP_SOURCE}/include
                            ${EZLOCALAI_SDCPP_SOURCE}/thirdparty ${EZLOCALAI_SDCPP_SOURCE}/ggml/include)
                        target_compile_definitions(${ltx_engine_test} PRIVATE GGML_MAX_NAME=160)
                        target_compile_options(${ltx_engine_test} PRIVATE -ffunction-sections -fdata-sections)
                        target_link_options(${ltx_engine_test} PRIVATE -Wl,--gc-sections)
                        target_link_libraries(${ltx_engine_test} PRIVATE
                            "-Wl,--start-group" ${EZLOCALAI_SDCPP_BUILD}/libstable-diffusion.a
                            ${EZLOCALAI_SDCPP_BUILD}/thirdparty/libwebp/libwebp.a
                            ${EZLOCALAI_SDCPP_BUILD}/thirdparty/libwebp/libwebpmux.a
                            ${EZLOCALAI_SDCPP_BUILD}/thirdparty/libwebp/libsharpyuv.a
                            ${EZLOCALAI_SDCPP_BUILD}/thirdparty/libwebm/libwebm.a
                            ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml.a
                            ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-cpu.a
                            ${EZLOCALAI_SDCPP_BUILD}/ggml/src/ggml-cuda/libggml-cuda.a
                            ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-base.a "-Wl,--end-group"
                            CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver
                            OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
                    endforeach()
                    add_test(NAME ltx_connectors COMMAND test_ltx_connectors)
                endif()
            endif()
            add_test(NAME nf4_storage_reference COMMAND test_nf4_storage
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nf4_cases.json
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nf4_f16_cases.json
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nf4_storage_cases.json)
            if(EXISTS "${EZLOCALAI_SDCPP_SOURCE}/src/ezlocalai/nf4.hpp")
                enable_language(C)
                add_executable(test_ltx_nf4_loader tests/test_ltx_nf4_loader.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/model_loader.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/model_loader_files.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/name_conversion.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/core/util.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/core/ggml_extend.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/core/ggml_extend_backend.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/runtime/imatrix.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/model_io/gguf_io.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/model_io/safetensors_io.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/model_io/torch_legacy_io.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/model_io/torch_zip_io.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/src/model_io/pickle_io.cpp
                    ${EZLOCALAI_SDCPP_SOURCE}/thirdparty/zip.c)
                target_include_directories(test_ltx_nf4_loader PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/include ${EZLOCALAI_SDCPP_SOURCE}/src
                    ${EZLOCALAI_SDCPP_SOURCE}/include ${EZLOCALAI_SDCPP_SOURCE}/thirdparty
                    ${EZLOCALAI_SDCPP_SOURCE}/ggml/include ${EZLOCALAI_SDCPP_SOURCE}/ggml/src)
                target_compile_definitions(test_ltx_nf4_loader PRIVATE GGML_MAX_NAME=160)
                target_compile_options(test_ltx_nf4_loader PRIVATE -ffunction-sections -fdata-sections)
                target_link_options(test_ltx_nf4_loader PRIVATE -Wl,--gc-sections)
                target_link_libraries(test_ltx_nf4_loader PRIVATE
                    "-Wl,--start-group" ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-cpu.a
                    ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-base.a "-Wl,--end-group"
                    OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
                add_test(NAME ltx_nf4_loader_reference COMMAND test_ltx_nf4_loader
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nf4_cases.json
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/nf4_f16_cases.json)
            endif()
            if(EXISTS "${EZLOCALAI_SDCPP_SOURCE}/src/ltx_timesteps.hpp")
                add_executable(test_ltx_timesteps tests/test_ltx_timesteps.cpp)
                target_include_directories(test_ltx_timesteps PRIVATE
                    ${EZLOCALAI_SDCPP_SOURCE}/src ${EZLOCALAI_SDCPP_SOURCE}/ggml/include)
                target_compile_definitions(test_ltx_timesteps PRIVATE GGML_MAX_NAME=160)
                target_link_libraries(test_ltx_timesteps PRIVATE nlohmann_json::nlohmann_json
                    "-Wl,--start-group" ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-cpu.a
                    ${EZLOCALAI_SDCPP_BUILD}/ggml/src/libggml-base.a "-Wl,--end-group"
                    OpenMP::OpenMP_CXX Threads::Threads ${CMAKE_DL_LIBS})
                add_test(NAME ltx_timesteps_reference COMMAND test_ltx_timesteps ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ltx_timestep_cases.json)
            endif()
        endif()
        add_executable(test_ltx_audio tests/test_ltx_audio.cpp)
        target_include_directories(test_ltx_audio PRIVATE ${EZLOCALAI_SDCPP_SOURCE}/src)
        target_link_libraries(test_ltx_audio PRIVATE nlohmann_json::nlohmann_json)
        add_test(NAME ltx_audio_reference COMMAND test_ltx_audio ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ltx_audio_length_cases.json)
        add_executable(test_ltx_mapping tests/test_ltx_mapping.cpp)
        target_include_directories(test_ltx_mapping PRIVATE ${EZLOCALAI_SDCPP_SOURCE}/src)
        add_test(NAME ltx_mapping_reference COMMAND test_ltx_mapping ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ltx_mapping_cases.json)
    endif()
endif()
target_link_libraries(test_music_deployment PRIVATE ezlocalai_native)
set(EZLOCALAI_ACE_STEP_MODELS "" CACHE PATH "Cached ACE-Step models for opt-in real music inference")
set(EZLOCALAI_ACE_STEP_BIN "" CACHE FILEPATH "Native ace-server for opt-in real music inference")
option(EZLOCALAI_ACE_STEP_GPU_TESTS "Run hardware-specific ACE-Step CUDA model fixtures" OFF)
if(EZLOCALAI_ACE_STEP_MODELS AND EZLOCALAI_ACE_STEP_BIN)
    add_test(NAME music_deployment_cpu_reference COMMAND test_music_deployment
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/music_deployment_cpu.json
        ${EZLOCALAI_ACE_STEP_MODELS} ${EZLOCALAI_ACE_STEP_BIN})
    set_tests_properties(music_deployment_cpu_reference PROPERTIES TIMEOUT 1800 ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
    if(EZLOCALAI_ACE_STEP_GPU_TESTS)
        add_test(NAME music_deployment_cuda_reference COMMAND test_music_deployment
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/music_deployment_cuda.json
            ${EZLOCALAI_ACE_STEP_MODELS} ${EZLOCALAI_ACE_STEP_BIN})
        set_tests_properties(music_deployment_cuda_reference PROPERTIES TIMEOUT 600 RESOURCE_LOCK stt_torch_gpu)
    endif()
endif()

add_executable(test_ltx_condition_cache tests/test_ltx_condition_cache.cpp)
target_include_directories(test_ltx_condition_cache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine)
add_test(NAME ltx_condition_cache COMMAND test_ltx_condition_cache)

add_executable(test_ltx_fixture tests/ltx_process_fixture.cpp)
target_link_libraries(test_ltx_fixture PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(test_ltx_fixture PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/engine)
add_executable(test_ltx_runtime tests/test_ltx_runtime.cpp)
target_link_libraries(test_ltx_runtime PRIVATE ezlocalai_native)
target_compile_definitions(test_ltx_runtime PRIVATE TEST_LTX_FIXTURE_BIN="$<TARGET_FILE:test_ltx_fixture>" TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_cuda_driver>")
add_dependencies(test_ltx_runtime test_ltx_fixture test_cuda_driver)
add_test(NAME ltx_runtime COMMAND test_ltx_runtime ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ltx_i2v_resize_cases.json)
set_tests_properties(ltx_runtime PROPERTIES TIMEOUT 120)

add_executable(test_runtime_version tests/test_runtime_version.cpp)
target_link_libraries(test_runtime_version PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME runtime_version_reference COMMAND test_runtime_version ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/worker_version_cases.json)
set_tests_properties(runtime_version_reference PROPERTIES TIMEOUT 30)

add_executable(test_wan_fixture tests/wan_process_fixture.cpp)
add_executable(test_wan tests/test_wan.cpp)
add_executable(test_qwen_image tests/test_qwen_image.cpp)
target_link_libraries(test_qwen_image PRIVATE ezlocalai_native)
target_compile_definitions(test_qwen_image PRIVATE TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_cuda_driver>")
add_dependencies(test_qwen_image test_cuda_driver)
add_test(NAME qwen_image_reference COMMAND test_qwen_image ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/qwen_image_cases.json)
target_link_libraries(test_wan PRIVATE ezlocalai_native)
target_compile_definitions(test_wan PRIVATE TEST_WAN_FIXTURE_BIN="$<TARGET_FILE:test_wan_fixture>")
add_dependencies(test_wan test_wan_fixture)
add_test(NAME wan_reference COMMAND test_wan ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/wan_cases.tsv)
set_tests_properties(wan_reference PROPERTIES TIMEOUT 90)

add_executable(test_image_processing tests/test_image_processing.cpp)
target_link_libraries(test_image_processing PRIVATE ezlocalai_native)
add_test(NAME image_processing_reference COMMAND test_image_processing ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/image_cases.tsv ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/video_image_cases.tsv)
set_tests_properties(image_processing_reference PROPERTIES TIMEOUT 90)

add_executable(test_video_api tests/test_video_api.cpp)
target_link_libraries(test_video_api PRIVATE ezlocalai_native)
add_test(NAME video_api_reference COMMAND test_video_api ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/video_api_cases.tsv)
set_tests_properties(video_api_reference PROPERTIES TIMEOUT 90)

add_library(test_cuda_driver SHARED tests/cuda_driver_fixture.cpp)
target_compile_definitions(test_worker_metadata PRIVATE TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_cuda_driver>")
add_dependencies(test_worker_metadata test_cuda_driver)
set_target_properties(test_cuda_driver PROPERTIES OUTPUT_NAME cuda SOVERSION 1 LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cuda-fixture)
add_library(test_hip_driver SHARED tests/cuda_driver_fixture.cpp)
target_compile_definitions(test_hip_driver PRIVATE EZLOCALAI_TEST_HIP=1)
set_target_properties(test_hip_driver PROPERTIES OUTPUT_NAME amdhip64 LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hip-fixture)
add_executable(test_hip_runtime tests/test_hip_runtime.cpp src/video_runtime.cpp)
target_compile_definitions(test_hip_runtime PRIVATE EZLOCALAI_USE_HIP=1)
target_link_libraries(test_hip_runtime PRIVATE ezlocalai_native)
add_dependencies(test_hip_runtime test_hip_driver)
add_test(NAME hip_device_runtime COMMAND test_hip_runtime)
set_tests_properties(hip_device_runtime PROPERTIES TIMEOUT 30 ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:test_hip_driver>")
add_executable(test_worker_metadata_hip tests/test_worker_metadata.cpp src/video_runtime.cpp)
target_link_libraries(test_worker_metadata_hip PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_worker_metadata_hip PRIVATE EZLOCALAI_USE_HIP=1
    TEST_HIP_GPU_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/worker_hip_gpu_cases.json"
    TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_hip_driver>")
add_dependencies(test_worker_metadata_hip test_hip_driver)
add_test(NAME worker_metadata_hip_reference COMMAND test_worker_metadata_hip
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/worker_extra_cases.json
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/worker_model_reporting_cases.json)
set_tests_properties(worker_metadata_hip_reference PROPERTIES TIMEOUT 90)
if(EZLOCALAI_USE_HIP)
    add_executable(test_hip_abi tests/test_hip_abi.cpp)
    target_include_directories(test_hip_abi PRIVATE ${EZLOCALAI_HIP_INCLUDE_DIR})
    target_compile_definitions(test_hip_abi PRIVATE __HIP_PLATFORM_AMD__=1)
    target_link_libraries(test_hip_abi PRIVATE ${CMAKE_DL_LIBS})
    add_test(NAME hip_runtime_abi COMMAND test_hip_abi)
endif()
add_executable(test_video_runtime tests/test_video_runtime.cpp)
add_executable(test_video_auxiliary tests/test_video_auxiliary.cpp)
target_link_libraries(test_video_auxiliary PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_video_auxiliary PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_video_auxiliary test_embedding_fixture)
add_test(NAME video_auxiliary COMMAND test_video_auxiliary)
set_tests_properties(video_auxiliary PROPERTIES TIMEOUT 90)
add_executable(test_media_slots tests/test_media_slots.cpp)
target_link_libraries(test_media_slots PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_media_slots PRIVATE TEST_WAN_FIXTURE_BIN="$<TARGET_FILE:test_wan_fixture>")
add_dependencies(test_media_slots test_wan_fixture)
add_test(NAME media_slots_reference COMMAND test_media_slots ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/media_slot_cases.json)
set_tests_properties(media_slots_reference PROPERTIES TIMEOUT 90)
target_link_libraries(test_video_runtime PRIVATE ezlocalai_native)
target_compile_definitions(test_video_runtime PRIVATE TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_cuda_driver>" TEST_WAN_FIXTURE_BIN="$<TARGET_FILE:test_wan_fixture>")
add_dependencies(test_video_runtime test_cuda_driver test_wan_fixture)
add_test(NAME video_runtime_reference COMMAND test_video_runtime ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/video_runtime_cases.tsv ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/video_auxiliary_cases.tsv)
set_tests_properties(video_runtime_reference PROPERTIES TIMEOUT 90)

add_executable(test_embedding_fixture tests/embedding_process_fixture.cpp)
target_link_libraries(test_embedding_fixture PRIVATE Threads::Threads)
add_executable(test_embedding_runtime tests/test_embedding_runtime.cpp)
target_link_libraries(test_embedding_runtime PRIVATE ezlocalai_native)
target_compile_definitions(test_embedding_runtime PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>" TEST_NATIVE_SERVER_BIN="$<TARGET_FILE:ezlocalai-native>")
add_dependencies(test_embedding_runtime test_embedding_fixture ezlocalai-native)
add_test(NAME embedding_runtime_reference COMMAND test_embedding_runtime ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/embedding_pool_cases.tsv)
set_tests_properties(embedding_runtime_reference PROPERTIES TIMEOUT 90)

add_executable(test_embedding_placement tests/test_embedding_placement.cpp)
target_link_libraries(test_embedding_placement PRIVATE ezlocalai_native)
target_compile_definitions(test_embedding_placement PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>" TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_cuda_driver>")
add_dependencies(test_embedding_placement test_embedding_fixture test_cuda_driver)
add_test(NAME embedding_placement_reference COMMAND test_embedding_placement ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/embedding_placement_cases.tsv)
set_tests_properties(embedding_placement_reference PROPERTIES TIMEOUT 90)

add_executable(test_model_download tests/test_model_download.cpp)
target_link_libraries(test_model_download PRIVATE ezlocalai_native)
target_compile_definitions(test_model_download PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>" TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_cuda_driver>")
add_dependencies(test_model_download test_embedding_fixture test_cuda_driver)
add_test(NAME model_download_reference COMMAND test_model_download ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/model_download_cases.tsv ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/hub_local_cases.tsv)
set_tests_properties(model_download_reference PROPERTIES TIMEOUT 90)

add_executable(test_resource_manager tests/test_resource_manager.cpp)
add_executable(test_cuda_allocator tests/test_cuda_allocator.cpp)
target_link_libraries(test_cuda_allocator PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME cuda_allocator_reference COMMAND test_cuda_allocator ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/cuda_allocator_cases.json)
set_tests_properties(cuda_allocator_reference PROPERTIES TIMEOUT 30)
add_executable(test_speech_resources tests/test_speech_resources.cpp)
target_link_libraries(test_speech_resources PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME speech_resources_reference COMMAND test_speech_resources ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/speech_resource_cases.json)
set_tests_properties(speech_resources_reference PROPERTIES TIMEOUT 60)
add_executable(test_speech_slots tests/test_speech_slots.cpp)
target_link_libraries(test_speech_slots PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME speech_slots_reference COMMAND test_speech_slots ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/speech_slot_cases.json)
set_tests_properties(speech_slots_reference PROPERTIES TIMEOUT 60)
target_link_libraries(test_resource_manager PRIVATE ezlocalai_native)
target_compile_definitions(test_resource_manager PRIVATE TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_cuda_driver>")
add_dependencies(test_resource_manager test_cuda_driver)
add_test(NAME resource_manager_reference COMMAND test_resource_manager ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/resource_cases.tsv)
set_tests_properties(resource_manager_reference PROPERTIES TIMEOUT 30)

add_executable(test_fallback tests/test_fallback.cpp)
target_link_libraries(test_fallback PRIVATE ezlocalai_native)
add_test(NAME fallback_reference COMMAND test_fallback ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/fallback_cases.tsv)
set_tests_properties(fallback_reference PROPERTIES TIMEOUT 90)

add_executable(test_http_deadline tests/test_http_deadline.cpp)
target_link_libraries(test_http_deadline PRIVATE ezlocalai_native)
add_test(NAME http_deadline_reference COMMAND test_http_deadline ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(http_deadline_reference PROPERTIES TIMEOUT 60)

add_executable(test_http_compression tests/test_http_compression.cpp)
target_link_libraries(test_http_compression PRIVATE ezlocalai_native)
add_test(NAME http_compression_reference COMMAND test_http_compression ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(http_compression_reference PROPERTIES TIMEOUT 60)

add_executable(test_http_redirect tests/test_http_redirect.cpp)
target_link_libraries(test_http_redirect PRIVATE ezlocalai_native)
add_test(NAME http_redirect_reference COMMAND test_http_redirect ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(http_redirect_reference PROPERTIES TIMEOUT 60)

add_executable(test_http_cookies tests/test_http_cookies.cpp)
target_link_libraries(test_http_cookies PRIVATE ezlocalai_native)
add_test(NAME http_cookies_reference COMMAND test_http_cookies ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/http_cookie_cases.tsv)
set_tests_properties(http_cookies_reference PROPERTIES TIMEOUT 60)

add_executable(test_llm_fallback tests/test_llm_fallback.cpp)
target_link_libraries(test_llm_fallback PRIVATE ezlocalai_native)
add_test(NAME llm_fallback_reference COMMAND test_llm_fallback ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(llm_fallback_reference PROPERTIES TIMEOUT 90)

add_executable(test_model_lifecycle tests/test_model_lifecycle.cpp)
target_link_libraries(test_model_lifecycle PRIVATE ezlocalai_native)
target_compile_definitions(test_model_lifecycle PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_model_lifecycle test_embedding_fixture)
add_test(NAME model_lifecycle_reference COMMAND test_model_lifecycle ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/model_lifecycle_cases.tsv)
set_tests_properties(model_lifecycle_reference PROPERTIES TIMEOUT 45)

add_executable(test_llm_slots tests/test_llm_slots.cpp)
add_executable(test_llm_residency tests/test_llm_residency.cpp)
add_executable(test_llm_headroom tests/test_llm_headroom.cpp)
add_executable(test_llm_preload tests/test_llm_preload.cpp)
add_executable(test_live_capacity tests/test_live_capacity.cpp)
add_executable(test_llm_instances tests/test_llm_instances.cpp)
add_executable(test_media_engine tests/test_media_engine.cpp)
target_link_libraries(test_media_engine PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_include_directories(test_media_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/engine)
add_test(NAME media_engine COMMAND test_media_engine)
set_tests_properties(media_engine PROPERTIES TIMEOUT 30)

add_executable(test_media_startup tests/test_media_startup.cpp)
target_link_libraries(test_media_startup PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_dependencies(test_media_startup test_cuda_driver)
add_test(NAME media_startup_reference COMMAND test_media_startup ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/media_startup_cases.json)
set_tests_properties(media_startup_reference PROPERTIES TIMEOUT 90 ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:test_cuda_driver>")

add_executable(test_startup tests/test_startup.cpp)
target_link_libraries(test_startup PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_startup PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_startup test_embedding_fixture ezlocalai-native)
add_test(NAME startup_reference COMMAND test_startup ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/startup_cases.json $<TARGET_FILE:ezlocalai-native>)
set_tests_properties(startup_reference PROPERTIES TIMEOUT 90)
target_link_libraries(test_llm_instances PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_llm_instances PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_instances test_embedding_fixture)
add_test(NAME llm_instances_reference COMMAND test_llm_instances ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_instance_cases.json)
set_tests_properties(llm_instances_reference PROPERTIES TIMEOUT 60)
target_link_libraries(test_live_capacity PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_live_capacity PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_live_capacity test_embedding_fixture)
add_test(NAME live_capacity_reference COMMAND test_live_capacity ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/live_capacity_cases.json)
set_tests_properties(live_capacity_reference PROPERTIES TIMEOUT 90)
target_link_libraries(test_llm_preload PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_llm_preload PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_preload test_embedding_fixture)
add_test(NAME llm_preload_reference COMMAND test_llm_preload ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_preload_cases.json)
set_tests_properties(llm_preload_reference PROPERTIES TIMEOUT 90)
target_link_libraries(test_llm_headroom PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_llm_headroom PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_headroom test_embedding_fixture)
add_test(NAME llm_headroom_reference COMMAND test_llm_headroom ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_headroom_cases.json)
set_tests_properties(llm_headroom_reference PROPERTIES TIMEOUT 90)
target_link_libraries(test_llm_residency PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_llm_residency PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_residency test_embedding_fixture)
add_test(NAME llm_residency_reference COMMAND test_llm_residency ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_residency_cases.json)
set_tests_properties(llm_residency_reference PROPERTIES TIMEOUT 60)
target_link_libraries(test_llm_slots PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_slots PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_slots test_embedding_fixture)
add_test(NAME llm_slots_reference COMMAND test_llm_slots ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_slots_cases.tsv)
set_tests_properties(llm_slots_reference PROPERTIES TIMEOUT 45)

add_executable(test_llm_catalog tests/test_llm_catalog.cpp)
target_link_libraries(test_llm_catalog PRIVATE ezlocalai_native)
add_test(NAME llm_catalog_reference COMMAND test_llm_catalog ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_catalog_cases.tsv)
set_tests_properties(llm_catalog_reference PROPERTIES TIMEOUT 45)

add_executable(test_llm_startup tests/test_llm_startup.cpp)
target_link_libraries(test_llm_startup PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_startup PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_startup test_embedding_fixture)
add_test(NAME llm_startup_reference COMMAND test_llm_startup ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_startup_cases.tsv)
set_tests_properties(llm_startup_reference PROPERTIES TIMEOUT 45)

add_executable(test_llm_memory tests/test_llm_memory.cpp)
target_link_libraries(test_llm_memory PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_memory PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_memory test_embedding_fixture)
add_test(NAME llm_memory_reference COMMAND test_llm_memory ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_memory_cases.jsonl)
set_tests_properties(llm_memory_reference PROPERTIES TIMEOUT 90)

add_executable(test_llm_load tests/test_llm_load.cpp)
target_link_libraries(test_llm_load PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_load PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_load test_embedding_fixture)
add_test(NAME llm_load_reference COMMAND test_llm_load ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_load_cases.tsv)
set_tests_properties(llm_load_reference PROPERTIES TIMEOUT 90)

add_executable(test_llm_batch tests/test_llm_batch.cpp)
target_link_libraries(test_llm_batch PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_batch PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>" TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_cuda_driver>")
add_dependencies(test_llm_batch test_embedding_fixture test_cuda_driver)
add_test(NAME llm_batch_reference COMMAND test_llm_batch ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(llm_batch_reference PROPERTIES TIMEOUT 45)

add_executable(test_llm_cache tests/test_llm_cache.cpp)
target_link_libraries(test_llm_cache PRIVATE ezlocalai_native)
set(EZLOCALAI_CACHE_TEST_DRIVER test_cuda_driver)
if(EZLOCALAI_USE_HIP)
    set(EZLOCALAI_CACHE_TEST_DRIVER test_hip_driver)
endif()
target_compile_definitions(test_llm_cache PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>" TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:${EZLOCALAI_CACHE_TEST_DRIVER}>" TEST_NATIVE_SOURCE_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")
add_dependencies(test_llm_cache test_embedding_fixture ${EZLOCALAI_CACHE_TEST_DRIVER})
add_test(NAME llm_cache_reference COMMAND test_llm_cache ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(llm_cache_reference PROPERTIES TIMEOUT 45)
# Images contain the recorded cases, not the external Python checkout. Keep
# the package gate independent while retaining strict source verification above.
add_test(NAME llm_cache_package COMMAND test_llm_cache --fixtures ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(llm_cache_package PROPERTIES TIMEOUT 45)

add_executable(test_llm_speculative tests/test_llm_speculative.cpp)
target_link_libraries(test_llm_speculative PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_speculative PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>" TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_cuda_driver>")
add_dependencies(test_llm_speculative test_embedding_fixture test_cuda_driver)
add_test(NAME llm_speculative_reference COMMAND test_llm_speculative ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(llm_speculative_reference PROPERTIES TIMEOUT 60)
add_executable(test_llm_speculative_hip tests/test_llm_speculative.cpp src/video_runtime.cpp)
target_link_libraries(test_llm_speculative_hip PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_speculative_hip PRIVATE EZLOCALAI_USE_HIP=1
    TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>"
    TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_hip_driver>")
add_dependencies(test_llm_speculative_hip test_embedding_fixture test_hip_driver)
add_test(NAME llm_speculative_hip_reference COMMAND test_llm_speculative_hip
    ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(llm_speculative_hip_reference PROPERTIES TIMEOUT 60)
add_test(NAME llm_speculative_hip_package COMMAND test_llm_speculative_hip
    --fixtures ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(llm_speculative_hip_package PROPERTIES TIMEOUT 60)

add_executable(test_llm_parameters tests/test_llm_parameters.cpp)
target_link_libraries(test_llm_parameters PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_parameters PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>" TEST_CUDA_DRIVER_DIRECTORY="$<TARGET_FILE_DIR:test_cuda_driver>" TEST_NATIVE_SOURCE_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}")
add_dependencies(test_llm_parameters test_embedding_fixture test_cuda_driver)
add_test(NAME llm_parameters_reference COMMAND test_llm_parameters ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_parameter_cases.tsv)
set_tests_properties(llm_parameters_reference PROPERTIES TIMEOUT 60)

add_executable(test_llm_post_fallback tests/test_llm_post_fallback.cpp)
target_link_libraries(test_llm_post_fallback PRIVATE ezlocalai_native)
add_test(NAME llm_post_fallback_reference COMMAND test_llm_post_fallback ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_post_fallback_cases.tsv)
set_tests_properties(llm_post_fallback_reference PROPERTIES TIMEOUT 90)

add_executable(test_router_fallback tests/test_router_fallback.cpp)
target_link_libraries(test_router_fallback PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME router_fallback COMMAND test_router_fallback)
set_tests_properties(router_fallback PROPERTIES TIMEOUT 120)

add_executable(test_llm_response tests/test_llm_response.cpp)
target_link_libraries(test_llm_response PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_response PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_response test_embedding_fixture)
add_test(NAME llm_response_reference COMMAND test_llm_response ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_response_cases.tsv)
set_tests_properties(llm_response_reference PROPERTIES TIMEOUT 60)

add_executable(test_llm_context tests/test_llm_context.cpp)
target_link_libraries(test_llm_context PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_context PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_context test_embedding_fixture)
add_test(NAME llm_context_reference COMMAND test_llm_context ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_context_cases.tsv)
set_tests_properties(llm_context_reference PROPERTIES TIMEOUT 60)

add_executable(test_llm_output tests/test_llm_output.cpp)
target_link_libraries(test_llm_output PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_output PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_output test_embedding_fixture)
add_test(NAME llm_output_reference COMMAND test_llm_output ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_output_cases.tsv)
set_tests_properties(llm_output_reference PROPERTIES TIMEOUT 90)

add_executable(test_llm_request tests/test_llm_request.cpp)
target_link_libraries(test_llm_request PRIVATE ezlocalai_native)
add_executable(bench_llm_request EXCLUDE_FROM_ALL benchmarks/bench_request.cpp)
target_link_libraries(bench_llm_request PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_executable(bench_llm_input EXCLUDE_FROM_ALL benchmarks/bench_input.cpp)
target_link_libraries(bench_llm_input PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_llm_request PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_request test_embedding_fixture)
add_test(NAME llm_request_reference COMMAND test_llm_request ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_request_cases.tsv)
set_tests_properties(llm_request_reference PROPERTIES TIMEOUT 90)

add_executable(test_llm_input tests/test_llm_input.cpp)
target_link_libraries(test_llm_input PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_input PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_input test_embedding_fixture)
add_test(NAME llm_input_reference COMMAND test_llm_input ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_input_cases.tsv)
set_tests_properties(llm_input_reference PROPERTIES TIMEOUT 60)
add_test(NAME llm_input_package COMMAND test_llm_input --fixtures ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_input_cases.tsv)
set_tests_properties(llm_input_package PROPERTIES TIMEOUT 60)

add_executable(test_llm_audio_fixture tests/llm_audio_process_fixture.cpp)
add_executable(test_llm_audio tests/test_llm_audio.cpp tests/test_llm_audio_runtime.cpp)
target_link_libraries(test_llm_audio PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_audio PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>" TEST_LLM_AUDIO_FIXTURE_BIN="$<TARGET_FILE:test_llm_audio_fixture>")
add_dependencies(test_llm_audio test_embedding_fixture test_llm_audio_fixture)
add_test(NAME llm_audio_reference COMMAND test_llm_audio ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_audio_cases.tsv ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_audio_guard_cases.tsv)
set_tests_properties(llm_audio_reference PROPERTIES TIMEOUT 60)

add_executable(test_llm_video tests/test_llm_video.cpp)
target_link_libraries(test_llm_video PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_video PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>" TEST_LLM_AUDIO_FIXTURE_BIN="$<TARGET_FILE:test_llm_audio_fixture>")
add_dependencies(test_llm_video test_embedding_fixture test_llm_audio_fixture)
add_test(NAME llm_video_reference COMMAND test_llm_video ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_video_cases.tsv ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_video_media_cases.tsv)
set_tests_properties(llm_video_reference PROPERTIES TIMEOUT 90)

add_executable(test_whisper_features tests/test_whisper_features.cpp)
target_link_libraries(test_whisper_features PRIVATE ezlocalai_native)
add_test(NAME whisper_features_reference COMMAND test_whisper_features ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/whisper_feature_cases.tsv)
set_tests_properties(whisper_features_reference PROPERTIES TIMEOUT 60 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")

add_executable(test_whisper_tokenizer tests/test_whisper_tokenizer.cpp)
target_link_libraries(test_whisper_tokenizer PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME whisper_tokenizer_reference COMMAND test_whisper_tokenizer ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/whisper_tokenizer_cases.tsv)
set_tests_properties(whisper_tokenizer_reference PROPERTIES TIMEOUT 120)

add_executable(test_whisper_transcription tests/test_whisper_transcription.cpp)
add_executable(test_whisper_preprocessor tests/test_whisper_preprocessor.cpp)
target_link_libraries(test_whisper_preprocessor PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME whisper_preprocessor_reference COMMAND test_whisper_preprocessor ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(whisper_preprocessor_reference PROPERTIES TIMEOUT 120 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
add_executable(test_whisper_tokenizer_loader tests/test_whisper_tokenizer_loader.cpp)
target_link_libraries(test_whisper_tokenizer_loader PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME whisper_tokenizer_loader_reference COMMAND test_whisper_tokenizer_loader ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(whisper_tokenizer_loader_reference PROPERTIES TIMEOUT 60)
target_link_libraries(test_whisper_transcription PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME whisper_transcription_reference COMMAND test_whisper_transcription ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(whisper_transcription_reference PROPERTIES TIMEOUT 120)

add_executable(test_whisper_batched tests/test_whisper_batched.cpp)
target_link_libraries(test_whisper_batched PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME whisper_batched_reference COMMAND test_whisper_batched ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(whisper_batched_reference PROPERTIES TIMEOUT 120)

add_executable(test_stt_runtime tests/test_stt_runtime.cpp)
add_executable(test_stt_pool tests/test_stt_pool.cpp)
add_executable(test_stt_api tests/test_stt_api.cpp)
target_link_libraries(test_stt_api PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME stt_api_reference COMMAND test_stt_api ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_api_cases.jsonl)
set_tests_properties(stt_api_reference PROPERTIES TIMEOUT 120)
add_executable(test_subtitles tests/test_subtitles.cpp)
target_link_libraries(test_subtitles PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME subtitles_reference COMMAND test_subtitles ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(subtitles_reference PROPERTIES TIMEOUT 120)
add_executable(test_stt_pipeline tests/test_stt_pipeline.cpp)
target_link_libraries(test_stt_pipeline PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME stt_pipeline_reference COMMAND test_stt_pipeline ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_pipeline_cases.jsonl)
set_tests_properties(stt_pipeline_reference PROPERTIES TIMEOUT 120)
add_executable(test_stt_pipeline_http tests/test_stt_pipeline_http.cpp)
target_link_libraries(test_stt_pipeline_http PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
target_compile_definitions(test_stt_pipeline_http PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_stt_pipeline_http test_embedding_fixture)
add_test(NAME stt_pipeline_http_reference COMMAND test_stt_pipeline_http ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
set_tests_properties(stt_pipeline_http_reference PROPERTIES TIMEOUT 240)
add_executable(test_voice_slots tests/test_voice_slots.cpp)
target_link_libraries(test_voice_slots PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME voice_slots_reference COMMAND test_voice_slots ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/voice_slot_cases.jsonl)
set_tests_properties(voice_slots_reference PROPERTIES TIMEOUT 120)
add_executable(test_voice_slots_async tests/test_voice_slots_async.cpp)
target_link_libraries(test_voice_slots_async PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME voice_slots_async_reference COMMAND test_voice_slots_async ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/voice_slot_async_cases.jsonl)
set_tests_properties(voice_slots_async_reference PROPERTIES TIMEOUT 120)
target_link_libraries(test_stt_pool PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME stt_pool_reference COMMAND test_stt_pool ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_pool_cases.jsonl)
set_tests_properties(stt_pool_reference PROPERTIES TIMEOUT 120)
target_link_libraries(test_stt_runtime PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME stt_runtime_reference COMMAND test_stt_runtime ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_runtime_cases.jsonl)
set_tests_properties(stt_runtime_reference PROPERTIES TIMEOUT 120)
add_library(portaudio_fixture SHARED tests/portaudio_fixture.cpp)
add_executable(test_stt_audio tests/test_stt_audio.cpp)
target_link_libraries(test_stt_audio PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_dependencies(test_stt_audio portaudio_fixture)
add_test(NAME stt_audio_reference COMMAND test_stt_audio ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures $<TARGET_FILE:portaudio_fixture>)
set_tests_properties(stt_audio_reference PROPERTIES TIMEOUT 30)
find_library(EZLOCALAI_PORTAUDIO_LIBRARY NAMES portaudio libportaudio.so.2)
if(EZLOCALAI_PORTAUDIO_LIBRARY)
    add_test(NAME stt_audio_live_reference COMMAND test_stt_audio ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures --live)
    set_tests_properties(stt_audio_live_reference PROPERTIES TIMEOUT 30)
endif()
add_executable(test_hub_snapshot tests/test_hub_snapshot.cpp)
target_link_libraries(test_hub_snapshot PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME hub_snapshot_reference COMMAND test_hub_snapshot ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/hub_snapshot_cases.jsonl)
set_tests_properties(hub_snapshot_reference PROPERTIES TIMEOUT 180)

add_executable(test_whisper_vad tests/test_whisper_vad.cpp)
target_link_libraries(test_whisper_vad PRIVATE ezlocalai_native)
add_test(NAME whisper_vad_reference COMMAND test_whisper_vad ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/whisper_vad_cases.tsv)
set_tests_properties(whisper_vad_reference PROPERTIES TIMEOUT 60)

set(EZLOCALAI_ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "ONNX Runtime native headers")
set(EZLOCALAI_ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "ONNX Runtime native shared library")
set(EZLOCALAI_WHISPER_VAD_MODERN_MODELS "" CACHE PATH "faster-whisper 1.2.0 Silero model directory")
set(EZLOCALAI_WHISPER_VAD_LEGACY_MODELS "" CACHE PATH "faster-whisper 1.0.3 Silero model directory")
if(EZLOCALAI_ONNXRUNTIME_INCLUDE_DIR AND EZLOCALAI_ONNXRUNTIME_LIBRARY)
    add_library(ezlocalai_whisper_vad src/whisper_vad_model.cpp)
    target_include_directories(ezlocalai_whisper_vad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${EZLOCALAI_ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(ezlocalai_whisper_vad PUBLIC ezlocalai_native PRIVATE ${EZLOCALAI_ONNXRUNTIME_LIBRARY})
    if(EZLOCALAI_WHISPER_VAD_MODERN_MODELS AND EZLOCALAI_WHISPER_VAD_LEGACY_MODELS)
        add_executable(test_whisper_vad_model tests/test_whisper_vad.cpp)
        target_compile_definitions(test_whisper_vad_model PRIVATE EZLOCALAI_TEST_VAD_MODEL)
        target_link_libraries(test_whisper_vad_model PRIVATE ezlocalai_whisper_vad)
        add_test(NAME whisper_vad_model_reference COMMAND test_whisper_vad_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/whisper_vad_model_cases.tsv ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS})
        set_tests_properties(whisper_vad_model_reference PROPERTIES TIMEOUT 180 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
    endif()
endif()

set(EZLOCALAI_LIBTORCH_ROOT "" CACHE PATH "Pinned LibTorch 2.9.1 distribution (C++ headers and libraries)")
set(EZLOCALAI_CUDA_INCLUDE_DIRS "" CACHE STRING "CUDA SDK include directories matching the optional LibTorch build")
set(EZLOCALAI_STT_CUDA_LIBRARY_PATH "" CACHE STRING "CUDA/cuDNN runtime search path for GPU reference tests")
option(EZLOCALAI_STT_TORCH_GPU_TESTS "Run hardware-pinned STT platform comparisons on an available CUDA GPU" OFF)
if(EZLOCALAI_LIBTORCH_ROOT)
    if(EXISTS "${EZLOCALAI_LIBTORCH_ROOT}/build-version")
        file(READ "${EZLOCALAI_LIBTORCH_ROOT}/build-version" EZLOCALAI_LIBTORCH_VERSION)
    else()
        # PyTorch wheels also contain the native C++ distribution. Read its
        # version metadata without finding or invoking a Python interpreter.
        file(READ "${EZLOCALAI_LIBTORCH_ROOT}/share/cmake/Torch/TorchConfigVersion.cmake" EZLOCALAI_LIBTORCH_VERSION_FILE)
        string(REGEX MATCH "set\\(PACKAGE_VERSION \"([^\"]+)\"\\)" EZLOCALAI_LIBTORCH_VERSION_MATCH "${EZLOCALAI_LIBTORCH_VERSION_FILE}")
        set(EZLOCALAI_LIBTORCH_VERSION "${CMAKE_MATCH_1}")
    endif()
    string(STRIP "${EZLOCALAI_LIBTORCH_VERSION}" EZLOCALAI_LIBTORCH_VERSION)
    if(NOT EZLOCALAI_LIBTORCH_VERSION MATCHES "^2\\.9\\.1(\\+.*)?$")
        message(FATAL_ERROR "STT platform parity requires LibTorch 2.9.1, found ${EZLOCALAI_LIBTORCH_VERSION}")
    endif()
    add_library(ezlocalai_stt_torch src/stt_torch.cpp)
    target_include_directories(ezlocalai_stt_torch SYSTEM PUBLIC "${EZLOCALAI_LIBTORCH_ROOT}/include" "${EZLOCALAI_LIBTORCH_ROOT}/include/torch/csrc/api/include")
    # The frontend used here is exported by torch_cpu/torch_cuda. The wheel's
    # libtorch.so also exports a libstdc++ compatibility shim, which interposes
    # filesystem functions and crashes remove_all with this native toolchain.
    # Link the implementation libraries directly, keeping the host C++ runtime.
    target_link_libraries(ezlocalai_stt_torch PUBLIC ezlocalai_native "${EZLOCALAI_LIBTORCH_ROOT}/lib/libtorch_cpu.so" "${EZLOCALAI_LIBTORCH_ROOT}/lib/libc10.so")
    if(EXISTS "${EZLOCALAI_LIBTORCH_ROOT}/lib/libtorch_cuda.so")
        if(EZLOCALAI_USE_HIP)
            message(FATAL_ERROR "CUDA LibTorch cannot be combined with the HIP device backend")
        endif()
        if(NOT EZLOCALAI_CUDA_INCLUDE_DIRS)
            message(FATAL_ERROR "CUDA LibTorch requires matching EZLOCALAI_CUDA_INCLUDE_DIRS")
        endif()
        target_compile_definitions(ezlocalai_stt_torch PUBLIC EZLOCALAI_TORCH_CUDA USE_CUDA)
        target_include_directories(ezlocalai_stt_torch SYSTEM PUBLIC ${EZLOCALAI_CUDA_INCLUDE_DIRS})
        target_link_libraries(ezlocalai_stt_torch PUBLIC "${EZLOCALAI_LIBTORCH_ROOT}/lib/libtorch_cuda.so" "${EZLOCALAI_LIBTORCH_ROOT}/lib/libc10_cuda.so")
    elseif(EXISTS "${EZLOCALAI_LIBTORCH_ROOT}/lib/libtorch_hip.so")
        if(NOT EZLOCALAI_USE_HIP)
            message(FATAL_ERROR "ROCm LibTorch requires EZLOCALAI_USE_HIP=ON for consistent device reporting")
        endif()
        target_compile_definitions(ezlocalai_stt_torch PUBLIC EZLOCALAI_TORCH_HIP USE_ROCM __HIP_PLATFORM_AMD__=1)
        target_include_directories(ezlocalai_stt_torch SYSTEM PUBLIC ${EZLOCALAI_HIP_INCLUDE_DIR})
        target_link_libraries(ezlocalai_stt_torch PUBLIC "${EZLOCALAI_LIBTORCH_ROOT}/lib/libtorch_hip.so"
            "${EZLOCALAI_LIBTORCH_ROOT}/lib/libc10_hip.so" "${EZLOCALAI_LIBTORCH_ROOT}/lib/libamdhip64.so")
    endif()
    if(EXISTS "${EZLOCALAI_LIBTORCH_ROOT}/lib/libtorch_cuda.so" OR EXISTS "${EZLOCALAI_LIBTORCH_ROOT}/lib/libtorch_hip.so")
        add_executable(test_stt_torch tests/test_stt_torch.cpp)
        target_link_libraries(test_stt_torch PRIVATE ezlocalai_stt_torch nlohmann_json::nlohmann_json)
        add_test(NAME stt_torch_unavailable_reference COMMAND test_stt_torch ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures --unavailable)
        if(EZLOCALAI_USE_HIP)
            set_tests_properties(stt_torch_unavailable_reference PROPERTIES TIMEOUT 60 ENVIRONMENT "HIP_VISIBLE_DEVICES=-1;CUDA_VISIBLE_DEVICES=-1")
        else()
            set_tests_properties(stt_torch_unavailable_reference PROPERTIES TIMEOUT 60 ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
        endif()
        if(EZLOCALAI_STT_TORCH_GPU_TESTS AND NOT EZLOCALAI_USE_HIP)
            add_test(NAME stt_torch_gpu_reference COMMAND test_stt_torch ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures --gpu)
            set_tests_properties(stt_torch_gpu_reference PROPERTIES TIMEOUT 120 RESOURCE_LOCK stt_torch_gpu)
        endif()
    endif()
endif()

# CTranslate2 distributions may use the old libstdc++ string ABI. Only the
# kernel adapter uses that ABI; the server talks to it through plain C types.
set(EZLOCALAI_CTRANSLATE2_INCLUDE_DIR "" CACHE PATH "CTranslate2 native headers")
set(EZLOCALAI_CTRANSLATE2_LIBRARY "" CACHE FILEPATH "CTranslate2 native shared library")
option(EZLOCALAI_CTRANSLATE2_LEGACY_ABI "Compile the CTranslate2 adapter with libstdc++ ABI 0" OFF)
set(EZLOCALAI_WHISPER_KERNEL_FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/whisper_kernel_assets" CACHE PATH "Reference real-model Whisper kernel fixtures")
set(EZLOCALAI_WHISPER_CT2_MODEL "" CACHE PATH "Pinned CTranslate2 Whisper model for verification")
set(EZLOCALAI_WHISPER_CT2_MULTILINGUAL_MODEL "" CACHE PATH "Pinned multilingual CTranslate2 Whisper model for verification")
set(EZLOCALAI_WHISPER_CT2_TURBO_MODEL "" CACHE PATH "Pinned large-v3-turbo model for installed-dependency verification")
if(EZLOCALAI_CTRANSLATE2_LIBRARY AND EZLOCALAI_CTRANSLATE2_INCLUDE_DIR)
    add_library(ezlocalai_whisper_kernel SHARED src/whisper_kernel.cpp)
    target_include_directories(ezlocalai_whisper_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${EZLOCALAI_CTRANSLATE2_INCLUDE_DIR})
    target_link_libraries(ezlocalai_whisper_kernel PRIVATE ${EZLOCALAI_CTRANSLATE2_LIBRARY})
    add_library(ezlocalai_whisper_engine src/whisper_native_engine.cpp)
    target_link_libraries(ezlocalai_whisper_engine PUBLIC ezlocalai_native ezlocalai_whisper_kernel)
    if(TARGET ezlocalai_whisper_vad)
        add_library(ezlocalai_stt_whisper src/stt_whisper.cpp)
        target_link_libraries(ezlocalai-native PRIVATE ezlocalai_stt_whisper)
        target_compile_definitions(ezlocalai-native PRIVATE EZLOCALAI_NATIVE_STT EZLOCALAI_WHISPER_VAD_DIRECTORY="${CMAKE_INSTALL_PREFIX}/share/ezlocalai/whisper-vad")
        if(EZLOCALAI_WHISPER_VAD_MODERN_MODELS)
            install(DIRECTORY "${EZLOCALAI_WHISPER_VAD_MODERN_MODELS}/" DESTINATION share/ezlocalai/whisper-vad FILES_MATCHING PATTERN "*.onnx" PATTERN "__pycache__" EXCLUDE)
            install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/licenses/silero-vad.txt" "${CMAKE_CURRENT_SOURCE_DIR}/licenses/faster-whisper.txt" DESTINATION share/ezlocalai/whisper-vad/licenses)
        endif()
        target_link_libraries(ezlocalai_stt_whisper PUBLIC ezlocalai_whisper_engine ezlocalai_whisper_vad PRIVATE nlohmann_json::nlohmann_json)
        if(TARGET ezlocalai_stt_torch)
            target_link_libraries(ezlocalai_stt_whisper PUBLIC ezlocalai_stt_torch)
            target_compile_definitions(ezlocalai_stt_whisper PRIVATE EZLOCALAI_STT_TORCH)
        endif()
        add_executable(test_stt_diarization_model tests/test_stt_diarization_model.cpp)
        target_link_libraries(test_stt_diarization_model PRIVATE ezlocalai_stt_whisper nlohmann_json::nlohmann_json)
        add_executable(test_stt_deployment tests/test_stt_deployment.cpp)
        target_link_libraries(test_stt_deployment PRIVATE ezlocalai_stt_whisper nlohmann_json::nlohmann_json)
        if(EZLOCALAI_WHISPER_CT2_TURBO_MODEL AND EZLOCALAI_WHISPER_VAD_MODERN_MODELS)
            add_test(NAME stt_deployment_cpu_reference COMMAND test_stt_deployment ${EZLOCALAI_REFERENCE_ROOT}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_deployment_cpu.json
                ${EZLOCALAI_WHISPER_CT2_TURBO_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_runtime_assets/speech.wav)
            set_tests_properties(stt_deployment_cpu_reference PROPERTIES TIMEOUT 300 ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OPENBLAS_NUM_THREADS=1")
            if(TARGET ezlocalai_stt_torch AND EZLOCALAI_STT_TORCH_GPU_TESTS)
                add_test(NAME stt_deployment_cuda_reference COMMAND test_stt_deployment ${EZLOCALAI_REFERENCE_ROOT}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_deployment_cuda.json
                    ${EZLOCALAI_WHISPER_CT2_TURBO_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_runtime_assets/speech.wav)
                set_tests_properties(stt_deployment_cuda_reference PROPERTIES TIMEOUT 300 RESOURCE_LOCK stt_torch_gpu ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
            endif()
        endif()
    endif()
    if(EZLOCALAI_CTRANSLATE2_LEGACY_ABI)
        target_compile_definitions(ezlocalai_whisper_kernel PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    endif()
    get_filename_component(CT2_LIBRARY_DIRECTORY ${EZLOCALAI_CTRANSLATE2_LIBRARY} DIRECTORY)
    set_target_properties(ezlocalai_whisper_kernel PROPERTIES BUILD_RPATH ${CT2_LIBRARY_DIRECTORY})
    if(UNIX AND NOT APPLE)
        target_link_options(ezlocalai_whisper_kernel PRIVATE -Wl,--disable-new-dtags)
        target_link_options(ezlocalai_whisper_kernel INTERFACE "-Wl,-rpath-link,${CT2_LIBRARY_DIRECTORY}")
    endif()
    if(EZLOCALAI_WHISPER_KERNEL_FIXTURES AND EZLOCALAI_WHISPER_CT2_MODEL)
        add_executable(test_whisper_kernel tests/test_whisper_kernel.cpp)
        target_link_libraries(test_whisper_kernel PRIVATE ezlocalai_native ezlocalai_whisper_kernel)
        add_test(NAME whisper_kernel_reference COMMAND test_whisper_kernel ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_KERNEL_FIXTURES})
        set_tests_properties(whisper_kernel_reference PROPERTIES TIMEOUT 180 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
        if(TARGET test_whisper_vad_model)
            add_executable(test_whisper_speech tests/test_whisper_speech.cpp)
            target_link_libraries(test_whisper_speech PRIVATE ezlocalai_whisper_vad ezlocalai_whisper_kernel)
            add_test(NAME whisper_speech_reference COMMAND test_whisper_speech ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/whisper_speech_cases.tsv ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS})
            set_tests_properties(whisper_speech_reference PROPERTIES TIMEOUT 180 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
            if(EZLOCALAI_WHISPER_CT2_MULTILINGUAL_MODEL)
                add_executable(test_whisper_transcription_model tests/test_whisper_transcription_model.cpp)
                target_link_libraries(test_whisper_transcription_model PRIVATE ezlocalai_whisper_engine ezlocalai_whisper_vad nlohmann_json::nlohmann_json)
                add_test(NAME whisper_transcription_model_reference COMMAND test_whisper_transcription_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_CT2_MULTILINGUAL_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS})
                set_tests_properties(whisper_transcription_model_reference PROPERTIES TIMEOUT 600 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
                add_executable(test_whisper_batched_model tests/test_whisper_batched_model.cpp)
                target_link_libraries(test_whisper_batched_model PRIVATE ezlocalai_whisper_engine ezlocalai_whisper_vad nlohmann_json::nlohmann_json)
                foreach(BLAS_THREADS IN ITEMS 1 8)
                    add_test(NAME whisper_batched_model_threads_${BLAS_THREADS}_reference COMMAND test_whisper_batched_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_CT2_MULTILINGUAL_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${BLAS_THREADS})
                    set_tests_properties(whisper_batched_model_threads_${BLAS_THREADS}_reference PROPERTIES TIMEOUT 600 ENVIRONMENT "OPENBLAS_NUM_THREADS=${BLAS_THREADS}")
                endforeach()
                add_executable(test_stt_runtime_model tests/test_stt_runtime_model.cpp)
                add_executable(test_stt_api_model tests/test_stt_api_model.cpp)
                add_executable(test_stt_pipeline_model tests/test_stt_pipeline_model.cpp)
                target_link_libraries(test_stt_pipeline_model PRIVATE ezlocalai_stt_whisper nlohmann_json::nlohmann_json)
                add_test(NAME stt_pipeline_model_reference COMMAND test_stt_pipeline_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS})
                set_tests_properties(stt_pipeline_model_reference PROPERTIES TIMEOUT 600 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
                add_executable(test_stt_api_main tests/test_stt_api_main.cpp)
                target_link_libraries(test_stt_api_main PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
                add_dependencies(test_stt_api_main ezlocalai-native)
                add_test(NAME stt_api_main_reference COMMAND test_stt_api_main $<TARGET_FILE:ezlocalai-native> ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS})
                set_tests_properties(stt_api_main_reference PROPERTIES TIMEOUT 300)
                target_link_libraries(test_stt_api_model PRIVATE ezlocalai_stt_whisper nlohmann_json::nlohmann_json)
                add_test(NAME stt_api_model_reference COMMAND test_stt_api_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS})
                set_tests_properties(stt_api_model_reference PROPERTIES TIMEOUT 600 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
                add_test(NAME subtitles_api_model_reference COMMAND test_stt_api_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS} subtitles_api_model)
                set_tests_properties(subtitles_api_model_reference PROPERTIES TIMEOUT 600 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
                add_executable(test_stt_pool_model tests/test_stt_pool_model.cpp)
                target_link_libraries(test_stt_pool_model PRIVATE ezlocalai_stt_whisper nlohmann_json::nlohmann_json)
                add_test(NAME stt_pool_model_reference COMMAND test_stt_pool_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS})
                set_tests_properties(stt_pool_model_reference PROPERTIES TIMEOUT 600 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
                target_link_libraries(test_stt_runtime_model PRIVATE ezlocalai_stt_whisper nlohmann_json::nlohmann_json)
                add_test(NAME stt_runtime_model_reference COMMAND test_stt_runtime_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_CT2_MULTILINGUAL_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS})
                set_tests_properties(stt_runtime_model_reference PROPERTIES TIMEOUT 600 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
                add_executable(test_whisper_preprocessor_model tests/test_whisper_preprocessor_model.cpp)
                target_link_libraries(test_whisper_preprocessor_model PRIVATE ezlocalai_stt_whisper nlohmann_json::nlohmann_json)
                add_test(NAME whisper_preprocessor_model_reference COMMAND test_whisper_preprocessor_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS})
                set_tests_properties(whisper_preprocessor_model_reference PROPERTIES TIMEOUT 600 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
                add_executable(test_whisper_tokenizer_fallback_model tests/test_whisper_tokenizer_fallback_model.cpp)
                target_link_libraries(test_whisper_tokenizer_fallback_model PRIVATE ezlocalai_stt_whisper nlohmann_json::nlohmann_json)
                add_test(NAME whisper_tokenizer_fallback_model_reference COMMAND test_whisper_tokenizer_fallback_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_CT2_MULTILINGUAL_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS})
                set_tests_properties(whisper_tokenizer_fallback_model_reference PROPERTIES TIMEOUT 600 ENVIRONMENT "OPENBLAS_NUM_THREADS=1")
                if(TARGET ezlocalai_stt_torch AND EZLOCALAI_STT_TORCH_GPU_TESTS)
                    add_test(NAME stt_runtime_gpu_model_reference COMMAND test_stt_runtime_model ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_CT2_MULTILINGUAL_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS} ${EZLOCALAI_WHISPER_VAD_LEGACY_MODELS} --gpu)
                    set_tests_properties(stt_runtime_gpu_model_reference PROPERTIES TIMEOUT 600 RESOURCE_LOCK stt_torch_gpu ENVIRONMENT "OPENBLAS_NUM_THREADS=1;LD_LIBRARY_PATH=${EZLOCALAI_STT_CUDA_LIBRARY_PATH}")
                endif()
            endif()
        endif()
    endif()
endif()

add_executable(test_llm_stream tests/test_llm_stream.cpp)
target_link_libraries(test_llm_stream PRIVATE ezlocalai_native)
target_compile_definitions(test_llm_stream PRIVATE TEST_EMBEDDING_FIXTURE_BIN="$<TARGET_FILE:test_embedding_fixture>")
add_dependencies(test_llm_stream test_embedding_fixture)
add_test(NAME llm_stream_reference COMMAND test_llm_stream ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_stream_cases.tsv)
set_tests_properties(llm_stream_reference PROPERTIES TIMEOUT 60)

add_executable(test_llm_stream_lifecycle tests/test_llm_stream_lifecycle.cpp)
target_link_libraries(test_llm_stream_lifecycle PRIVATE ezlocalai_native)
add_test(NAME llm_stream_lifecycle_reference COMMAND test_llm_stream_lifecycle ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llm_stream_lifecycle_cases.tsv)
set_tests_properties(llm_stream_lifecycle_reference PROPERTIES TIMEOUT 90)

# Model-free stream termination/admission contracts also run in images without
# an external Python reference checkout.
add_executable(test_stream_completion tests/test_stream_completion.cpp)
target_link_libraries(test_stream_completion PRIVATE ezlocalai_native)
add_test(NAME stream_completion COMMAND test_stream_completion)
set_tests_properties(stream_completion PROPERTIES TIMEOUT 30)
add_executable(bench_stream_completion EXCLUDE_FROM_ALL benchmarks/bench_stream_completion.cpp)
target_link_libraries(bench_stream_completion PRIVATE ezlocalai_native)

# Optional direct parser verification uses an independently built reference
# engine. First-party targets remain C++ only; no engine is fetched implicitly.
set(EZLOCALAI_LLAMA_CPP_REFERENCE_SOURCE "" CACHE PATH "Pinned llama.cpp source for engine contract tests")
set(EZLOCALAI_LLAMA_CPP_REFERENCE_BUILD "" CACHE PATH "Existing pinned llama.cpp build for engine contract tests")
set(EZLOCALAI_LLAMA_CPP_SMOKE_MODEL "" CACHE FILEPATH "Optional pinned stories260K.gguf for real CPU inference verification")
if(EZLOCALAI_LLAMA_CPP_REFERENCE_SOURCE AND EZLOCALAI_LLAMA_CPP_REFERENCE_BUILD)
    add_executable(test_llm_engine_contract tests/test_llm_engine_contract.cpp)
    target_include_directories(test_llm_engine_contract PRIVATE include
        ${EZLOCALAI_LLAMA_CPP_REFERENCE_SOURCE}/common ${EZLOCALAI_LLAMA_CPP_REFERENCE_SOURCE}/include ${EZLOCALAI_LLAMA_CPP_REFERENCE_SOURCE}/ggml/include)
    target_link_libraries(test_llm_engine_contract PRIVATE
        ${EZLOCALAI_LLAMA_CPP_REFERENCE_BUILD}/bin/libllama-common.so ${EZLOCALAI_LLAMA_CPP_REFERENCE_BUILD}/bin/libllama.so OpenSSL::Crypto)
    target_compile_definitions(test_llm_engine_contract PRIVATE TEST_LLAMA_SOURCE_DIRECTORY="${EZLOCALAI_LLAMA_CPP_REFERENCE_SOURCE}")
    if(EXISTS "${EZLOCALAI_LLAMA_CPP_REFERENCE_SOURCE}/common/speculative-mtp-state.h")
        target_compile_definitions(test_llm_engine_contract PRIVATE TEST_LLAMA_MANAGED_ENGINE=1)
        set(engine_source_fixture llm_engine_managed_source_hashes.tsv)
    else()
        set(engine_source_fixture llm_engine_source_hashes.tsv)
    endif()
    add_test(NAME llm_engine_contract COMMAND test_llm_engine_contract
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/${engine_source_fixture})
    set_tests_properties(llm_engine_contract PROPERTIES TIMEOUT 30)
    if(EZLOCALAI_LLAMA_CPP_SMOKE_MODEL)
        add_test(NAME llm_real_engine_smoke COMMAND test_model_download --real-engine
            ${EZLOCALAI_LLAMA_CPP_REFERENCE_BUILD}/bin/llama-server ${EZLOCALAI_LLAMA_CPP_SMOKE_MODEL})
        set_tests_properties(llm_real_engine_smoke PROPERTIES TIMEOUT 90 DEPENDS llm_engine_contract)
    endif()
endif()

set(EZLOCALAI_TTS_LLAMA_SOURCE "${EZLOCALAI_LLAMA_CPP_REFERENCE_SOURCE}" CACHE PATH "Pinned llama.cpp source for native TTS worker")
set(EZLOCALAI_TTS_LLAMA_BUILD "${EZLOCALAI_LLAMA_CPP_REFERENCE_BUILD}" CACHE PATH "Pinned llama.cpp build for native TTS worker")
# Cache policy checks link the actual managed engine implementation. Custom
# unpatched revisions do not expose this private cache API.
if(EXISTS "${EZLOCALAI_TTS_LLAMA_SOURCE}/tools/server/server-task.h" AND
   EXISTS "${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libllama-server-impl.so" AND
   EXISTS "${EZLOCALAI_TTS_LLAMA_BUILD}/ezlocalai-llama-cpp-patches.txt")
    add_executable(test_conversation_cache tests/test_conversation_cache.cpp)
    target_include_directories(test_conversation_cache PRIVATE
        ${EZLOCALAI_TTS_LLAMA_SOURCE}/tools/server ${EZLOCALAI_TTS_LLAMA_SOURCE}/common
        ${EZLOCALAI_TTS_LLAMA_SOURCE}/include ${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/include
        ${EZLOCALAI_TTS_LLAMA_SOURCE}/tools/mtmd ${EZLOCALAI_TTS_LLAMA_SOURCE}/vendor)
    target_link_libraries(test_conversation_cache PRIVATE ezlocalai_native
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libllama-server-impl.so
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libllama-common.so
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libmtmd.so
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libllama.so
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml.so
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml-base.so)
    add_test(NAME conversation_cache COMMAND test_conversation_cache)
    set_tests_properties(conversation_cache PROPERTIES TIMEOUT 30)
    add_executable(test_cache_restore_atomic tests/test_cache_restore_atomic.cpp
        ${EZLOCALAI_TTS_LLAMA_SOURCE}/tools/server/server-task.cpp)
    target_include_directories(test_cache_restore_atomic PRIVATE
        ${EZLOCALAI_TTS_LLAMA_SOURCE}/tools/server ${EZLOCALAI_TTS_LLAMA_SOURCE}/common
        ${EZLOCALAI_TTS_LLAMA_SOURCE}/include ${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/include
        ${EZLOCALAI_TTS_LLAMA_SOURCE}/tools/mtmd ${EZLOCALAI_TTS_LLAMA_SOURCE}/vendor)
    target_link_libraries(test_cache_restore_atomic PRIVATE
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libllama-server-impl.so
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libllama-common.so
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libmtmd.so
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libllama.so
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml.so
        ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml-base.so)
    add_test(NAME cache_restore_atomic COMMAND test_cache_restore_atomic)
    set_tests_properties(cache_restore_atomic PROPERTIES TIMEOUT 30)
    target_compile_definitions(test_cache_restore_atomic PRIVATE EZLOCALAI_TEST_MTP_TRANSPORT=1)
    add_executable(test_cache_storage tests/test_cache_storage.cpp
        ${EZLOCALAI_TTS_LLAMA_SOURCE}/tools/server/server-task.cpp)
    target_include_directories(test_cache_storage PRIVATE
        $<TARGET_PROPERTY:test_cache_restore_atomic,INCLUDE_DIRECTORIES>)
    target_link_libraries(test_cache_storage PRIVATE
        $<TARGET_PROPERTY:test_cache_restore_atomic,LINK_LIBRARIES>)
    add_test(NAME cache_storage COMMAND test_cache_storage)
    set_tests_properties(cache_storage PROPERTIES TIMEOUT 30)
    add_executable(test_llama_mtp_resume tests/test_llama_mtp_resume.cpp)
    target_include_directories(test_llama_mtp_resume PRIVATE
        $<TARGET_PROPERTY:test_cache_restore_atomic,INCLUDE_DIRECTORIES>)
    add_test(NAME llama_mtp_resume COMMAND test_llama_mtp_resume)
    set_tests_properties(llama_mtp_resume PROPERTIES TIMEOUT 30)
    add_executable(test_llama_checkpoint_coverage tests/test_llama_checkpoint_coverage.cpp)
    target_include_directories(test_llama_checkpoint_coverage PRIVATE
        $<TARGET_PROPERTY:test_cache_restore_atomic,INCLUDE_DIRECTORIES>
        ${EZLOCALAI_TTS_LLAMA_SOURCE}/src)
    target_link_libraries(test_llama_checkpoint_coverage PRIVATE
        $<TARGET_PROPERTY:test_cache_restore_atomic,LINK_LIBRARIES>)
    add_test(NAME llama_checkpoint_coverage COMMAND test_llama_checkpoint_coverage)
    # The exhaustive model-free phase grid also ships in ARM/QEMU packages.
    set_tests_properties(llama_checkpoint_coverage PROPERTIES TIMEOUT 300)
    if(EXISTS "${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/src/ggml-cuda/fattn-q4-kv-policy.h")
        # The CUDA dispatch policy is ordinary C++ and ships in every package.
        add_executable(test_q4_kv_layout tests/test_q4_kv_layout.cpp)
        target_include_directories(test_q4_kv_layout PRIVATE
            ${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/include
            ${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/src/ggml-cuda)
        add_test(NAME q4_kv_layout COMMAND test_q4_kv_layout)
        set_tests_properties(q4_kv_layout PROPERTIES TIMEOUT 30)
    endif()
    add_custom_target(test_managed_llama_patches SOURCES tests/test_managed_llama_patches.cmake)
    add_test(NAME managed_llama_patches COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${EZLOCALAI_TTS_LLAMA_SOURCE}
        -DHELPER=${CMAKE_CURRENT_SOURCE_DIR}/cmake/ApplyLlamaPatches.cmake
        -DPATCH_DIR=${CMAKE_CURRENT_SOURCE_DIR}/patches
        -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_managed_llama_patches.cmake)
    set_tests_properties(managed_llama_patches PROPERTIES TIMEOUT 120)
    if(EZLOCALAI_LLAMA_CPP_SMOKE_MODEL)
        add_test(NAME conversation_cache_engine COMMAND test_conversation_cache --engine
            ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/llama-server ${EZLOCALAI_LLAMA_CPP_SMOKE_MODEL})
        set_tests_properties(conversation_cache_engine PROPERTIES TIMEOUT 90)
    endif()
endif()
# Physical GPU oracles are explicitly built and manually executed on an owned
# device. They never enter a package's model-free CTest gate.
option(EZLOCALAI_BUILD_Q4_KV_GPU_ORACLES "Build developer-only Q4 KV CUDA oracles" OFF)
if(EZLOCALAI_BUILD_Q4_KV_GPU_ORACLES)
    include(cmake/Q4KvOracleTargets.cmake)
endif()
# The source application's private TTS worker, linked to an independently built
# pinned llama.cpp/libmtmd. No Python wrapper or implicit dependency fetch.
if(EZLOCALAI_TTS_LLAMA_SOURCE AND EZLOCALAI_TTS_LLAMA_BUILD)
    add_executable(ezlocalai-tts tts/worker.cpp)
    target_include_directories(ezlocalai-tts PRIVATE ${EZLOCALAI_TTS_LLAMA_SOURCE}/common ${EZLOCALAI_TTS_LLAMA_SOURCE}/include ${EZLOCALAI_TTS_LLAMA_SOURCE}/ggml/include ${EZLOCALAI_TTS_LLAMA_SOURCE}/tools/mtmd ${EZLOCALAI_TTS_LLAMA_SOURCE}/vendor)
    target_compile_definitions(ezlocalai-tts PRIVATE GGML_BACKEND_SHARED GGML_SHARED GGML_USE_CPU LLAMA_SHARED LLAMA_SUBPROCESS)
    # Do not put llama.cpp's GGML directory in the image-wide library path:
    # ACE-Step ships a different GGML ABI under the same shared-library names.
    set_target_properties(ezlocalai-tts PROPERTIES INSTALL_RPATH "${EZLOCALAI_TTS_LLAMA_BUILD}/bin")
    target_link_libraries(ezlocalai-tts PRIVATE ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libllama-common.so ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libmtmd.so ${EZLOCALAI_TTS_LLAMA_BUILD}/common/libllama-common-base.a ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libllama.so ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml.so ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml-cpu.so ${EZLOCALAI_TTS_LLAMA_BUILD}/bin/libggml-base.so)
    install(TARGETS ezlocalai-tts RUNTIME DESTINATION bin)
endif()

add_executable(test_llama_tts_policy tests/test_llama_tts_policy.cpp)
target_link_libraries(test_llama_tts_policy PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME llama_tts_policy_reference COMMAND test_llama_tts_policy ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama_tts_policy_cases.jsonl)
set_tests_properties(llama_tts_policy_reference PROPERTIES TIMEOUT 120)
add_executable(test_tts_process_fixture tests/tts_process_fixture.cpp)
target_link_libraries(test_tts_process_fixture PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
add_executable(test_llama_tts_process tests/test_llama_tts_process.cpp)
target_link_libraries(test_llama_tts_process PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_dependencies(test_llama_tts_process test_tts_process_fixture)
add_test(NAME llama_tts_process_reference COMMAND test_llama_tts_process ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama_tts_process_cases.jsonl $<TARGET_FILE:test_tts_process_fixture>)
set_tests_properties(llama_tts_process_reference PROPERTIES TIMEOUT 120)
set(EZLOCALAI_TTS_MODEL_DIRECTORY "" CACHE PATH "Pinned Qwen TTS GGUF directory for real CPU wrapper verification")
set(EZLOCALAI_TTS_MODEL_CASES "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama_tts_model_v080_cases.jsonl"
    CACHE FILEPATH "Version-specific real CPU Qwen TTS reference cases")
if(TARGET ezlocalai-tts AND EZLOCALAI_TTS_MODEL_DIRECTORY)
    add_executable(test_llama_tts_model tests/test_llama_tts_model.cpp)
    target_link_libraries(test_llama_tts_model PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
    add_dependencies(test_llama_tts_model ezlocalai-tts)
    add_test(NAME llama_tts_model_reference COMMAND test_llama_tts_model ${EZLOCALAI_REFERENCE_ROOT} ${EZLOCALAI_TTS_MODEL_CASES} $<TARGET_FILE:ezlocalai-tts> ${EZLOCALAI_TTS_MODEL_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_runtime_assets/speech.wav)
    set_tests_properties(llama_tts_model_reference PROPERTIES TIMEOUT 300 DEPENDS llama_tts_process_reference)
    if(EZLOCALAI_STT_TORCH_GPU_TESTS)
        add_test(NAME llama_tts_deployment_cuda_reference COMMAND test_llama_tts_model ${EZLOCALAI_REFERENCE_ROOT}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/llama_tts_deployment_cuda.jsonl $<TARGET_FILE:ezlocalai-tts>
            ${EZLOCALAI_TTS_MODEL_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_runtime_assets/speech.wav)
        set_tests_properties(llama_tts_deployment_cuda_reference PROPERTIES TIMEOUT 300 RESOURCE_LOCK stt_torch_gpu)
    endif()
endif()



add_executable(test_ctts_text tests/test_ctts_text.cpp)
target_link_libraries(test_ctts_text PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME ctts_text_reference COMMAND test_ctts_text ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ctts_text_cases.jsonl.gz)
set_tests_properties(ctts_text_reference PROPERTIES TIMEOUT 120)

add_executable(test_ctts_runtime tests/test_ctts_runtime.cpp)
target_link_libraries(test_ctts_runtime PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME ctts_runtime_reference COMMAND test_ctts_runtime ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ctts_runtime_cases.jsonl)
set_tests_properties(ctts_runtime_reference PROPERTIES TIMEOUT 120)
if(TARGET ezlocalai_stt_torch)
    add_library(ezlocalai_ctts_torch src/ctts_torch.cpp src/ctts_resample.cpp src/wakeword_torch.cpp src/wakeword_checkpoint.cpp src/wakeword_onnx.cpp)
    target_link_libraries(ezlocalai_ctts_torch PRIVATE nlohmann_json::nlohmann_json)
    target_link_libraries(ezlocalai-native PRIVATE ezlocalai_ctts_torch)
    target_compile_definitions(ezlocalai-native PRIVATE EZLOCALAI_CTTS_TORCH)
    target_link_libraries(ezlocalai_ctts_torch PUBLIC ezlocalai_stt_torch)
    add_executable(test_torch_resources tests/test_torch_resources.cpp)
    target_link_libraries(test_torch_resources PRIVATE ezlocalai_ctts_torch)
    add_test(NAME torch_resources COMMAND test_torch_resources)
    set_tests_properties(torch_resources PROPERTIES TIMEOUT 60 RESOURCE_LOCK stt_torch_gpu)
    add_executable(test_wakeword_model tests/test_wakeword_model.cpp)
    target_link_libraries(test_wakeword_model PRIVATE ezlocalai_ctts_torch nlohmann_json::nlohmann_json)
    add_executable(test_wakeword_training tests/test_wakeword_training.cpp)
    target_link_libraries(test_wakeword_training PRIVATE ezlocalai_ctts_torch nlohmann_json::nlohmann_json)
    if(EZLOCALAI_ONNXRUNTIME_INCLUDE_DIR AND EZLOCALAI_ONNXRUNTIME_LIBRARY)
        add_executable(test_wakeword_export tests/test_wakeword_export.cpp)
        target_include_directories(test_wakeword_export PRIVATE ${EZLOCALAI_ONNXRUNTIME_INCLUDE_DIR})
        target_link_libraries(test_wakeword_export PRIVATE ezlocalai_ctts_torch nlohmann_json::nlohmann_json ${EZLOCALAI_ONNXRUNTIME_LIBRARY})
    endif()
    if(EZLOCALAI_WAKEWORD_REFERENCE_DIR)
        add_test(NAME wakeword_model_reference COMMAND test_wakeword_model ${EZLOCALAI_WAKEWORD_REFERENCE_DIR})
        set_tests_properties(wakeword_model_reference PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=2;MKL_NUM_THREADS=2;OPENBLAS_NUM_THREADS=2")
        if(EXISTS "${EZLOCALAI_WAKEWORD_REFERENCE_DIR}/training/reference.json")
            add_test(NAME wakeword_training_reference COMMAND test_wakeword_training ${EZLOCALAI_WAKEWORD_REFERENCE_DIR})
            set_tests_properties(wakeword_training_reference PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=2;MKL_NUM_THREADS=2;OPENBLAS_NUM_THREADS=2")
        endif()
        if(TARGET test_wakeword_export AND EXISTS "${EZLOCALAI_WAKEWORD_REFERENCE_DIR}/exports/reference.json")
            add_test(NAME wakeword_export_reference COMMAND test_wakeword_export ${EZLOCALAI_WAKEWORD_REFERENCE_DIR})
            set_tests_properties(wakeword_export_reference PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=2;MKL_NUM_THREADS=2;OPENBLAS_NUM_THREADS=2")
        endif()
    endif()
    if(EXISTS "${EZLOCALAI_LIBTORCH_ROOT}/lib/libtorch_cuda.so")
        string(REPLACE ":" ";" EZLOCALAI_CTTS_CUDA_PATHS "${EZLOCALAI_STT_CUDA_LIBRARY_PATH}")
        # Prefer the configured runtime over an unrelated host CUDA toolkit.
        find_library(EZLOCALAI_CTTS_CUDART_LIBRARY NAMES cudart libcudart.so.12 NAMES_PER_DIR HINTS ${EZLOCALAI_CTTS_CUDA_PATHS} "${EZLOCALAI_LIBTORCH_ROOT}/../nvidia/cuda_runtime/lib" NO_DEFAULT_PATH)
        if(NOT EZLOCALAI_CTTS_CUDART_LIBRARY)
            find_library(EZLOCALAI_CTTS_CUDART_LIBRARY NAMES cudart libcudart.so.12 HINTS /usr/local/cuda/lib64)
        endif()
        if(NOT EZLOCALAI_CTTS_CUDART_LIBRARY)
            message(FATAL_ERROR "Native CTTS CUDA memory queries require the matching CUDA runtime library")
        endif()
        target_link_libraries(ezlocalai_ctts_torch PRIVATE ${EZLOCALAI_CTTS_CUDART_LIBRARY})
    endif()
endif()
if(TARGET ezlocalai-tts AND EZLOCALAI_TTS_MODEL_DIRECTORY)
    add_executable(test_ctts_model tests/test_ctts_model.cpp)
    target_link_libraries(test_ctts_model PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
    if(TARGET ezlocalai_ctts_torch)
        target_link_libraries(test_ctts_model PRIVATE ezlocalai_ctts_torch)
        target_compile_definitions(test_ctts_model PRIVATE EZLOCALAI_CTTS_TORCH)
    endif()
    add_dependencies(test_ctts_model ezlocalai-tts)
    add_test(NAME ctts_model_reference COMMAND test_ctts_model ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ctts_model_cases.jsonl $<TARGET_FILE:ezlocalai-tts> ${EZLOCALAI_TTS_MODEL_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_runtime_assets/speech.wav)
    set_tests_properties(ctts_model_reference PROPERTIES TIMEOUT 300 RESOURCE_LOCK stt_torch_gpu ENVIRONMENT "LD_LIBRARY_PATH=${EZLOCALAI_STT_CUDA_LIBRARY_PATH}")
endif()

add_executable(test_ctts_audio tests/test_ctts_audio.cpp)
target_link_libraries(test_ctts_audio PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
if(TARGET ezlocalai_ctts_torch)
    target_link_libraries(test_ctts_audio PRIVATE ezlocalai_ctts_torch)
    target_compile_definitions(test_ctts_audio PRIVATE EZLOCALAI_CTTS_TORCH)
endif()
add_test(NAME ctts_audio_reference COMMAND test_ctts_audio ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ctts_audio_cases.jsonl.gz "${EZLOCALAI_SNDFILE_LIBRARY}")
set_tests_properties(ctts_audio_reference PROPERTIES TIMEOUT 120 ENVIRONMENT "LD_LIBRARY_PATH=${EZLOCALAI_STT_CUDA_LIBRARY_PATH}")

add_executable(test_ctts_paths tests/test_ctts_paths.cpp)
target_link_libraries(test_ctts_paths PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME ctts_paths_reference COMMAND test_ctts_paths ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ctts_paths_cases.jsonl)
set_tests_properties(ctts_paths_reference PROPERTIES TIMEOUT 120)

add_executable(test_ctts_cache tests/test_ctts_cache.cpp)
target_link_libraries(test_ctts_cache PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME ctts_cache_reference COMMAND test_ctts_cache ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ctts_cache_cases.jsonl.gz)
set_tests_properties(ctts_cache_reference PROPERTIES TIMEOUT 120)

add_executable(test_ctts_generate tests/test_ctts_generate.cpp)
target_link_libraries(test_ctts_generate PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME ctts_generate_reference COMMAND test_ctts_generate ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ctts_generate_cases.jsonl.gz "${EZLOCALAI_SNDFILE_LIBRARY}")
set_tests_properties(ctts_generate_reference PROPERTIES TIMEOUT 120)

if(TARGET ezlocalai-tts AND EZLOCALAI_TTS_MODEL_DIRECTORY)
    add_executable(test_ctts_generate_model tests/test_ctts_generate_model.cpp)
    target_link_libraries(test_ctts_generate_model PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
    if(TARGET ezlocalai_ctts_torch)
        target_link_libraries(test_ctts_generate_model PRIVATE ezlocalai_ctts_torch)
        target_compile_definitions(test_ctts_generate_model PRIVATE EZLOCALAI_CTTS_TORCH)
    endif()
    add_dependencies(test_ctts_generate_model ezlocalai-tts)
    add_test(NAME ctts_generate_model_reference COMMAND test_ctts_generate_model ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ctts_generate_model_cases.jsonl $<TARGET_FILE:ezlocalai-tts> ${EZLOCALAI_TTS_MODEL_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_runtime_assets/speech.wav "${EZLOCALAI_SNDFILE_LIBRARY}")
    set_tests_properties(ctts_generate_model_reference PROPERTIES TIMEOUT 300 RESOURCE_LOCK stt_torch_gpu ENVIRONMENT "LD_LIBRARY_PATH=${EZLOCALAI_STT_CUDA_LIBRARY_PATH}")
endif()

add_executable(test_audio_stream_bridge tests/test_audio_stream_bridge.cpp)
target_link_libraries(test_audio_stream_bridge PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME audio_stream_bridge_reference COMMAND test_audio_stream_bridge ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/audio_stream_bridge_cases.jsonl)
set_tests_properties(audio_stream_bridge_reference PROPERTIES TIMEOUT 120)

add_executable(test_audio_stream_bridge_process tests/test_audio_stream_bridge_process.cpp)
target_link_libraries(test_audio_stream_bridge_process PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_dependencies(test_audio_stream_bridge_process test_tts_process_fixture)
add_test(NAME audio_stream_bridge_process COMMAND test_audio_stream_bridge_process $<TARGET_FILE:test_tts_process_fixture>)
set_tests_properties(audio_stream_bridge_process PROPERTIES TIMEOUT 60)

add_executable(test_ctts_stream tests/test_ctts_stream.cpp)
target_link_libraries(test_ctts_stream PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME ctts_stream_reference COMMAND test_ctts_stream ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ctts_stream_cases.jsonl.gz)
set_tests_properties(ctts_stream_reference PROPERTIES TIMEOUT 120)

if(TARGET ezlocalai-tts AND EZLOCALAI_TTS_MODEL_DIRECTORY)
    add_executable(test_ctts_stream_model tests/test_ctts_stream_model.cpp)
    target_link_libraries(test_ctts_stream_model PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
    if(TARGET ezlocalai_ctts_torch)
        target_link_libraries(test_ctts_stream_model PRIVATE ezlocalai_ctts_torch)
        target_compile_definitions(test_ctts_stream_model PRIVATE EZLOCALAI_CTTS_TORCH)
    endif()
    add_dependencies(test_ctts_stream_model ezlocalai-tts)
    add_test(NAME ctts_stream_model_reference COMMAND test_ctts_stream_model ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/ctts_stream_model_cases.jsonl $<TARGET_FILE:ezlocalai-tts> ${EZLOCALAI_TTS_MODEL_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_runtime_assets/speech.wav)
    set_tests_properties(ctts_stream_model_reference PROPERTIES TIMEOUT 300 RESOURCE_LOCK stt_torch_gpu ENVIRONMENT "LD_LIBRARY_PATH=${EZLOCALAI_STT_CUDA_LIBRARY_PATH}")
endif()

add_executable(test_ctts_stream_process tests/test_ctts_stream_process.cpp)
target_link_libraries(test_ctts_stream_process PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_dependencies(test_ctts_stream_process test_tts_process_fixture)
add_test(NAME ctts_stream_process COMMAND test_ctts_stream_process $<TARGET_FILE:test_tts_process_fixture>)
set_tests_properties(ctts_stream_process PROPERTIES TIMEOUT 60)

add_executable(test_tts_pool tests/test_tts_pool.cpp)
target_link_libraries(test_tts_pool PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_pool_reference COMMAND test_tts_pool ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_pool_cases.jsonl.gz)
set_tests_properties(tts_pool_reference PROPERTIES TIMEOUT 120)

add_executable(test_tts_pool_process tests/test_tts_pool_process.cpp)
target_link_libraries(test_tts_pool_process PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_dependencies(test_tts_pool_process test_tts_process_fixture)
add_test(NAME tts_pool_process COMMAND test_tts_pool_process $<TARGET_FILE:test_tts_process_fixture>)
set_tests_properties(tts_pool_process PROPERTIES TIMEOUT 60)

add_executable(test_tts_service tests/test_tts_service.cpp)
add_executable(test_tts_pipeline_lease tests/test_tts_pipeline_lease.cpp)
target_link_libraries(test_tts_pipeline_lease PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_pipeline_lease_reference COMMAND test_tts_pipeline_lease ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_pipeline_lease_cases.jsonl.gz)
set_tests_properties(tts_pipeline_lease_reference PROPERTIES TIMEOUT 120)
add_test(NAME stt_tts_pipeline_reference COMMAND test_stt_pipeline ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_tts_pipeline_cases.jsonl)
set_tests_properties(stt_tts_pipeline_reference PROPERTIES TIMEOUT 120)
add_executable(test_tts_audio_api tests/test_tts_audio_api.cpp)
target_link_libraries(test_tts_audio_api PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_audio_api_reference COMMAND test_tts_audio_api ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_audio_api_cases.jsonl)
set_tests_properties(tts_audio_api_reference PROPERTIES TIMEOUT 120)
add_executable(test_voice_files tests/test_voice_files.cpp)
target_link_libraries(test_voice_files PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME voice_files_reference COMMAND test_voice_files ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/voice_files_cases.jsonl)
set_tests_properties(voice_files_reference PROPERTIES TIMEOUT 120)
add_executable(test_voice_remote tests/test_voice_remote.cpp)
target_link_libraries(test_voice_remote PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME voice_remote_reference COMMAND test_voice_remote ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/voice_remote_cases.jsonl)
set_tests_properties(voice_remote_reference PROPERTIES TIMEOUT 120)
add_executable(test_voice_lifetime tests/test_voice_lifetime.cpp)
target_link_libraries(test_voice_lifetime PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME voice_lifetime_reference COMMAND test_voice_lifetime ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/voice_lifetime_cases.jsonl.gz)
set_tests_properties(voice_lifetime_reference PROPERTIES TIMEOUT 120)
add_executable(test_gtts tests/test_gtts.cpp)
add_executable(test_tts_api tests/test_tts_api.cpp)
add_executable(test_tts_routing tests/test_tts_routing.cpp)
add_executable(test_tts_websocket tests/test_tts_websocket.cpp)
target_link_libraries(test_tts_websocket PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_websocket_reference COMMAND test_tts_websocket ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_websocket_cases.jsonl.gz)
add_test(NAME tts_websocket_state COMMAND test_tts_websocket --state-notifications)
set_tests_properties(tts_websocket_state PROPERTIES TIMEOUT 30)
add_executable(test_process_capture tests/test_process_capture.cpp)
target_include_directories(test_process_capture PRIVATE include)
target_link_libraries(test_process_capture PRIVATE Threads::Threads)
add_test(NAME process_capture COMMAND test_process_capture)
set_tests_properties(process_capture PROPERTIES TIMEOUT 30)
set_tests_properties(tts_websocket_reference PROPERTIES TIMEOUT 120)
add_executable(test_tts_ws_lease tests/test_tts_ws_lease.cpp)
target_link_libraries(test_tts_ws_lease PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_ws_lease_reference COMMAND test_tts_ws_lease ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_ws_lease_cases.jsonl.gz)
set_tests_properties(tts_ws_lease_reference PROPERTIES TIMEOUT 120)
add_executable(test_tts_ws_api tests/test_tts_ws_api.cpp)
target_link_libraries(test_tts_ws_api PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_ws_api_reference COMMAND test_tts_ws_api ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_ws_api_cases.jsonl)
set_tests_properties(tts_ws_api_reference PROPERTIES TIMEOUT 120)
add_executable(test_tts_ws_session tests/test_tts_ws_session.cpp)
target_link_libraries(test_tts_ws_session PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_ws_session_reference COMMAND test_tts_ws_session ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_ws_session_cases.jsonl)
set_tests_properties(tts_ws_session_reference PROPERTIES TIMEOUT 60)
add_executable(test_tts_ws_transport tests/test_tts_ws_transport.cpp)
target_link_libraries(test_tts_ws_transport PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_ws_transport_reference COMMAND test_tts_ws_transport ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_ws_transport_cases.jsonl)
set_tests_properties(tts_ws_transport_reference PROPERTIES TIMEOUT 60)
target_link_libraries(test_tts_routing PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_routing_reference COMMAND test_tts_routing ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_routing_cases.jsonl)
set_tests_properties(tts_routing_reference PROPERTIES TIMEOUT 120)
add_executable(test_tts_remote tests/test_tts_remote.cpp)
target_link_libraries(test_tts_remote PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_remote_reference COMMAND test_tts_remote ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_remote_cases.jsonl)
set_tests_properties(tts_remote_reference PROPERTIES TIMEOUT 120)
target_link_libraries(test_tts_api PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_api_reference COMMAND test_tts_api ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_api_cases.jsonl)
set_tests_properties(tts_api_reference PROPERTIES TIMEOUT 120)
add_executable(test_tts_api_process tests/test_tts_api_process.cpp)
target_link_libraries(test_tts_api_process PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_dependencies(test_tts_api_process test_tts_process_fixture)
add_test(NAME tts_api_process COMMAND test_tts_api_process $<TARGET_FILE:test_tts_process_fixture>)
set_tests_properties(tts_api_process PROPERTIES TIMEOUT 60)
target_link_libraries(test_gtts PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME gtts_reference COMMAND test_gtts ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gtts_cases.jsonl.gz)
set_tests_properties(gtts_reference PROPERTIES TIMEOUT 120)
add_executable(test_gtts_http tests/test_gtts_http.cpp)
target_link_libraries(test_gtts_http PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME gtts_http_reference COMMAND test_gtts_http ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gtts_http_cases.jsonl ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/gtts_process_cases.jsonl)
set_tests_properties(gtts_http_reference PROPERTIES TIMEOUT 120)
target_link_libraries(test_tts_service PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_test(NAME tts_service_reference COMMAND test_tts_service ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_service_cases.jsonl.gz)
set_tests_properties(tts_service_reference PROPERTIES TIMEOUT 120)

add_executable(test_tts_service_process tests/test_tts_service_process.cpp)
target_link_libraries(test_tts_service_process PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
add_dependencies(test_tts_service_process test_tts_process_fixture)
add_test(NAME tts_service_process COMMAND test_tts_service_process $<TARGET_FILE:test_tts_process_fixture>)
set_tests_properties(tts_service_process PROPERTIES TIMEOUT 60)

if(TARGET ezlocalai-tts AND EZLOCALAI_TTS_MODEL_DIRECTORY)
    add_executable(test_tts_service_model tests/test_tts_service_model.cpp)
    add_executable(test_tts_api_model tests/test_tts_api_model.cpp)
    target_link_libraries(test_tts_api_model PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
    add_dependencies(test_tts_api_model ezlocalai-native ezlocalai-tts)
    set(EZLOCALAI_TTS_API_STT_ARGS "")
    if(TARGET test_stt_api_main)
        set(EZLOCALAI_TTS_API_STT_ARGS ${EZLOCALAI_WHISPER_CT2_MODEL} ${EZLOCALAI_WHISPER_VAD_MODERN_MODELS})
    endif()
    add_test(NAME tts_api_model_reference COMMAND test_tts_api_model $<TARGET_FILE:ezlocalai-native> ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_service_model_cases.jsonl $<TARGET_FILE:ezlocalai-tts> ${EZLOCALAI_TTS_MODEL_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_runtime_assets/speech.wav ${EZLOCALAI_TTS_API_STT_ARGS})
    set_tests_properties(tts_api_model_reference PROPERTIES TIMEOUT 600 RESOURCE_LOCK stt_torch_gpu ENVIRONMENT "LD_LIBRARY_PATH=${EZLOCALAI_STT_CUDA_LIBRARY_PATH}")
    target_link_libraries(test_tts_service_model PRIVATE ezlocalai_native nlohmann_json::nlohmann_json)
    if(TARGET ezlocalai_ctts_torch)
        target_link_libraries(test_tts_service_model PRIVATE ezlocalai_ctts_torch)
        target_compile_definitions(test_tts_service_model PRIVATE EZLOCALAI_CTTS_TORCH)
    endif()
    add_dependencies(test_tts_service_model ezlocalai-tts)
    add_test(NAME tts_service_model_reference COMMAND test_tts_service_model ${EZLOCALAI_REFERENCE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tts_service_model_cases.jsonl $<TARGET_FILE:ezlocalai-tts> ${EZLOCALAI_TTS_MODEL_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/stt_runtime_assets/speech.wav)
    set_tests_properties(tts_service_model_reference PROPERTIES TIMEOUT 300 RESOURCE_LOCK stt_torch_gpu ENVIRONMENT "LD_LIBRARY_PATH=${EZLOCALAI_STT_CUDA_LIBRARY_PATH}")
endif()

# These tests retain the NVIDIA device adapter used by their captured Python
# references, even in a HIP build. HIP device/reporting/launch tests above
# explicitly exercise the AMD adapter instead; the production library is HIP.
if(EZLOCALAI_USE_HIP)
    add_library(cuda_reference_runtime OBJECT src/video_runtime.cpp)
    target_link_libraries(cuda_reference_runtime PRIVATE ezlocalai_native)
    foreach(reference_target IN ITEMS test_worker_metadata test_ltx_runtime
            test_video_runtime test_embedding_placement test_model_download
            test_resource_manager test_media_startup test_llm_batch
            test_llm_cache test_llm_speculative test_llm_parameters)
        target_sources(${reference_target} PRIVATE $<TARGET_OBJECTS:cuda_reference_runtime>)
    endforeach()
endif()

# The images build these existing targets together so independent tests
# compile in parallel instead of serializing separate Make target invocations.
if(TARGET test_ltx_norm AND TARGET ezlocalai-tts)
    set(native_package_targets
        ezlocalai-native
        ezlocalai-native-entrypoint
        llama_cpp_tool
        ezlocalai-tts
        test_storyboard
        test_ltx_norm
        test_ltx_timesteps
        test_ltx_schedule
        test_ltx_rng
        test_nf4
        test_nf4_mean
        test_nf4_storage
        test_ltx_condition_cache
        test_ltx_nf4_loader
        test_torch_checkpoint
        test_wakeword_cache
        test_wakeword_model
        test_wakeword_jobs
        test_wakeword_qwen
        test_speech_resources
        test_speech_slots
        test_worker_metadata
        test_file_response
        test_native_http
        test_runtime_version
        test_media_slots
        test_video_auxiliary
        test_embedding_discovery
        test_queue_history
        test_queue_http
        test_dotenv
        test_dotenv_startup
        test_cuda_mps
        test_openapi
        test_router_fallback
        test_worker_heartbeat
        test_worker_heartbeat_http
        test_worker_tunnel
        test_voice_conversion test_realtime test_tunnel_connection
        test_tunnel_redirects
        test_http_cookies
        test_tunnel_transport
        test_tunnel_idle
        test_stt_diarization
        test_stt_diarization_model
        test_tunnel_framing
        test_service_clients
        test_llm_residency
        test_llm_cache
        test_llm_input
        test_llm_headroom
        test_cuda_allocator
        test_torch_resources
        test_llm_preload
        test_live_capacity
        test_llm_instances
        test_startup
        test_media_startup
        test_media_engine
        test_ltx_runtime
        test_core
        test_http_stream_socket
        test_request_gzip_codec
        test_request_gzip_http
        test_stream_completion
        test_tts_websocket
        test_process_capture
    )
    if(TARGET test_conversation_cache)
        list(APPEND native_package_targets test_conversation_cache test_cache_restore_atomic
            test_cache_storage test_llama_mtp_resume test_llama_checkpoint_coverage test_managed_llama_patches)
    endif()
    if(TARGET test_q4_kv_layout)
        list(APPEND native_package_targets test_q4_kv_layout)
    endif()
    if(TARGET meanvc2_engine_check)
        list(APPEND native_package_targets meanvc2_engine_check)
    endif()
    add_custom_target(cpu_package DEPENDS ${native_package_targets})
    if(EZLOCALAI_USE_HIP)
        add_custom_target(rocm_package DEPENDS ${native_package_targets}
            test_hip_abi test_hip_runtime test_llm_speculative_hip test_worker_metadata_hip test_stt_torch)
    endif()
    if(TARGET test_ltx_gemma_residency_cuda)
        add_custom_target(cuda_package DEPENDS ${native_package_targets}
            test_nf4_storage_cuda test_ltx_gemma_residency_cuda test_ltx_connectors test_cuda_architectures)
    endif()
endif()

include(cmake/ComposeGate.cmake)
