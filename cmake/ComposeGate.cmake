# Deployment configuration is checked on the host, where the Docker Compose
# client is available. Worker images do not need a Docker client or daemon.
find_package(nlohmann_json 3.11 REQUIRED)
find_package(Threads REQUIRED)
add_executable(test_compose_profiles tests/test_compose_profiles.cpp)
target_compile_features(test_compose_profiles PRIVATE cxx_std_17)
target_include_directories(test_compose_profiles PRIVATE include)
target_link_libraries(test_compose_profiles PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
find_program(EZLOCALAI_DOCKER docker)
if(EZLOCALAI_DOCKER)
    add_test(NAME compose_profiles COMMAND test_compose_profiles
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/compose_profiles.json
        ${EZLOCALAI_DOCKER})
    set_tests_properties(compose_profiles PROPERTIES TIMEOUT 60)
endif()
