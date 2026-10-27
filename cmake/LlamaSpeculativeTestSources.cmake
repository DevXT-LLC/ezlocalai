# Compile the real managed method/selector with an instrumented backend in the
# regression test. Deliberately fail if their surrounding source changes.
function(extract_speculative_test_source path begin end output)
    file(READ "${path}" source)
    string(FIND "${source}" "${begin}" first)
    if(first LESS 0)
        message(FATAL_ERROR "Missing speculative test source boundary: ${begin}")
    endif()
    string(SUBSTRING "${source}" ${first} -1 source)
    string(FIND "${source}" "${end}" last)
    if(last LESS 0)
        message(FATAL_ERROR "Missing speculative test source boundary: ${end}")
    endif()
    string(SUBSTRING "${source}" 0 ${last} source)
    string(REPLACE ") override {" ") {" source "${source}")
    file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${output}" CONTENT "${source}" @ONLY)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
endfunction()
file(READ "${EZLOCALAI_TTS_LLAMA_SOURCE}/common/speculative.cpp" mtp_source)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${EZLOCALAI_TTS_LLAMA_SOURCE}/common/speculative.cpp")
string(FIND "${mtp_source}" "struct common_speculative_impl_draft_mtp :" mtp_start)
if(mtp_start LESS 0)
    message(FATAL_ERROR "Missing actual MTP implementation")
endif()
string(SUBSTRING "${mtp_source}" ${mtp_start} -1 mtp_source)
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/llama_mtp_implementation_under_test.txt"
    CONTENT "${mtp_source}" @ONLY)
extract_speculative_test_source("${CMAKE_CURRENT_BINARY_DIR}/llama_mtp_implementation_under_test.txt"
    "    bool process(const common_batch & batch_in) override {" "    void draft("
    "llama_mtp_process_under_test.inc")
extract_speculative_test_source("${EZLOCALAI_TTS_LLAMA_SOURCE}/tools/server/server-context.cpp"
    "                const bool may_replay_checkpoint =" "                GGML_ASSERT(slot.spec_i_batch.size()"
    "llama_sampler_snapshot_under_test.inc")
