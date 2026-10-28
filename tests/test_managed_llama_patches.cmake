cmake_minimum_required(VERSION 3.19)
foreach(required SOURCE_DIR HELPER PATCH_DIR TEST_ROOT)
    if(NOT DEFINED ${required} OR NOT EXISTS "${${required}}")
        message(FATAL_ERROR "Missing patch-test input ${required}")
    endif()
endforeach()
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef nonce)
set(root "${TEST_ROOT}/managed-patch-test-${nonce}")
set(copy "${root}/source")
set(out "${root}/metadata")
set(pin 8642541b255bfb5dc39cd351a178be09013a55d6)
set(first "${PATCH_DIR}/llama-conversation-cache.patch")
file(MAKE_DIRECTORY "${root}" "${out}")
execute_process(COMMAND git clone --shared --no-checkout --quiet "${SOURCE_DIR}" "${copy}"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND git -C "${copy}" checkout --quiet --detach "${pin}"
    COMMAND_ERROR_IS_FATAL ANY)

function(run_helper)
    execute_process(COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${copy}" "-DBUILD_DIR=${out}"
        "-DPATCH_FILE=${first}" ${ARGN} -P "${HELPER}"
        OUTPUT_VARIABLE output ERROR_VARIABLE error RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "Managed patch helper failed: ${output}\n${error}")
    endif()
endfunction()
function(check_clean)
    execute_process(COMMAND git -C "${copy}" status --porcelain
        OUTPUT_VARIABLE dirty OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    if(NOT dirty STREQUAL "")
        message(FATAL_ERROR "Patch removal did not restore a pristine tree: ${dirty}")
    endif()
endfunction()
function(check_metadata)
    set(path "${out}/ezlocalai-llama-cpp-patches.txt")
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "Managed patch metadata missing")
    endif()
    file(STRINGS "${path}" rows)
    list(LENGTH rows count)
    if(NOT count EQUAL 9)
        message(FATAL_ERROR "The managed stack did not record all engine fixes")
    endif()
    set(seen "")
    foreach(row IN LISTS rows)
        if(NOT row MATCHES "^patch\\.([^=]+)=([0-9a-f]+)$")
            message(FATAL_ERROR "Invalid patch metadata row")
        endif()
        set(name "${CMAKE_MATCH_1}")
        set(recorded "${CMAKE_MATCH_2}")
        if(name IN_LIST seen OR NOT EXISTS "${PATCH_DIR}/${name}.patch")
            message(FATAL_ERROR "Unknown/duplicate patch metadata: ${name}")
        endif()
        list(APPEND seen "${name}")
        file(SHA256 "${PATCH_DIR}/${name}.patch" actual)
        if(NOT actual STREQUAL recorded)
            message(FATAL_ERROR "Incorrect patch fingerprint: ${name}")
        endif()
    endforeach()
endfunction()

# Pristine installation, then the overlapping stack must be idempotent without
# updating source mtimes and recompiling the engine on every configure.
run_helper()
check_metadata()
file(SHA256 "${copy}/tools/server/server-task.cpp" first_hash)
file(TIMESTAMP "${copy}/tools/server/server-task.cpp" first_time "%s" UTC)
file(READ "${out}/ezlocalai-llama-cpp-patches.txt" first_metadata)
execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1.1 COMMAND_ERROR_IS_FATAL ANY)
run_helper()
file(SHA256 "${copy}/tools/server/server-task.cpp" second_hash)
file(TIMESTAMP "${copy}/tools/server/server-task.cpp" second_time "%s" UTC)
file(READ "${out}/ezlocalai-llama-cpp-patches.txt" second_metadata)
if(NOT first_hash STREQUAL second_hash OR NOT first_time STREQUAL second_time OR
        NOT first_metadata STREQUAL second_metadata)
    message(FATAL_ERROR "Reapplying the stack modified source or metadata")
endif()
run_helper(-DREMOVE_MANAGED_PATCH=ON)
check_clean()
if(EXISTS "${out}/ezlocalai-llama-cpp-patches.txt")
    message(FATAL_ERROR "Removed patches left stale metadata")
endif()

# Upgrade an existing original conversation-cache installation, not just a new
# checkout. Removing a pristine installation must also remain idempotent.
run_helper(-DREMOVE_MANAGED_PATCH=ON)
check_clean()
execute_process(COMMAND git -C "${copy}" apply "${first}" COMMAND_ERROR_IS_FATAL ANY)
run_helper()
check_metadata()
run_helper(-DREMOVE_MANAGED_PATCH=ON)
check_clean()

# Upgrade the previous five-patch release while preserving cache/state fixes.
foreach(name llama-conversation-cache llama-cache-restore-atomic llama-cache-storage
        llama-mtp-resume llama-checkpoint-coverage)
    execute_process(COMMAND git -C "${copy}" apply "${PATCH_DIR}/${name}.patch"
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
run_helper()
check_metadata()
run_helper(-DREMOVE_MANAGED_PATCH=ON)
check_clean()

# Upgrade the preceding six-patch release, including its overlapping entry-window
# header fix, and remove the complete upgraded stack back to the original pin.
foreach(name llama-conversation-cache llama-cache-restore-atomic llama-cache-storage
        llama-mtp-resume llama-checkpoint-coverage llama-checkpoint-entry-window)
    execute_process(COMMAND git -C "${copy}" apply "${PATCH_DIR}/${name}.patch"
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
run_helper()
check_metadata()
run_helper(-DREMOVE_MANAGED_PATCH=ON)
check_clean()

# Removing an old six-patch installation directly must also peel the complete
# prefix without leaving the new helper or qualification metadata behind.
foreach(name llama-conversation-cache llama-cache-restore-atomic llama-cache-storage
        llama-mtp-resume llama-checkpoint-coverage llama-checkpoint-entry-window)
    execute_process(COMMAND git -C "${copy}" apply "${PATCH_DIR}/${name}.patch"
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
run_helper(-DREMOVE_MANAGED_PATCH=ON)
check_clean()
if(EXISTS "${out}/ezlocalai-llama-cpp-patches.txt")
    message(FATAL_ERROR "Removing the preceding stack left stale metadata")
endif()

# Upgrade the preceding eight-patch release, then remove the complete stack.
foreach(name llama-conversation-cache llama-cache-restore-atomic llama-cache-storage
        llama-mtp-resume llama-checkpoint-coverage llama-checkpoint-entry-window
        llama-cache-retention llama-cuda-q4-kv-fusion)
    execute_process(COMMAND git -C "${copy}" apply "${PATCH_DIR}/${name}.patch"
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
run_helper()
check_metadata()
run_helper(-DREMOVE_MANAGED_PATCH=ON)
check_clean()

# An eight-patch installation can also be removed directly without leaving
# qualification metadata behind.
foreach(name llama-conversation-cache llama-cache-restore-atomic llama-cache-storage
        llama-mtp-resume llama-checkpoint-coverage llama-checkpoint-entry-window
        llama-cache-retention llama-cuda-q4-kv-fusion)
    execute_process(COMMAND git -C "${copy}" apply "${PATCH_DIR}/${name}.patch"
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
run_helper(-DREMOVE_MANAGED_PATCH=ON)
check_clean()
if(EXISTS "${out}/ezlocalai-llama-cpp-patches.txt")
    message(FATAL_ERROR "Removing the eight-patch stack left stale metadata")
endif()

# A real v0.7 -> v0.8 update removes the previous pin's eight patches before
# git checkout, then installs all nine at the new pin. The old pin is accepted
# only for removal; it must never be advertised as a qualified new engine.
execute_process(COMMAND git -C "${copy}" checkout --quiet --detach
    620d2225cc436ab0b001e3200e590e3b286ef9ab COMMAND_ERROR_IS_FATAL ANY)
foreach(name llama-conversation-cache llama-cache-restore-atomic llama-cache-storage
        llama-mtp-resume llama-checkpoint-coverage llama-checkpoint-entry-window
        llama-cache-retention llama-cuda-q4-kv-fusion)
    execute_process(COMMAND git -C "${copy}" apply "${PATCH_DIR}/${name}.patch"
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()
run_helper(-DREMOVE_MANAGED_PATCH=ON)
check_clean()
if(EXISTS "${out}/ezlocalai-llama-cpp-patches.txt")
    message(FATAL_ERROR "Previous-pin removal left stale metadata")
endif()
run_helper()
check_clean()
execute_process(COMMAND git -C "${copy}" checkout --quiet --detach "${pin}"
    COMMAND_ERROR_IS_FATAL ANY)
run_helper()
check_metadata()
run_helper(-DREMOVE_MANAGED_PATCH=ON)
check_clean()

# Missing and malformed inputs must fail explicitly, not ship a partial worker.
execute_process(COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${copy}" "-DBUILD_DIR=${out}"
    "-DPATCH_FILES=${root}/missing.patch" -P "${HELPER}"
    RESULT_VARIABLE missing OUTPUT_QUIET ERROR_QUIET)
if(missing EQUAL 0)
    message(FATAL_ERROR "A missing managed patch was silently accepted")
endif()
file(WRITE "${root}/invalid.patch" "not a unified patch\n")
execute_process(COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${copy}" "-DBUILD_DIR=${out}"
    "-DPATCH_FILES=${root}/invalid.patch" -P "${HELPER}"
    RESULT_VARIABLE invalid OUTPUT_QUIET ERROR_QUIET)
if(invalid EQUAL 0)
    message(FATAL_ERROR "An invalid managed patch was silently accepted")
endif()
check_clean()

# Custom revisions must keep their own behavior and lose stale qualification
# metadata, even when the caller supplies the ordinary managed stack.
execute_process(COMMAND git -C "${copy}" -c user.name=PatchTest
    -c user.email=patch-test@example.invalid commit --quiet --allow-empty --no-verify
    -m "custom engine revision" COMMAND_ERROR_IS_FATAL ANY)
file(WRITE "${out}/ezlocalai-llama-cpp-patches.txt" "stale\n")
run_helper()
check_clean()
if(EXISTS "${out}/ezlocalai-llama-cpp-patches.txt")
    message(FATAL_ERROR "Custom revision retained managed qualification metadata")
endif()
file(REMOVE_RECURSE "${root}")
message(STATUS "Managed engine stack: pristine/prefix, idempotence, removal, metadata and failure checks passed")
