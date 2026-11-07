#
# SPDX-FileCopyrightText: René Widera
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.25)

alpaka_install_catch2()
function(alpakaV_add_test_target target_prefix folder)
    file(GLOB_RECURSE test_sources CONFIGURE_DEPENDS "${folder}/*.cpp")
    foreach(source_file ${test_sources})
        get_filename_component(source_file_name ${source_file} NAME_WE)
        get_filename_component(test_folder_name ${source_file} DIRECTORY)
        get_filename_component(test_folder_name ${test_folder_name} NAME)

        set(test_target_name "${target_prefix}${test_folder_name}_${source_file_name}")
        add_executable(${test_target_name} ${source_file})
        target_link_libraries(
            ${test_target_name}
            PRIVATE Catch2::Catch2WithMain alpakaVendor::alpakaVendor alpaka::alpaka
        )
        target_include_directories(
            ${test_target_name}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        if(NOT alpakaV_DEP_FFTW AND NOT alpakaV_DEP_OPENBLAS)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Host_Cpu ALPAKA_DISABLE_Host_NumaCpu)
        endif()
        if(NOT alpakaV_DEP_CUFFT)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Cuda_NvidiaGpu)
        endif()
        if(NOT alpakaV_DEP_CUBLAS)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Cuda_NvidiaGpu)
        endif()
        if(NOT alpakaV_DEP_ROCFFT)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Hip_AmdGpu)
        endif()
        if(NOT alpakaV_DEP_ROCBLAS)
            target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Hip_AmdGpu)
        endif()
        if(NOT alpakaV_DEP_ONEMKL)
            target_compile_definitions(
                ${test_target_name}
                PRIVATE
                    ALPAKA_DISABLE_OneApi_Cpu
                    ALPAKA_DISABLE_OneApi_IntelGpu
                    ALPAKA_DISABLE_OneApi_NvidiaGpu
                    ALPAKA_DISABLE_OneApi_AmdGpu
            )
        endif()
        # ThreadSanitizer needs low ASLR entropy. A CI job container cannot change the host
        # `/proc/sys/vm/mmap_rnd_bits`, so disable ASLR per process for the discovery run and every
        # ctest invocation via Catch2's test executor. `setarch -R` only affects this process tree.
        if(alpaka_TSAN AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
            find_program(ALPAKAV_SETARCH_EXECUTABLE setarch)
            if(ALPAKAV_SETARCH_EXECUTABLE)
                set_target_properties(
                    ${test_target_name}
                    PROPERTIES CROSSCOMPILING_EMULATOR "${ALPAKAV_SETARCH_EXECUTABLE};${CMAKE_SYSTEM_PROCESSOR};-R"
                )
            else()
                message(WARNING "TSAN is enabled but setarch was not found; tests may fail with high ASLR entropy")
            endif()
        endif()
        alpaka_finalize(${test_target_name})
        catch_discover_tests(${test_target_name})
    endforeach()
endfunction()

# The const-element output contract of the BLAS writers is a negative compile-time guarantee: the guarded half of
# constOutputGuards.cpp must NOT compile (dev-identical rejection, see the file header). A dedicated target compiles
# that translation unit with ALPAKAV_BLAS_TEST_NEGATIVE_CONST_OUTPUT defined and is expected to fail, so ctest
# enforces the const-output regression class machine-checked (e.g. a scal accepting a const output would silently
# "fix" the compile and fail the WILL_FAIL test below).
function(alpakaV_add_negative_const_output_check)
    set(test_target_name "unit_test_blas_constOutputGuards_neg")
    add_executable(${test_target_name} EXCLUDE_FROM_ALL "${CMAKE_CURRENT_SOURCE_DIR}/unit/blas/constOutputGuards.cpp")
    target_link_libraries(${test_target_name} PRIVATE Catch2::Catch2WithMain alpakaVendor::alpakaVendor alpaka::alpaka)
    target_include_directories(
        ${test_target_name}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(${test_target_name} PRIVATE ALPAKAV_BLAS_TEST_NEGATIVE_CONST_OUTPUT)
    if(NOT alpakaV_DEP_FFTW AND NOT alpakaV_DEP_OPENBLAS)
        target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Host_Cpu ALPAKA_DISABLE_Host_NumaCpu)
    endif()
    if(NOT alpakaV_DEP_CUFFT)
        target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Cuda_NvidiaGpu)
    endif()
    if(NOT alpakaV_DEP_CUBLAS)
        target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Cuda_NvidiaGpu)
    endif()
    if(NOT alpakaV_DEP_ROCFFT)
        target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Hip_AmdGpu)
    endif()
    if(NOT alpakaV_DEP_ROCBLAS)
        target_compile_definitions(${test_target_name} PRIVATE ALPAKA_DISABLE_Hip_AmdGpu)
    endif()
    if(NOT alpakaV_DEP_ONEMKL)
        target_compile_definitions(
            ${test_target_name}
            PRIVATE
                ALPAKA_DISABLE_OneApi_Cpu
                ALPAKA_DISABLE_OneApi_IntelGpu
                ALPAKA_DISABLE_OneApi_NvidiaGpu
                ALPAKA_DISABLE_OneApi_AmdGpu
        )
    endif()
    alpaka_finalize(${test_target_name})
    # The compile is expected to fail: WILL_FAIL turns the build failure into a passing test, so a regression that
    # makes a const output form flips the result to "failed" and the CI turns red.
    add_test(
        NAME blas_const_output_negative_compile
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${test_target_name} --config $<CONFIG>
    )
    set_tests_properties(blas_const_output_negative_compile PROPERTIES WILL_FAIL TRUE)
endfunction()

# Regression guard against re-introducing bare 32-bit narrowing on any BLAS descriptor value in the backend
# dispatches under include/alpaka/blas/internal/api/. Every descriptor value must go through
# checkedVendorInt/checkedCast; a bare `int(xd.n)`, `static_cast<int>(ad.ld)` or `static_cast<int32_t>(n)` silently
# truncates (and on a non-blocking host queue hides the failure in a discarded future). The guard scans the backend
# headers line by line and flags every C-style/`static_cast` narrowing to `int`/`int32_t`/`unsigned int`/`signed
# int`, except the explicit allow-listed error-enum formatting `int(status)` in the host/cuda/hip error helpers,
# which does not narrow a descriptor. The offending file and line are reported.
function(alpakaV_add_blas_descriptor_narrowing_guard)
    set(guard_script "${CMAKE_CURRENT_BINARY_DIR}/check_blas_descriptor_narrowing.cmake")
    file(
        WRITE
        "${guard_script}"
        [==[
# Generated by test/CMakeLists.txt; scans the checked-in backend headers for bare int/int32_t narrowing.
# ALPAKAV_BLAS_GUARD_ROOT may point the scan at an alternate header tree (used by the negative fixture below).
if(NOT DEFINED ALPAKAV_BLAS_GUARD_ROOT)
    set(ALPAKAV_BLAS_GUARD_ROOT "${ALPAKAV_SOURCE_DIR}/include/alpaka/blas/internal/api")
endif()
file(GLOB_RECURSE headers "${ALPAKAV_BLAS_GUARD_ROOT}/*.hpp")
if(NOT headers)
    message(FATAL_ERROR "No BLAS backend headers found under ${ALPAKAV_BLAS_GUARD_ROOT}")
endif()
set(violations "")
foreach(header ${headers})
    file(READ "${header}" content)
    # Only `int(status)` (the vendor error-enum formatting in the host/cuda/hip error helpers) is allowed; strip it
    # before scanning so the allow-list is explicit and local.
    string(REGEX REPLACE "int[ \t]*\\([ \t]*status[ \t]*\\)" "" content "${content}")
    # Scan physical lines one at a time instead of splitting into a CMake list: list semantics collapse empty
    # elements (and would also treat ';' in a source line as a separator), so a list-based loop reports the
    # non-empty-line ordinal rather than the real file line number. Semicolons are ordinary characters here.
    set(remaining "${content}")
    set(lineNo 0)
    while(NOT remaining STREQUAL "")
        string(FIND "${remaining}" "\n" nlPos)
        if(nlPos EQUAL -1)
            set(line "${remaining}")
            set(remaining "")
        else()
            string(SUBSTRING "${remaining}" 0 ${nlPos} line)
            math(EXPR next "${nlPos} + 1")
            string(SUBSTRING "${remaining}" ${next} -1 remaining)
        endif()
        math(EXPR lineNo "${lineNo} + 1")
        set(found "")
        if(line MATCHES "static_cast[ \t]*<[ \t]*int[ \t]*>")
            set(found "static_cast<int>")
        elseif(line MATCHES "static_cast[ \t]*<[ \t]*(std::)?int32_t[ \t]*>")
            set(found "static_cast<int32_t>")
        elseif(line MATCHES "static_cast[ \t]*<[ \t]*(unsigned|signed)[ \t]+int[ \t]*>")
            set(found "static_cast<unsigned/signed int>")
        elseif(line MATCHES "\\([ \t]*(std::)?int[ \t]*\\)")
            # C-style cast `(int)x` (the char after `int` is `)`, not `(`, so the function-call pattern below
            # cannot match it).
            set(found "(int) C-style cast")
        elseif(line MATCHES "\\([ \t]*(std::)?int32_t[ \t]*\\)")
            # C-style cast `(int32_t)x` / `(std::int32_t)x`.
            set(found "(int32_t) C-style cast")
        elseif(line MATCHES "\\([ \t]*(unsigned|signed)[ \t]+int[ \t]*\\)")
            set(found "(unsigned/signed int) C-style cast")
        elseif(line MATCHES "^int[ \t]*\\(" OR line MATCHES "[^A-Za-z0-9_]int[ \t]*\\(")
            set(found "int(")
        elseif(line MATCHES "^int32_t[ \t]*\\(" OR line MATCHES "[^A-Za-z0-9_]int32_t[ \t]*\\(")
            # Functional cast `int32_t(x)`; `int32_t` must be followed by `(` (modulo blanks) so a declaration like
            # `int32_t foo(...)` is not misflagged. `std::int32_t(...)` also matches here (the char before `int32_t`
            # is `:`, which is outside `[A-Za-z0-9_]`).
            set(found "int32_t(")
        endif()
        if(NOT found STREQUAL "")
            set(violations "${violations}\n  ${header}:${lineNo}: ${found} in: ${line}")
        endif()
    endwhile()
endforeach()
if(NOT violations STREQUAL "")
    message(FATAL_ERROR "Bare 32-bit narrowing to int/int32_t in a BLAS backend header (use checkedVendorInt/checkedCast):${violations}")
endif()
message(STATUS "blas descriptor narrowing guard: no bare int()/static_cast<int> found")
]==]
    )
    add_test(
        NAME blas_descriptor_narrowing_guard
        COMMAND ${CMAKE_COMMAND} -DALPAKAV_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/.. -P "${guard_script}"
    )
endfunction()

# Negative fixtures proving the descriptor-narrowing guard actually fires. The guard script is reused with
# ALPAKAV_BLAS_GUARD_ROOT pointed at a generated temp header tree, so a regression that stops detecting a
# `static_cast<int>(...)`, a C-style `(int)` narrowing, or an `int32_t(...)`/`(int32_t)` narrowing flips the
# matching WILL_FAIL test to "passed" and turns CI red.
function(alpakaV_add_blas_descriptor_narrowing_negative_fixtures)
    set(guard_script "${CMAKE_CURRENT_BINARY_DIR}/check_blas_descriptor_narrowing.cmake")
    set(fixture_root "${CMAKE_CURRENT_BINARY_DIR}/blas_narrowing_fixtures")
    file(MAKE_DIRECTORY "${fixture_root}/static_cast_int/include/alpaka/blas/internal/api")
    file(
        WRITE
        "${fixture_root}/static_cast_int/include/alpaka/blas/internal/api/fixture.hpp"
        [==[
// Generated negative fixture: a bare static-cast narrowing of a descriptor field must be flagged.
void fixture(auto const& xd)
{
    auto const n = static_cast<int>(xd.n);
    (void)n;
}
]==]
    )
    add_test(
        NAME blas_descriptor_narrowing_guard_static_cast_int
        COMMAND
            ${CMAKE_COMMAND} -DALPAKAV_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
            -DALPAKAV_BLAS_GUARD_ROOT=${fixture_root}/static_cast_int/include/alpaka/blas/internal/api -P
            "${guard_script}"
    )
    set_tests_properties(blas_descriptor_narrowing_guard_static_cast_int PROPERTIES WILL_FAIL TRUE)

    file(MAKE_DIRECTORY "${fixture_root}/c_style_int/include/alpaka/blas/internal/api")
    file(
        WRITE
        "${fixture_root}/c_style_int/include/alpaka/blas/internal/api/fixture.hpp"
        [==[
// Generated negative fixture: a C-style narrowing cast of a descriptor field must be flagged.
void fixture(auto const& xd)
{
    auto const n = (int)xd.n;
    (void)n;
}
]==]
    )
    add_test(
        NAME blas_descriptor_narrowing_guard_c_style_int
        COMMAND
            ${CMAKE_COMMAND} -DALPAKAV_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
            -DALPAKAV_BLAS_GUARD_ROOT=${fixture_root}/c_style_int/include/alpaka/blas/internal/api -P "${guard_script}"
    )
    set_tests_properties(blas_descriptor_narrowing_guard_c_style_int PROPERTIES WILL_FAIL TRUE)

    # Third fixture: the two int32_t spellings the first two fixtures do not cover -- functional `int32_t(x)` and
    # C-style `(int32_t)x`. Both are emitted so the guard is proven to flag either form independently.
    file(MAKE_DIRECTORY "${fixture_root}/int32_t_casts/include/alpaka/blas/internal/api")
    file(
        WRITE
        "${fixture_root}/int32_t_casts/include/alpaka/blas/internal/api/fixture.hpp"
        [==[
// Generated negative fixture: functional and C-style int32_t narrowing of a descriptor field must be flagged,
// including the std::int32_t spellings.
void fixture(auto const& xd)
{
    auto const n = int32_t(xd.n);
    auto const m = (int32_t)xd.n;
    auto const s = std::int32_t(xd.n);
    auto const t = (std::int32_t)xd.n;
    (void)n;
    (void)m;
    (void)s;
    (void)t;
}
]==]
    )
    add_test(
        NAME blas_descriptor_narrowing_guard_int32_t_casts
        COMMAND
            ${CMAKE_COMMAND} -DALPAKAV_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
            -DALPAKAV_BLAS_GUARD_ROOT=${fixture_root}/int32_t_casts/include/alpaka/blas/internal/api -P
            "${guard_script}"
    )
    set_tests_properties(blas_descriptor_narrowing_guard_int32_t_casts PROPERTIES WILL_FAIL TRUE)
endfunction()

alpakav_add_test_target(unit_test_ ${CMAKE_CURRENT_SOURCE_DIR}/unit)
alpakav_add_test_target(integr_test_ ${CMAKE_CURRENT_SOURCE_DIR}/integr)
if(alpakaV_TESTS)
    alpakav_add_negative_const_output_check()
    alpakav_add_blas_descriptor_narrowing_guard()
    alpakav_add_blas_descriptor_narrowing_negative_fixtures()
endif()
