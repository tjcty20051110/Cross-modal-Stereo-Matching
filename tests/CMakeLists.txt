# ==============================================================================
# Tests: unit (GoogleTest) + property (RapidCheck) + integration (GoogleTest)
# ==============================================================================

# ------------------------------------------------------------------------------
# GoogleTest
# ------------------------------------------------------------------------------
find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    # Fall back to the module-mode finder that ships with CMake.
    find_package(GTest REQUIRED)
endif()

# Normalize GoogleTest targets — different distros expose different names.
if(NOT TARGET GTest::gtest)
    if(TARGET GTest::GTest)
        add_library(GTest::gtest ALIAS GTest::GTest)
    endif()
endif()
if(NOT TARGET GTest::gtest_main)
    if(TARGET GTest::Main)
        add_library(GTest::gtest_main ALIAS GTest::Main)
    endif()
endif()

include(GoogleTest)

# ------------------------------------------------------------------------------
# RapidCheck (property-based testing)
# ------------------------------------------------------------------------------
# Sourcing strategy:
#   * If CMS_ENABLE_PBT_TESTS=ON we *always* fetch a pinned upstream revision
#     via FetchContent so the new cms_tests target has a reproducible build,
#     regardless of what the host system ships.
#   * Otherwise we try find_package first (for the legacy per-file property
#     tests under property/) and only fall back to FetchContent if the
#     system package is missing and the legacy suite is being built.
# RC_ENABLE_GTEST=ON is required either way so rapidcheck_gtest is created.
set(RC_ENABLE_GTEST ON CACHE BOOL "" FORCE)

if(CMS_ENABLE_PBT_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        rapidcheck
        GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
        GIT_TAG        a7db92ffc3d50cda3c7b05fc7a8ee06b08e2f7de  # pinned stable mid-2023 commit
    )
    FetchContent_MakeAvailable(rapidcheck)
else()
    find_package(rapidcheck CONFIG QUIET)
    if(NOT rapidcheck_FOUND AND CMS_BUILD_TESTS)
        include(FetchContent)
        FetchContent_Declare(
            rapidcheck
            GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
            GIT_TAG        ff6af6fc683159deb51c543b065eba14dfcf329b  # legacy fallback commit
        )
        FetchContent_MakeAvailable(rapidcheck)
    endif()
endif()

# Provide the rapidcheck target under a common alias (only if the target
# actually exists — it may be absent when PBT is off and no rapidcheck is
# installed, in which case only non-property suites are compiled).
if(NOT TARGET rapidcheck::rapidcheck AND TARGET rapidcheck)
    add_library(rapidcheck::rapidcheck ALIAS rapidcheck)
endif()

# ==============================================================================
# Legacy per-file test targets (pre-existing property/unit/integration suites).
# Built when CMS_BUILD_TESTS=ON (the default). Each source under unit/,
# property/, integration/ becomes its own executable via cms_add_test().
# ==============================================================================
if(CMS_BUILD_TESTS)
    # --------------------------------------------------------------------------
    # Helper: register a test executable
    # --------------------------------------------------------------------------
    function(cms_add_test name)
        set(options PROPERTY INTEGRATION)
        set(oneValueArgs "")
        set(multiValueArgs SOURCES)
        cmake_parse_arguments(CMS_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        if(NOT CMS_TEST_SOURCES)
            message(FATAL_ERROR "cms_add_test(${name}): SOURCES is required")
        endif()

        add_executable(${name} ${CMS_TEST_SOURCES})
        target_link_libraries(${name}
            PRIVATE
                cms_core
                GTest::gtest
                GTest::gtest_main
        )

        if(CMS_TEST_PROPERTY)
            target_link_libraries(${name} PRIVATE rapidcheck)
            if(TARGET rapidcheck_gtest)
                target_link_libraries(${name} PRIVATE rapidcheck_gtest)
            endif()
        endif()

        target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_compile_definitions(${name} PRIVATE CMS_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

        gtest_discover_tests(${name}
            PROPERTIES
                LABELS "$<IF:$<BOOL:${CMS_TEST_PROPERTY}>,property,$<IF:$<BOOL:${CMS_TEST_INTEGRATION}>,integration,unit>>"
            DISCOVERY_MODE PRE_TEST
        )
    endfunction()

    # Glob detection: if any test source exists, register it.
    file(GLOB CMS_UNIT_TESTS        CONFIGURE_DEPENDS "unit/*.cpp")
    file(GLOB CMS_PROPERTY_TESTS    CONFIGURE_DEPENDS "property/*.cpp")
    file(GLOB CMS_INTEGRATION_TESTS CONFIGURE_DEPENDS "integration/*.cpp")

    foreach(test_src IN LISTS CMS_UNIT_TESTS)
        get_filename_component(test_name ${test_src} NAME_WE)
        cms_add_test(${test_name} SOURCES ${test_src})
    endforeach()

    foreach(test_src IN LISTS CMS_PROPERTY_TESTS)
        get_filename_component(test_name ${test_src} NAME_WE)
        cms_add_test(${test_name} PROPERTY SOURCES ${test_src})
    endforeach()

    foreach(test_src IN LISTS CMS_INTEGRATION_TESTS)
        get_filename_component(test_name ${test_src} NAME_WE)
        cms_add_test(${test_name} INTEGRATION SOURCES ${test_src})
    endforeach()
endif()

# ==============================================================================
# Aggregated PBT test binary: cms_tests
# ------------------------------------------------------------------------------
# Task 9.1: new property-based tests added by later tasks land under
# tests/test_*.cpp (flat layout, one translation unit per test suite). They
# all compile into a single `cms_tests` executable gated by
# CMS_ENABLE_PBT_TESTS. The CMS_HAS_RAPIDCHECK compile definition is set so
# the test sources can `#ifdef` their rapidcheck-dependent bodies.
# ==============================================================================
if(CMS_ENABLE_PBT_TESTS)
    file(GLOB CMS_PBT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

    # Provide a placeholder TU so the target is valid before any
    # tests/test_*.cpp files have been written (tasks 9.2+ populate them).
    if(NOT CMS_PBT_TEST_SOURCES)
        set(_cms_tests_stub "${CMAKE_CURRENT_BINARY_DIR}/cms_tests_stub.cpp")
        file(WRITE "${_cms_tests_stub}"
            "// Auto-generated stub: no tests/test_*.cpp present yet.\n"
            "// Remove once real PBT sources exist.\n"
            "int main() { return 0; }\n")
        list(APPEND CMS_PBT_TEST_SOURCES "${_cms_tests_stub}")
        set(_cms_tests_has_real_sources FALSE)
    else()
        set(_cms_tests_has_real_sources TRUE)
    endif()

    add_executable(cms_tests ${CMS_PBT_TEST_SOURCES})
    target_link_libraries(cms_tests
        PRIVATE
            cms_core
            GTest::gtest
            GTest::gtest_main
            rapidcheck
    )
    if(TARGET rapidcheck_gtest)
        target_link_libraries(cms_tests PRIVATE rapidcheck_gtest)
    endif()

    target_include_directories(cms_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(cms_tests PRIVATE
        CMS_HAS_RAPIDCHECK
        CMS_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    if(_cms_tests_has_real_sources)
        gtest_discover_tests(cms_tests
            PROPERTIES LABELS "pbt"
            DISCOVERY_MODE PRE_TEST
        )
    endif()
endif()
