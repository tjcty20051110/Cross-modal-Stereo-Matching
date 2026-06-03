# ==============================================================================
# Core Library: cms_core
# ==============================================================================

set(CMS_CORE_SOURCES
    config/config_manager.cpp
    data/data_loader.cpp
    data/manifest_splitter.cpp
    preprocessing/preprocessor.cpp
    features/feature_extractor.cpp
    cost/cost_volume_builder.cpp
    aggregation/cost_aggregator.cpp
    disparity/disparity_estimator.cpp
    stereo/stereo_matcher.cpp
    evaluation/evaluator.cpp
    visualization/visualizer.cpp
    inference/dl_inference_engine.cpp
    cli/cli.cpp
)

if(CMS_ENABLE_LIBTORCH AND Torch_FOUND)
    list(APPEND CMS_CORE_SOURCES
        training/siamese_network.cpp
        training/trainer.cpp
        cfm/cfm_network.cpp
        cfm/cfm_attention.cpp
        cfm/cfm_checkpoint_meta.cpp
        cfm/cfm_mode_resolver.cpp
        cfm/mdp_backbone.cpp
        cfm/backbone_exporter.cpp
        cfm/cfm_trainer.cpp
        cfm/cfm_inference.cpp
    )
endif()

add_library(cms_core STATIC ${CMS_CORE_SOURCES})

target_include_directories(cms_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(cms_core
    PUBLIC
        ${OpenCV_LIBS}
        yaml-cpp::yaml-cpp
)

target_include_directories(cms_core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})

if(CMS_ENABLE_ONNXRUNTIME AND TARGET onnxruntime::onnxruntime)
    target_link_libraries(cms_core PUBLIC onnxruntime::onnxruntime)
endif()

if(CMS_ENABLE_LIBTORCH AND TORCH_LIBRARIES)
    target_link_libraries(cms_core PUBLIC ${TORCH_LIBRARIES})
    target_include_directories(cms_core SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
endif()

set_target_properties(cms_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(cms main.cpp)
target_link_libraries(cms PRIVATE cms_core)

install(TARGETS cms RUNTIME DESTINATION bin)
