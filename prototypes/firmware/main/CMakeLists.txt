# Register the main component
idf_component_register(SRCS "app_main.cpp"
                       PRIV_REQUIRES nvs_flash esp-dl)

# ------------------------------------------------------------------
# Import esp-dl Utilities
# ------------------------------------------------------------------

# 1. Find the path where esp-dl is installed (whether local or managed)
idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
else()
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()

# 2. Include the CMake utility file that defines "target_add_aligned_binary_data"
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# ------------------------------------------------------------------
# Embed the Model
# ------------------------------------------------------------------
target_add_aligned_binary_data(${COMPONENT_LIB} "swish_lut_test.espdl" BINARY)