# Register the main component
    idf_component_register(SRCS "main.cpp"
                        PRIV_REQUIRES nvs_flash esp-dl yolo26)

    # Import esp-dl Utilities
    idf_build_get_property(component_targets __COMPONENT_TARGETS)
    if ("___idf_espressif__esp-dl" IN_LIST component_targets)
        idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
    else()
        idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
    endif()

    set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
    include(${cmake_dir}/utilities.cmake)

    # Embed Both Models (LUT + IDEAL)
    target_add_aligned_binary_data(${COMPONENT_LIB} "models/yolo26n_p4.espdl" BINARY)
    target_add_aligned_binary_data(${COMPONENT_LIB} "models/yolo26n_ideal_p4.espdl" BINARY)
    