cmake_minimum_required(VERSION 3.8)
project(d1_description)

# 检查编译器是否符合 C++14 标准
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(ament_lint_auto REQUIRED)
find_package(urdf REQUIRED)
#find_package(xacro REQUIRED)
find_package(robot_state_publisher REQUIRED)
# 安装所有资源文件（简化版本）
install(
  DIRECTORY config launch meshes urdf 
  DESTINATION share/${PROJECT_NAME}
)

# 导出依赖 - 添加 Gazebo 相关依赖
ament_export_dependencies(
  robot_state_publisher 
  joint_state_publisher_gui 
  rviz2
  gazebo_ros
)

ament_package()