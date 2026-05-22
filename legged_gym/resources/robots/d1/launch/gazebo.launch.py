import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command

def generate_launch_description():
    package_name = 'd1_description' 
    pkg_path = get_package_share_directory(package_name)

    # ========================================================
    # 1. 启动参数
    # ========================================================
    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='true', description='Use simulation (Gazebo) clock if true')
    
    gui = LaunchConfiguration('gui')
    declare_gui_cmd = DeclareLaunchArgument(
        'gui', default_value='true', description='Set to "false" to run Gazebo headless')

    # ========================================================
    # 2. XACRO/URDF 处理
    # ========================================================
    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    # 使用 Command 方式处理 XACRO 文件
    robot_description_config = Command(['xacro ', xacro_file])
    
    # ========================================================
    # 3. 核心 ROS 2 节点
    # ========================================================
    
    # 3.1 机器人状态发布节点
    # robot_state_publisher_node = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     output='screen',
    #     parameters=[
    #         {'robot_description': robot_description_config},
    #         {'use_sim_time': use_sim_time}
    #     ]
    # )

    # 3.2 Gazebo 启动
    gazebo_launch_file = os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={'world': 'empty.world', 'gui': gui, 'verbose': 'false'}.items() 
    )
    
    # 3.3 模型注入节点
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', 
                   '-entity', 'legged_dm'],
        output='screen'
    )

    # 3.4 控制器 Spawner (使用默认 spawner)
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
        output='screen'
    )
    
    robot_joint_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["robot_joint_controller", "-c", "/controller_manager"],
        output='screen'
    )
    
    # ========================================================
    # 4. Launch 描述 (并行启动)
    # ========================================================
    return LaunchDescription([
        # 声明参数
        declare_use_sim_time_cmd,
        declare_gui_cmd,

        # 核心节点
        # robot_state_publisher_node,
        gazebo,
        spawn_entity,
        joint_state_broadcaster_spawner,
        robot_joint_controller_spawner,
    ])
