from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    # 获取当前 launch 文件所在目录
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    # 获取包根目录（launch 目录的父目录）
    package_dir = os.path.dirname(launch_dir)
    
    urdf_path = os.path.join(package_dir, 'urdf', 'd1_description.urdf')
    rviz_config_path = os.path.join(package_dir, 'rviz', 'rviz.rviz')

    # 读取 URDF 文件内容
    with open(urdf_path, 'r') as f:
        robot_description = f.read()
    
    # 机器人状态发布器
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description
        }],
        output='screen'
    )
    
    # 关节状态发布器 GUI
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )
    
    # RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )
    
    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node
    ])
