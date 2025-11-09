from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():
    pkg_share_dir = get_package_share_directory('stereo_usb_cam_setup')
    params_file_path = os.path.join(pkg_share_dir, 'config', 'bringup_recording_config.yaml')

    with open(params_file_path, 'r') as f:
        full_params = yaml.safe_load(f)

    def get_node_params(node_name):
        node = full_params.get(node_name, {}) or {}
        params = node.get('ros__parameters', {}) or {}
        return params

    # Global args
    global_args = full_params.get('global_args', {})
    width_arg = DeclareLaunchArgument('width', default_value=TextSubstitution(text=str(global_args.get('width', 1280))))
    height_arg = DeclareLaunchArgument('height', default_value=TextSubstitution(text=str(global_args.get('height', 720))))
    fps_arg = DeclareLaunchArgument('fps', default_value=TextSubstitution(text=str(global_args.get('fps', 30))))
    align_depth_arg = DeclareLaunchArgument('align_depth', default_value=TextSubstitution(text=str(global_args.get('align_depth', True))))

    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')
    fps = LaunchConfiguration('fps')
    align_depth = LaunchConfiguration('align_depth')

    # RealSense include
    realsense_pkg_dir = get_package_share_directory('realsense2_camera')
    realsense_launch_file = os.path.join(realsense_pkg_dir, 'launch', 'rs_launch.py')

    realsense_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_file),
        launch_arguments={
            'rgb_camera.color_profile': [width, 'x', height, 'x', fps],
            'align_depth.enable': align_depth,
            'enable_color': 'true',
            'enable_depth': 'true',
        }.items()
    )

    # Dual USB camera node
    dual_camera_node = Node(
        package='stereo_usb_cam_setup',
        executable='dual_camera_node',
        name='dual_camera_publisher',
        output='screen',
        parameters=[get_node_params('dual_camera_node')]
    )

    # Dataset recorder node
    dataset_recorder = Node(
        package='stereo_usb_cam_setup',
        executable='dataset_recorder_node',
        name='dataset_recorder_node',
        output='screen',
        parameters=[get_node_params('dataset_recorder_node')]
    )

    return LaunchDescription([
        # width_arg,
        # height_arg,
        # fps_arg,
        # align_depth_arg,
        # realsense_include,
        # dual_camera_node,
        dataset_recorder,
    ])
