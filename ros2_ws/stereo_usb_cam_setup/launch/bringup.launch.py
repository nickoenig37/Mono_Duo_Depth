from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    # ---------------------- CONFIG LOADING ----------------------
    # Load the shared config YAML from this package
    pkg_share_dir = get_package_share_directory('stereo_usb_cam_setup')
    params_file_path = os.path.join(pkg_share_dir, 'config', 'bringup_recording_config.yaml')
    
    # Load parameters from the YAML file into a Python dictionary
    with open(params_file_path, 'r') as f:
        full_params = yaml.safe_load(f)

    # Extract global arguments for use in DeclareLaunchArgument defaults
    global_args = full_params.get('global_args', {})

    # Helper function to get node-specific parameters if introduced later
    def get_node_params(node_name):
        return full_params.get(node_name, {}).get('ros__parameters', {})

    # ---------------------- REALSENSE CAMERA ----------------------
    realsense_pkg_dir = get_package_share_directory('realsense2_camera')
    realsense_launch_file = os.path.join(realsense_pkg_dir, 'launch', 'rs_launch.py')
    align_depth_arg = DeclareLaunchArgument(
        'align_depth',
        default_value=TextSubstitution(text=str(global_args.get('align_depth', 'true'))),
        description='Whether to enable depth image alignment for RealSense camera.'
    )
    width_arg = DeclareLaunchArgument('width', default_value=TextSubstitution(text=str(global_args.get('width', 848))))
    height_arg = DeclareLaunchArgument('height', default_value=TextSubstitution(text=str(global_args.get('height', 480))))
    fps_arg = DeclareLaunchArgument('fps', default_value=TextSubstitution(text=str(global_args.get('fps', 30))))

    # Get LaunchConfiguration values AFTER declaring the arguments
    align_depth = LaunchConfiguration('align_depth')
    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')
    fps = LaunchConfiguration('fps')

    # ---------------------- DUAL USB CAMERAS ----------------------
    dual_camera_node = Node(
        package='stereo_usb_cam_setup',
        executable='dual_camera_node',
        name='dual_camera_publisher',
        output='screen',
        parameters=[get_node_params('dual_camera_node')]
    )

    return LaunchDescription([
        # ---------------------- REALSENSE LAUNCH ----------------------
        align_depth_arg,
        width_arg,
        height_arg,
        fps_arg,

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file),
            launch_arguments={
                'rgb_camera.color_profile': [width, 'x', height, 'x', fps],
                'align_depth.enable': align_depth,
                'enable_color': 'true',
                'enable_depth': 'true',
            }.items()
        ),

    # ---------------------- DUAL USB CAMERA NODE ----------------------
    dual_camera_node,
    ])