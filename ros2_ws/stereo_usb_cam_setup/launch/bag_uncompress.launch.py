from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml # Import the yaml library

def generate_launch_description():
    # Get package share directory
    pkg_bringup_share_dir = get_package_share_directory('stereo_usb_cam_setup') 

    # Path to your YAML parameter file
    params_file_path = os.path.join(pkg_bringup_share_dir, 'config', 'processing_config.yaml')

    # Load parameters from the YAML file into a Python dictionary
    # This is crucial for handling global_args and passing the dictionary to Nodes
    with open(params_file_path, 'r') as f:
        full_params = yaml.safe_load(f)

    # Extract global arguments for use in DeclareLaunchArgument defaults
    global_args = full_params.get('global_args', {})

    # Helper function to get node-specific parameters
    # This safely retrieves the ros__parameters for a given node, or an empty dict if not found
    def get_node_params(node_name):
        return full_params.get(node_name, {}).get('ros__parameters', {})


    return LaunchDescription([

        # ---------------------- Bag Playback and Image Decompression Nodes ----------------------
        Node(
            package='stereo_usb_cam_setup',
            executable='compressed_to_raw_node',
            name='color_decompressor',
            parameters=[get_node_params('color_decompressor')], # Load node-specific parameters
            output='screen'
        ),

        Node(
            package='stereo_usb_cam_setup',
            executable='depth_compressed_to_raw_node',
            name='depth_decompressor',
            parameters=[get_node_params('depth_decompressor')], # Load node-specific parameters
            output='screen'
        ),
    ])