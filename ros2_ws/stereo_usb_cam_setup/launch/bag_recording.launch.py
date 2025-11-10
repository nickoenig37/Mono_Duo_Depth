# File Launches Setup for the real robot

import os
from ament_index_python.packages import get_package_share_directory
from datetime import datetime


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros import actions

# Function to format and organize bags when running launch
def generate_bag_command():
    # Get the current user's home directory
    home_dir = os.path.expanduser('~')

    # Define the target directory
    target_dir = os.path.join(home_dir, 'Documents', 'AIRLAB_ROS2_bags')

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Get the current date and time
    current_date = datetime.now().strftime('%Y-%m-%d')
    formatted_time = datetime.now().strftime('%H-%M')

    # Define the output bag path
    output_bag_path = os.path.join(target_dir, f'{current_date}-{formatted_time}')

    # ****************** COMMAND FOR RECORDING BAGS ******************
    
    # If you want to bag all live topics and exclude certain topics you can follow this cmd:
    # cmd = ['ros2', 'bag', 'record', '--include-hidden-topics', '-o', output_bag_path, '-a', '-x', "(/topic1|/topic2)"]

    # If you want to bag only certain topics you can follow this cmd:
    topics_to_record = [
        '/camera/camera/color/camera_info',
        '/camera/camera/color/image_raw/compressed',
        '/camera/camera/aligned_depth_to_color/image_raw/compressedDepth',
        # '/camera/camera/aligned_depth_to_color/image_raw/compressed',
        'tf_static',
        '/camera/camera/extrinsics/depth_to_color',
        '/camera/left/image_raw',
        '/camera/right/image_raw',
    ]

    # Construct the command
    cmd = ['ros2', 'bag', 'record', '--include-hidden-topics', '-o', output_bag_path] + topics_to_record

    return cmd

def generate_launch_description():
    # Launch file logging
    bag_command = generate_bag_command()
    file_logging = ExecuteProcess(
        cmd= bag_command,
        output='screen'
    )

    ld = LaunchDescription()

    # Logging laucnh part
    logging = True
    if logging:
        ld.add_action(file_logging)
    return ld