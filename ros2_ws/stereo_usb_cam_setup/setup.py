from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'stereo_usb_cam_setup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.py")),),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.rviz")),),
        # Install config files
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='koener',
    maintainer_email='nic.koenig37@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dual_camera_node = stereo_usb_cam_setup.dual_camera_node:main',
            'stereo_fusion_node = stereo_usb_cam_setup.stereo_fusion_node:main',
            'dataset_recorder_node = stereo_usb_cam_setup.dataset_recorder_node:main',
        ],
    },
)
