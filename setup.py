from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tvvf_vo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_dir={package_name: 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'rclpy',
        'geometry_msgs',
        'nav_msgs',
        'sensor_msgs',
        'std_msgs',
        'visualization_msgs',
        'tf2_ros',
        'tf2_geometry_msgs',
        'transforms3d',
    ],
    zip_safe=True,
    maintainer='ryo',
    maintainer_email='s24s1040du@s.chibakoudai.jp',
    description='TVVF-VO integrated navigation system for ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tvvf_vo_node = tvvf_vo.tvvf_vo_node:main',
            'tvvf_vo = tvvf_vo.tvvf_vo:main'
        ],
    },
)
