import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg_share = get_package_share_directory("turtlebot3_drl")

    slam_params_file = os.path.join(pkg_share, "config", "slam_handheld.yaml")
    ekf_params_file = os.path.join(pkg_share, "config", "ekf.yaml")

    return LaunchDescription(
        [
            # LiDAR Driver
            Node(
                package="hls_lfcd_lds_driver",
                executable="hlds_laser_publisher",
                name="hlds_laser_publisher",
                parameters=[{"port": "/dev/ttyUSB0", "frame_id": "laser", "use_sim_time": False}],
                output="screen",
            ),
            # base_link -> laser
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_tf_laser",
                arguments=["0", "0", "0.1", "0", "0", "0", "base_link", "laser"],
            ),
            # base_link -> base_footprint
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_tf_footprint",
                arguments=["0", "0", "0", "0", "0", "0", "base_link", "base_footprint"],
            ),
            # base_link -> imu_link
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_tf_imu",
                arguments=[
                    "0.095",
                    "-0.07",
                    "0.02",
                    "0",
                    "0",
                    "0",
                    "base_link",
                    "imu_link",
                ],
            ),
            # Placeholder so odom frame exists before EKF starts publishing
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_tf_odom_init",
                arguments=["0", "0", "0", "0", "0", "0", "odom", "base_link"],
            ),
            # Wheel odometry (needed by EKF)
            Node(
                package="turtlebot3_drl",
                executable="diff_drive_controller",
                name="diff_drive_controller",
                output="screen",
                parameters=[{"use_sim_time": False}],
            ),
            # Fuse /odom + /imu/data → TF: odom -> base_link
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[ekf_params_file, {"use_sim_time": False}],
            ),
            # SLAM Toolbox
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[slam_params_file, {"use_sim_time": False}],
            ),
        ]
    )