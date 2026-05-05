import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.actions import SetRemap
from launch_ros.substitutions import FindPackageShare


def _nav2_bringup_include(nav2_bringup_dir, map_yaml_file, params_file):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "map": map_yaml_file,
            "use_sim_time": "False",
            "params_file": params_file,
            "autostart": "True",
            "slam": "False",
            "use_composition": "False",
        }.items(),
    )


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    pkg_share = get_package_share_directory("turtlebot3_drl")

    default_map = os.path.join(os.path.expanduser("~"), "my_room_map.yaml")
    default_params_file = os.path.join(pkg_share, "config", "nav2_params.yaml")
    default_ekf_params_file = os.path.join(pkg_share, "config", "ekf.yaml")
    default_drl_config = PathJoinSubstitution(
        [FindPackageShare("turtlebot3_drl"), "config", "gap_sac.yaml"]
    )
    default_policy_checkpoint = PathJoinSubstitution(
        [FindPackageShare("turtlebot3_drl"), "models", "deployment_checkpoint.pt"]
    )

    map_yaml_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    ekf_params_file = LaunchConfiguration("ekf_params_file")
    drl_direct = LaunchConfiguration("drl_direct")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value=default_map,
                description="Full path to the map yaml file to load.",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
                description="Real-robot Nav2 parameter file.",
            ),
            DeclareLaunchArgument(
                "ekf_params_file",
                default_value=default_ekf_params_file,
                description="Real-robot EKF parameter file.",
            ),
            DeclareLaunchArgument(
                "policy_checkpoint",
                default_value=default_policy_checkpoint,
                description="Trained DRL deployment checkpoint.",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=default_drl_config,
                description="DRL controller configuration file.",
            ),
            DeclareLaunchArgument(
                "drl_direct",
                default_value="false",
                description="When true, DRL publishes /cmd_vel and Nav2 is remapped to /cmd_vel_nav2.",
            ),
            DeclareLaunchArgument("lookahead_distance_m", default_value="0.4"),
            DeclareLaunchArgument("safety_stop_distance", default_value="0.18"),
            DeclareLaunchArgument("goal_stop_distance_m", default_value="0.35"),
            Node(
                package="hls_lfcd_lds_driver",
                executable="hlds_laser_publisher",
                name="hlds_laser_publisher",
                parameters=[
                    {"port": "/dev/ttyUSB0", "frame_id": "laser", "use_sim_time": False}
                ],
                output="screen",
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_tf_laser",
                arguments=["0", "0", "0.1", "0", "0", "0", "base_link", "laser"],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_tf_footprint",
                arguments=["0", "0", "0", "0", "0", "0", "base_link", "base_footprint"],
            ),
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
            Node(
                package="agv_controller",
                executable="diff_drive_controller",
                name="diff_drive_controller",
                output="screen",
                parameters=[{"use_sim_time": False}],
            ),
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                output="screen",
                parameters=[ekf_params_file, {"use_sim_time": False}],
            ),
            GroupAction(
                condition=UnlessCondition(drl_direct),
                actions=[
                    _nav2_bringup_include(nav2_bringup_dir, map_yaml_file, params_file),
                ],
            ),
            GroupAction(
                condition=IfCondition(drl_direct),
                actions=[
                    SetRemap(src="/cmd_vel", dst="/cmd_vel_nav2"),
                    SetRemap(src="cmd_vel", dst="/cmd_vel_nav2"),
                    _nav2_bringup_include(nav2_bringup_dir, map_yaml_file, params_file),
                ],
            ),
            Node(
                package="turtlebot3_drl",
                executable="drl_controller_node",
                name="drl_controller_node",
                output="screen",
                parameters=[
                    {
                        "config_file": LaunchConfiguration("config_file"),
                        "policy_checkpoint": LaunchConfiguration("policy_checkpoint"),
                        "input_scan_topic": "/scan",
                        "input_odom_topic": "/odometry/filtered",
                        "input_goal_topic": "/goal_pose",
                        "input_plan_topic": "/plan",
                        "output_cmd_vel_topic": "/cmd_vel_drl",
                        "publish_directly_to_cmd_vel": drl_direct,
                        "plan_target_mode": "lookahead",
                        "lookahead_distance_m": LaunchConfiguration("lookahead_distance_m"),
                        "prefer_plan_goal": True,
                        "publish_debug": True,
                        "safety_stop_enabled": True,
                        "safety_stop_distance": LaunchConfiguration("safety_stop_distance"),
                        "goal_stop_distance_m": LaunchConfiguration("goal_stop_distance_m"),
                        "speed_governor_enabled": True,
                        "recovery_enabled": True,
                        "narrow_passage_enabled": True,
                    }
                ],
            ),
        ]
    )
