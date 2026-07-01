"""
QBot3 Base Controller (ROS2 Foxy Compatible)

Hardware interface node that:
- Communicates with QBot3 hardware
- Publishes sensor data (IMU, encoders, battery, bumpers, cliff, wheel drop)
- Receives velocity commands via Twist messages
- Publishes diagnostic information

Compatible with ROS2 Foxy on Raspberry Pi.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
from qbot3.lib_qbot import QBot3
from std_msgs.msg import Bool, String, Float32, Int64MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import BatteryState, Imu
from kobuki_ros_interfaces.msg import BumperEvent, CliffEvent, WheelDropEvent
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue



import random

class MockQBot3:
    """Mock class to simulate QBot3 hardware when not connected."""
    def __init__(self):
        self.left_encoder = 0
        self.right_encoder = 0
        self.bat_voltage = 12.0
        self.accelerometer = [0.0, 0.0, 9.8]
        self.gyroscope = [0.0, 0.0, 0.0]
        self.bumpers = [0, 0, 0]
        self.cliff = [0, 0, 0]
        self.wheel_drop = [0, 0]
        self.buttons = [0, 0]
        
        # Internal state for simulation
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
    
    def read_write_std(self, right_vel, left_vel, led1, led2):
        """Simulate hardware interaction."""
        # Simple kinematic update
        dt = 1.0 / 50.0  # Assumes 50Hz loop
        
        # Update encoders (approximate)
        # Ticks per meter = 2578
        self.right_encoder += int(right_vel * dt * 2578)
        self.left_encoder += int(left_vel * dt * 2578)
        
        # Simulate battery drain
        self.bat_voltage = max(10.5, self.bat_voltage - 0.0001)
        
        # Simulate gyro noise
        self.gyroscope = [random.gauss(0, 0.01), random.gauss(0, 0.01), random.gauss(0, 0.01)]
        
        # Simulate accel noise (stationary)
        self.accelerometer = [random.gauss(0, 0.05), random.gauss(0, 0.05), random.gauss(9.8, 0.05)]
    
    def terminate(self):
        pass

class Qbot3Node(Node):
    """Main hardware interface for QBot3 robot."""
    
    def __init__(self):
        super().__init__('qbot3_base')
        
        # ===== CONFIGURATION =====
        # ===== CONFIGURATION =====
        self.sample_rate = 50.0  # Hz (Reduced from 240Hz for stability)
        self.sample_time = 1.0 / self.sample_rate
        
        # Initialize Hardware with retry and fallback to simulation
        try:
            self.myQbot3 = QBot3()
            time.sleep(1.0) # Allow hardware to settle
            self.simulation_mode = False
        except Exception as e:
            self.get_logger().warn(f"⚠️ HARDWARE INIT FAILED: {e}")
            self.get_logger().warn("⚠️ SWITCHING TO SIMULATION MODE (Mock Hardware)")
            self.myQbot3 = MockQBot3()
            self.simulation_mode = True
        
        # ===== STATE =====
        self.current_yaw = 0.0
        self.last_time = self.get_clock().now()
        self.command = np.array([0.0, 0.0])  # [right_wheel, left_wheel] velocities
        
        # Encoder tracking for diagnostics
        self.last_enc_l = 0
        self.last_enc_r = 0
        self.enc_update_count = 0
        self.last_enc_check_time = time.time()
        
        # Gyro tracking for drift detection
        self.gyro_z_samples = []
        self.max_gyro_samples = 100

        # ===== GYRO BIAS / CALIBRATION =====
        # Hardware gyros report a small constant value when stationary
        # (the "bias"); integrating it produces silent yaw drift over time.
        # By default we do a 5 s startup calibration where the robot is
        # enforced stationary, sample gyro_z, and lock in the mean as the
        # bias. Afterwards we keep refining the bias slowly whenever the
        # robot is detected stationary again. Yaw integration uses gz-bias.
        #
        # The startup window is a ROS2 parameter so the user can disable it
        # at launch:
        #     ros2 run qbot3 qbot3_base --ros-args -p enable_gyro_calibration:=false
        # …or shorten it:
        #     ros2 run qbot3 qbot3_base --ros-args -p gyro_calibration_duration_s:=2.0
        # …and at runtime they can publish True to /qbot3/imu/skip_calibration
        # to abort an in-progress calibration window.
        self.declare_parameter('enable_gyro_calibration', True)
        self.declare_parameter('gyro_calibration_duration_s', 5.0)
        self.GYRO_CALIBRATION_ENABLED = bool(
            self.get_parameter('enable_gyro_calibration').value
        )
        self.GYRO_CALIBRATION_DURATION_S = float(
            self.get_parameter('gyro_calibration_duration_s').value
        )
        self.GYRO_BIAS_MAX_SAMPLES = 200
        self._calibration_start_ts = time.time()
        self._calibration_samples = []
        # When calibration is disabled we mark it done immediately — bias
        # starts at 0 and online refinement during stationary periods will
        # estimate it eventually. cmd_vel is NOT blocked.
        self._calibration_done = not self.GYRO_CALIBRATION_ENABLED
        self.gyro_bias_z = 0.0
        # Stationary detection state
        self._last_left_enc_for_still = 0
        self._last_right_enc_for_still = 0
        # Post-motion settling window for the online bias refinement.
        # After a turn the gyro takes a moment to settle (motor vibration,
        # power-rail recovery). If we accept those samples into the bias
        # EMA, the bias drifts to a wrong value and `gz_corrected` becomes
        # systematically signed → yaw silently rotates back toward zero,
        # and via the d_yaw·(b/2) rotation-compensation in odometry_node
        # that also creates phantom translation. We block the refinement
        # for SETTLING_S seconds after the last non-stationary cycle so
        # only truly settled samples reach the EMA.
        self.GYRO_SETTLING_S = 2.0
        self._t_last_motion = time.time()
        # Even-slower EMA — was 0.95/0.05. With 0.99/0.01 a contaminated
        # sample shifts the bias 5× less per cycle, so any leakage from
        # the settling window converges out instead of biasing yaw.
        self.GYRO_BIAS_EMA_RETAIN = 0.99

        # ===== PUBLISHERS (Sensors) =====
        self.imu_pub = self.create_publisher(Imu, '/qbot3/imu', 10)
        self.encoder_pub = self.create_publisher(Int64MultiArray, '/qbot3/encoders', 10)
        self.battery_pub = self.create_publisher(BatteryState, '/qbot3/battery_state', 10)
        self.bumper_pub = self.create_publisher(BumperEvent, '/qbot3/bumpers', 10)
        self.cliff_pub = self.create_publisher(CliffEvent, '/qbot3/cliff', 10)
        self.wheel_drop_pub = self.create_publisher(WheelDropEvent, '/qbot3/wheel_drop', 10)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        # Calibration status — host UI watches these to show a banner +
        # disable mission-start buttons until calibration completes.
        self.imu_calibrated_pub = self.create_publisher(Bool, '/qbot3/imu/calibrated', 10)
        self.imu_calib_progress_pub = self.create_publisher(Float32, '/qbot3/imu/calibration_progress', 10)

        # ===== TIMERS =====
        # Low-speed sensor publishing (10 Hz)
        self.create_timer(0.1, self.battery_callback)
        self.create_timer(0.1, self.bumper_callback)
        self.create_timer(0.1, self.cliff_callback)
        self.create_timer(0.1, self.wheel_drop_callback)
        self.create_timer(0.1, self.diagnostics_callback)
        # Calibration status (5 Hz — fast enough for a smooth countdown)
        self.create_timer(0.2, self.publish_calibration_status)

        # High-speed control loop (240 Hz)
        self.create_timer(self.sample_time, self.control_loop)

        # ===== SUBSCRIBERS =====
        self.create_subscription(Twist, '/qbot3/cmd_vel', self.process_cmd, 10)
        # Remote yaw reset — re-triggers the calibration window
        self.create_subscription(Bool, '/qbot3/imu/reset_yaw', self.reset_yaw_callback, 10)
        # Runtime skip — operator presses "Skip" on the host banner; we
        # finalise calibration immediately using whatever samples we already
        # gathered (or bias=0 if the window had just started).
        self.create_subscription(Bool, '/qbot3/imu/skip_calibration', self.skip_calibration_callback, 10)

        self.get_logger().info("QBot3 Base Controller Online")
        self.get_logger().info(f"   Control loop: {self.sample_rate} Hz")
        self.get_logger().info(f"   Sensor publishing: 10 Hz")
        self.get_logger().info(f"   Listening for commands on /qbot3/cmd_vel")
        if self.GYRO_CALIBRATION_ENABLED:
            self.get_logger().info(
                f"   Gyro calibration: {self.GYRO_CALIBRATION_DURATION_S} s "
                "— keep robot STILL until status: calibrated"
            )
        else:
            self.get_logger().warn(
                "   Gyro calibration: DISABLED (param enable_gyro_calibration=false). "
                "Bias starts at 0; online refinement will estimate it during "
                "stationary periods, but expect drift until then."
            )
    
    def control_loop(self):
        """
        Main control loop running at 240 Hz.
        Reads sensors and writes motor commands to hardware.
        """
        # 1. Read/Write hardware
        LED_state = np.zeros((2, 1))
        self.myQbot3.read_write_std(
            self.command[0],  # right wheel
            self.command[1],  # left wheel
            int(LED_state[0]),
            int(LED_state[1])
        )
        
        # 2. Publish high-speed sensor data
        self.publish_imu()
        self.publish_encoders()
    
    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - \
             math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + \
             math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - \
             math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + \
             math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]
    
    def _is_stationary(self) -> bool:
        """True when the robot is genuinely not moving.

        We check three signals:
          - commanded velocity ≈ 0 (we know what we asked the wheels to do)
          - left encoder ticks unchanged since last call
          - right encoder ticks unchanged since last call

        Encoders confirm wheels physically didn't turn. Without them, a
        stuck-in-place motor that's commanding zero would still report
        stationary even if a hand was pushing the robot.
        """
        cmd_zero = abs(self.command[0]) < 0.001 and abs(self.command[1]) < 0.001
        enc_still = (
            self.myQbot3.left_encoder == self._last_left_enc_for_still
            and self.myQbot3.right_encoder == self._last_right_enc_for_still
        )
        self._last_left_enc_for_still = self.myQbot3.left_encoder
        self._last_right_enc_for_still = self.myQbot3.right_encoder
        return cmd_zero and enc_still

    def publish_calibration_status(self) -> None:
        """Periodic broadcast so the host UI can show the calibration banner."""
        bool_msg = Bool()
        bool_msg.data = self._calibration_done
        self.imu_calibrated_pub.publish(bool_msg)

        progress_msg = Float32()
        if self._calibration_done:
            progress_msg.data = 1.0
        else:
            elapsed = time.time() - self._calibration_start_ts
            progress_msg.data = float(min(1.0, elapsed / self.GYRO_CALIBRATION_DURATION_S))
        self.imu_calib_progress_pub.publish(progress_msg)

    def reset_yaw_callback(self, msg):
        """Remote reset — zero yaw and re-trigger gyro calibration.

        Honours the `enable_gyro_calibration` parameter: if disabled, just
        zeroes yaw and keeps motion enabled (bias stays at its current value).
        """
        if not bool(msg.data):
            return
        self.current_yaw = 0.0
        if self.GYRO_CALIBRATION_ENABLED:
            self._calibration_samples = []
            self._calibration_done = False
            self._calibration_start_ts = time.time()
            self.command = np.array([0.0, 0.0])
            self.get_logger().info(
                f"Yaw reset — recalibrating gyro for {self.GYRO_CALIBRATION_DURATION_S}s, "
                "motion blocked until done"
            )
        else:
            self.get_logger().info("Yaw reset (calibration disabled — motion not blocked)")

    def skip_calibration_callback(self, msg):
        """Runtime override — operator pressed Skip on the host banner.

        Finalise the calibration window with whatever samples we have so far
        (could be 0 if Skip is hit immediately after launch — bias stays 0
        and online refinement takes over).
        """
        if not bool(msg.data):
            return
        if self._calibration_done:
            self.get_logger().debug("Skip ignored — calibration already finished")
            return
        if self._calibration_samples:
            self.gyro_bias_z = (
                sum(self._calibration_samples) / len(self._calibration_samples)
            )
            self.get_logger().info(
                f"Calibration skipped: bias = {self.gyro_bias_z:+.5f} rad/s "
                f"({len(self._calibration_samples)} samples)"
            )
        else:
            self.get_logger().warn(
                "Calibration skipped with 0 samples — bias kept at 0; "
                "online refinement will estimate it on the next stationary period"
            )
        self._calibration_done = True
        self.current_yaw = 0.0
        self._calibration_samples = self._calibration_samples[-self.GYRO_BIAS_MAX_SAMPLES:]

    def publish_imu(self):
        """Publish IMU data with integrated yaw + bias-corrected gyro_z."""
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'qbot3_imu'

        # Read accelerometer and gyroscope
        ax = float(self.myQbot3.accelerometer[0])
        ay = float(self.myQbot3.accelerometer[1])
        az = float(self.myQbot3.accelerometer[2])
        gx = float(self.myQbot3.gyroscope[0])
        gy = float(self.myQbot3.gyroscope[1])
        gz_raw = float(self.myQbot3.gyroscope[2])

        # Time delta (computed even during calibration so callers see fresh stamps)
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # ----- Gyro bias / yaw integration -----
        if not self._calibration_done:
            # ===== Phase 1: 5 s startup calibration =====
            # Robot is enforced stationary by process_cmd(). Just gather samples.
            elapsed = time.time() - self._calibration_start_ts
            self._calibration_samples.append(gz_raw)
            if elapsed >= self.GYRO_CALIBRATION_DURATION_S:
                if len(self._calibration_samples) >= 30:
                    self.gyro_bias_z = (
                        sum(self._calibration_samples) / len(self._calibration_samples)
                    )
                    self.get_logger().info(
                        f"Gyro calibrated: bias = {self.gyro_bias_z:+.5f} rad/s "
                        f"({len(self._calibration_samples)} samples)"
                    )
                else:
                    self.get_logger().warn(
                        f"Gyro calibration finished with only "
                        f"{len(self._calibration_samples)} samples — bias kept at 0"
                    )
                self._calibration_done = True
                self.current_yaw = 0.0
                # Trim sample buffer for online refinement
                self._calibration_samples = self._calibration_samples[-self.GYRO_BIAS_MAX_SAMPLES:]
            # During calibration, force yaw to 0; don't integrate yet.
            gz_corrected = 0.0
        else:
            # ===== Phase 2: online refinement during stationary periods =====
            # Maintain a "time of last motion" so we can gate samples until
            # the gyro has had time to settle. Without this, the EMA absorbs
            # post-turn vibration and yaw drifts backward over the next
            # second or two (visible in the SLAM viewer as the cone
            # rotating back toward its starting heading even though the
            # robot is sitting still).
            stationary = self._is_stationary()
            if not stationary:
                self._t_last_motion = time.time()
            settled = (time.time() - self._t_last_motion) >= self.GYRO_SETTLING_S

            if stationary and settled:
                self._calibration_samples.append(gz_raw)
                if len(self._calibration_samples) > self.GYRO_BIAS_MAX_SAMPLES:
                    self._calibration_samples.pop(0)
                # Very slow EMA — even if one settled sample slips through
                # contaminated, it only nudges the bias 1% per cycle.
                if len(self._calibration_samples) >= 20:
                    recent_mean = (
                        sum(self._calibration_samples[-100:])
                        / min(len(self._calibration_samples), 100)
                    )
                    self.gyro_bias_z = (
                        self.GYRO_BIAS_EMA_RETAIN * self.gyro_bias_z
                        + (1.0 - self.GYRO_BIAS_EMA_RETAIN) * recent_mean
                    )
            gz_corrected = gz_raw - self.gyro_bias_z
            self.current_yaw += gz_corrected * dt

        # Track corrected gyro samples for the diagnostics drift report
        sample_for_diag = gz_corrected if self._calibration_done else gz_raw - self.gyro_bias_z
        if len(self.gyro_z_samples) < self.max_gyro_samples:
            self.gyro_z_samples.append(sample_for_diag)
        else:
            self.gyro_z_samples.pop(0)
            self.gyro_z_samples.append(sample_for_diag)

        # Convert to quaternion
        q = self.get_quaternion_from_euler(0, 0, self.current_yaw)
        imu_msg.orientation.x = q[0]
        imu_msg.orientation.y = q[1]
        imu_msg.orientation.z = q[2]
        imu_msg.orientation.w = q[3]

        imu_msg.angular_velocity.x = gx
        imu_msg.angular_velocity.y = gy
        # Publish the BIAS-CORRECTED gz so all consumers (motion_controller,
        # odometry_node, host bridge) see a clean reading.
        imu_msg.angular_velocity.z = gz_corrected
        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az
        
        # Covariance (Diagonal matrices with estimated variance)
        # Orientation: High confidence (0.01)
        imu_msg.orientation_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]
        
        # Angular Velocity: Medium confidence
        imu_msg.angular_velocity_covariance = [
            0.02, 0.0, 0.0,
            0.0, 0.02, 0.0,
            0.0, 0.0, 0.02
        ]
        
        # Linear Acceleration: Lower confidence due to vibration
        imu_msg.linear_acceleration_covariance = [
            0.04, 0.0, 0.0,
            0.0, 0.04, 0.0,
            0.0, 0.0, 0.04
        ]
        
        self.imu_pub.publish(imu_msg)
    
    def publish_encoders(self):
        """Publish encoder data as [left, right] array."""
        msg = Int64MultiArray()
        msg.data = [int(self.myQbot3.left_encoder), int(self.myQbot3.right_encoder)]
        self.encoder_pub.publish(msg)
        
        # Track for diagnostics
        if self.myQbot3.left_encoder != self.last_enc_l or \
           self.myQbot3.right_encoder != self.last_enc_r:
            self.enc_update_count += 1
            self.last_enc_l = self.myQbot3.left_encoder
            self.last_enc_r = self.myQbot3.right_encoder
    
    def battery_callback(self):
        """Publish battery voltage."""
        msg = BatteryState()
        msg.voltage = float(self.myQbot3.bat_voltage)
        self.battery_pub.publish(msg)
    
    def bumper_callback(self):
        """Publish bumper events."""
        msg = BumperEvent()
        if self.myQbot3.bumpers[0] == 1:
            msg.bumper = 0
            msg.state = 1
        elif self.myQbot3.bumpers[1] == 1:
            msg.bumper = 1
            msg.state = 1
        elif self.myQbot3.bumpers[2] == 1:
            msg.bumper = 2
            msg.state = 1
        else:
            msg.state = 0
        self.bumper_pub.publish(msg)
    
    def cliff_callback(self):
        """Publish cliff detection events."""
        msg = CliffEvent()
        if self.myQbot3.cliff[0] == 1:
            msg.sensor = CliffEvent.LEFT
            msg.state = 1
        elif self.myQbot3.cliff[1] == 1:
            msg.sensor = CliffEvent.CENTER
            msg.state = 1
        elif self.myQbot3.cliff[2] == 1:
            msg.sensor = CliffEvent.RIGHT
            msg.state = 1
        else:
            msg.state = 0
        self.cliff_pub.publish(msg)
    
    def wheel_drop_callback(self):
        """Publish wheel drop events."""
        msg = WheelDropEvent()
        if self.myQbot3.wheel_drop[0] == 1:
            msg.wheel = WheelDropEvent.LEFT
            msg.state = WheelDropEvent.DROPPED
        elif self.myQbot3.wheel_drop[1] == 1:
            msg.wheel = WheelDropEvent.RIGHT
            msg.state = WheelDropEvent.DROPPED
        else:
            msg.state = WheelDropEvent.RAISED
        self.wheel_drop_pub.publish(msg)
    
    def diagnostics_callback(self):
        """Publish system diagnostics."""
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # ===== ENCODER DIAGNOSTICS =====
        enc_status = DiagnosticStatus()
        enc_status.name = "Encoders"
        enc_status.hardware_id = "qbot3_encoders"
        
        # Calculate encoder update rate
        current_time = time.time()
        time_elapsed = current_time - self.last_enc_check_time
        if time_elapsed >= 1.0:  # Check every second
            update_rate = self.enc_update_count / time_elapsed
            self.enc_update_count = 0
            self.last_enc_check_time = current_time
            
            if update_rate > 200:  # Expect ~240 Hz
                enc_status.level = DiagnosticStatus.OK
                enc_status.message = "Encoder data rate excellent"
            elif update_rate > 100:
                enc_status.level = DiagnosticStatus.WARN
                enc_status.message = "Encoder data rate acceptable"
            else:
                enc_status.level = DiagnosticStatus.ERROR
                enc_status.message = "Encoder data rate too low"
            
            enc_status.values.append(KeyValue(key="update_rate_hz", value=f"{update_rate:.1f}"))
        else:
            enc_status.level = DiagnosticStatus.OK
            enc_status.message = "Monitoring..."
        
        enc_status.values.append(KeyValue(key="left_encoder", value=str(self.myQbot3.left_encoder)))
        enc_status.values.append(KeyValue(key="right_encoder", value=str(self.myQbot3.right_encoder)))
        
        msg.status.append(enc_status)
        
        # ===== GYRO DIAGNOSTICS =====
        gyro_status = DiagnosticStatus()
        gyro_status.name = "Gyroscope"
        gyro_status.hardware_id = "qbot3_imu"
        
        if len(self.gyro_z_samples) >= 50:
            # Calculate drift (mean when stationary should be ~0)
            mean_drift = np.mean(self.gyro_z_samples)
            std_drift = np.std(self.gyro_z_samples)
            
            if abs(mean_drift) < 0.01 and std_drift < 0.02:
                gyro_status.level = DiagnosticStatus.OK
                gyro_status.message = "Gyro performance excellent"
            elif abs(mean_drift) < 0.05:
                gyro_status.level = DiagnosticStatus.WARN
                gyro_status.message = "Gyro drift detected"
            else:
                gyro_status.level = DiagnosticStatus.ERROR
                gyro_status.message = "Significant gyro drift - recalibration needed"
            
            gyro_status.values.append(KeyValue(key="drift_rad_s", value=f"{mean_drift:.4f}"))
            gyro_status.values.append(KeyValue(key="noise_std", value=f"{std_drift:.4f}"))
        else:
            gyro_status.level = DiagnosticStatus.OK
            gyro_status.message = "Collecting samples..."
        
        msg.status.append(gyro_status)
        
        # ===== BATTERY DIAGNOSTICS =====
        batt_status = DiagnosticStatus()
        batt_status.name = "Battery"
        batt_status.hardware_id = "qbot3_battery"
        
        # Convert numpy array to float for formatting
        voltage = float(self.myQbot3.bat_voltage)
        if voltage > 12.5:
            batt_status.level = DiagnosticStatus.OK
            batt_status.message = "Battery healthy"
        elif voltage > 11.5:
            batt_status.level = DiagnosticStatus.WARN
            batt_status.message = "Battery low - recharge soon"
        else:
            batt_status.level = DiagnosticStatus.ERROR
            batt_status.message = "Battery critical - recharge NOW"
        
        batt_status.values.append(KeyValue(key="voltage", value=f"{voltage:.2f}"))
        msg.status.append(batt_status)
        
        # Publish diagnostics
        self.diagnostics_pub.publish(msg)
    
    def process_cmd(self, msg):
        """
        Convert Twist command to differential drive wheel velocities.

        While the gyro is still being calibrated (first 5 s after node start
        or after a yaw reset) we silently drop the command and force the
        wheels to zero. The host UI shows a banner so the operator knows
        why nothing's happening.

        Args:
            msg: Twist message with linear.x and angular.z
        """
        if not self._calibration_done:
            # Force stationary so the bias estimator isn't poisoned
            self.command = np.array([0.0, 0.0])
            return

        linear_vel = msg.linear.x  # m/s
        angular_rate = msg.angular.z  # rad/s


        # Wheelbase distance
        wheel_base = 0.235  # meters
        
        # Differential drive kinematics
        right_vel = linear_vel + 0.5 * wheel_base * angular_rate
        left_vel = linear_vel - 0.5 * wheel_base * angular_rate
        
        self.command = np.array([right_vel, left_vel])
    
    def stop_bot(self):
        """Emergency stop - zero all velocities."""
        self.command = np.array([0.0, 0.0])
        self.myQbot3.terminate()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Qbot3Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if node:
            node.get_logger().info("Shutting down...")
            try:
                node.stop_bot()
            except:
                pass
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

