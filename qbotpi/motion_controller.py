"""
QBot3 Closed-Loop Motion Controller (ROS2 Foxy Compatible)

This node provides precise, feedback-driven motion control using:
- Encoder-based distance measurement
- Gyro-based angle measurement
- Real-time progress feedback
- Completion detection
- Emergency stop support

Compatible with ROS2 Foxy on Raspberry Pi.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool, String, Int64MultiArray
from sensor_msgs.msg import Imu
import math
import threading
import time


class ClosedLoopMotionController(Node):
    """Executes motion commands with precise encoder and gyro feedback."""
    
    def __init__(self):
        super().__init__('motion_controller')
        
        # ===== SUBSCRIBERS =====
        # Command input (Dedicated topic for precise moves to avoid teleop conflict)
        self.create_subscription(Twist, '/qbot3/precise_cmd', self.cmd_callback, 10)
        # Sensor inputs
        self.create_subscription(Imu, '/qbot3/imu', self.imu_callback, 10)
        self.create_subscription(Int64MultiArray, '/qbot3/encoders', self.encoder_callback, 10)
        
        # ===== PUBLISHERS =====
        # Send velocity commands to robot base
        self.velocity_pub = self.create_publisher(Twist, '/qbot3/cmd_vel', 10)
        # Feedback topics
        self.feedback_pub = self.create_publisher(Float32, '/motion/feedback', 10)
        self.result_pub = self.create_publisher(Bool, '/motion/result', 10)
        self.status_pub = self.create_publisher(String, '/motion/status', 10)
        
        # ===== STATE =====
        self.current_yaw = 0.0  # degrees
        self.left_enc = 0
        self.right_enc = 0
        self.is_moving = False
        self.emergency_stop = False
        
        # ===== CALIBRATION =====
        # Tune these values for your robot!
        self.TICKS_PER_METER = 2578.0  # Encoder ticks per meter
        self.KP_DIST = 0.8  # P gain for distance control
        self.KP_TURN = 0.025  # P gain for turning control
        
        # Control limits
        self.MAX_LINEAR_SPEED = 0.3  # m/s
        self.MIN_LINEAR_SPEED = 0.05  # m/s (minimum to overcome friction)
        self.MAX_ANGULAR_SPEED = 1.0  # rad/s
        self.MIN_ANGULAR_SPEED = 0.15  # rad/s
        
        # Tolerance for completion
        self.DISTANCE_TOLERANCE = 50  # ticks (~2cm)
        self.ANGLE_TOLERANCE = 2.0  # degrees
        
        # Timeout
        self.ACTION_TIMEOUT = 30.0  # seconds
        
        self.get_logger().info("✅ Closed-Loop Motion Controller Online")
        self.get_logger().info(f"   Ticks/meter: {self.TICKS_PER_METER}")
        self.get_logger().info(f"   Distance tolerance: ± {self.DISTANCE_TOLERANCE / self.TICKS_PER_METER * 100:.1f} cm")
        self.get_logger().info(f"   Angle tolerance: ± {self.ANGLE_TOLERANCE}°")
        
        # Publish status
        self._publish_status("idle")
    
    def imu_callback(self, msg):
        """Extract yaw angle from IMU quaternion."""
        q = msg.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    
    def encoder_callback(self, msg):
        """Extract encoder values from array."""
        if len(msg.data) >= 2:
            self.left_enc = msg.data[0]
            self.right_enc = msg.data[1]
    
    def cmd_callback(self, msg):
        """Handle incoming Twist command."""
        if self.is_moving:
            self.get_logger().warn("Already executing action, ignoring new command")
            return
        
        # Emergency stop command (both zero)
        if abs(msg.linear.x) < 0.001 and abs(msg.angular.z) < 0.001:
            self.emergency_stop = True
            self.stop_robot()
            self._publish_status("emergency_stop")
            return
        
        # Determine action type
        if abs(msg.linear.x) > 0.001:
            # Linear motion
            distance_m = msg.linear.x
            self.get_logger().info(f"🎯 Command: MOVE {distance_m:.3f} m")
            threading.Thread(target=self.execute_move, args=(distance_m,), daemon=True).start()
        
        elif abs(msg.angular.z) > 0.001:
            # Angular motion  
            angle_rad = msg.angular.z
            angle_deg = math.degrees(angle_rad)
            self.get_logger().info(f"🎯 Command: TURN {angle_deg:.1f}°")
            threading.Thread(target=self.execute_turn, args=(angle_deg,), daemon=True).start()
    
    def execute_move(self, distance_m: float):
        """
        Execute linear movement with encoder feedback.
        
        Args:
            distance_m: Target distance in meters (positive=forward, negative=backward)
        """
        self.is_moving = True
        self.emergency_stop = False
        self._publish_status("moving")
        
        try:
            # Calculate target
            target_ticks = distance_m * self.TICKS_PER_METER
            start_avg = (self.left_enc + self.right_enc) / 2.0
            start_time = time.time()
            
            self.get_logger().info(f"   Target: {target_ticks:.0f} ticks ({distance_m:.2f} m)")
            self.get_logger().info(f"   Start position: {start_avg:.0f}")
            
            # Control loop
            while True:
                # Check timeout
                if time.time() - start_time > self.ACTION_TIMEOUT:
                    self.get_logger().error("⏱️ TIMEOUT: Motion exceeded 30 seconds")
                    self.stop_robot()
                    self._publish_result(False)
                    break
                
                # Check emergency stop
                if self.emergency_stop:
                    self.get_logger().warn("🛑 Emergency stop triggered")
                    self.stop_robot()
                    self._publish_result(False)
                    break
                
                # Calculate current progress
                current_avg = (self.left_enc + self.right_enc) / 2.0
                distance_moved = current_avg - start_avg
                error_ticks = target_ticks - distance_moved
                
                # Completion check
                if abs(error_ticks) <= self.DISTANCE_TOLERANCE:
                    self.stop_robot()
                    actual_dist = distance_moved / self.TICKS_PER_METER
                    error_pct = abs((actual_dist - distance_m) / distance_m * 100) if distance_m != 0 else 0
                    
                    self.get_logger().info(f"✅ COMPLETE: Moved {actual_dist:.3f} m (error: {error_pct:.1f}%)")
                    self._publish_feedback(100.0)
                    self._publish_result(True)
                    break
                
                # Calculate progress percentage
                progress = (abs(distance_moved) / abs(target_ticks)) * 100.0
                progress = min(progress, 99.0)  # Cap at 99% until truly complete
                self._publish_feedback(progress)
                
                # Calculate velocity (proportional control)
                error_m = error_ticks / self.TICKS_PER_METER
                velocity = error_m * self.KP_DIST
                
                # Apply limits
                velocity = max(min(velocity, self.MAX_LINEAR_SPEED), -self.MAX_LINEAR_SPEED)
                
                # Apply minimum speed to overcome friction
                if 0 < velocity < self.MIN_LINEAR_SPEED:
                    velocity = self.MIN_LINEAR_SPEED
                elif -self.MIN_LINEAR_SPEED < velocity < 0:
                    velocity = -self.MIN_LINEAR_SPEED
                
                # Send velocity command
                msg = Twist()
                msg.linear.x = velocity
                self.velocity_pub.publish(msg)
                
                time.sleep(0.05)  # 20 Hz control loop
                
        except Exception as e:
            self.get_logger().error(f"❌ Error during move: {e}")
            self.stop_robot()
            self._publish_result(False)
        
        finally:
            self.is_moving = False
            self._publish_status("idle")
    
    def execute_turn(self, angle_deg: float):
        """
        Execute rotation with gyro feedback.
        
        Args:
            angle_deg: Target angle in degrees (positive=left/CCW, negative=right/CW)
        """
        self.is_moving = True
        self.emergency_stop = False
        self._publish_status("turning")
        
        try:
            # Calculate target
            target_yaw = self.current_yaw + angle_deg
            start_time = time.time()
            
            self.get_logger().info(f"   Current yaw: {self.current_yaw:.1f}°")
            self.get_logger().info(f"   Target yaw: {target_yaw:.1f}°")
            
            # Control loop
            while True:
                # Check timeout
                if time.time() - start_time > self.ACTION_TIMEOUT:
                    self.get_logger().error("⏱️ TIMEOUT: Turn exceeded 30 seconds")
                    self.stop_robot()
                    self._publish_result(False)
                    break
                
                # Check emergency stop
                if self.emergency_stop:
                    self.get_logger().warn("🛑 Emergency stop triggered")
                    self.stop_robot()
                    self._publish_result(False)
                    break
                
                # Calculate error (handle wraparound)
                error = target_yaw - self.current_yaw
                
                # Normalize to [-180, 180]
                while error > 180:
                    error -= 360
                while error < -180:
                    error += 360
                
                # Completion check
                if abs(error) <= self.ANGLE_TOLERANCE:
                    self.stop_robot()
                    actual_turn = angle_deg - error
                    
                    self.get_logger().info(f"✅ COMPLETE: Turned {actual_turn:.1f}° (error: {error:.1f}°)")
                    self._publish_feedback(100.0)
                    self._publish_result(True)
                    break
                
                # Calculate progress percentage
                progress = (abs(angle_deg - error) / abs(angle_deg)) * 100.0 if angle_deg != 0 else 0
                progress = min(progress, 99.0)
                self._publish_feedback(progress)
                
                # Calculate angular velocity (proportional control)
                angular_velocity = error * self.KP_TURN
                
                # Apply limits
                angular_velocity = max(min(angular_velocity, self.MAX_ANGULAR_SPEED), -self.MAX_ANGULAR_SPEED)
                
                # Apply minimum speed
                if 0 < angular_velocity < self.MIN_ANGULAR_SPEED:
                    angular_velocity = self.MIN_ANGULAR_SPEED
                elif -self.MIN_ANGULAR_SPEED < angular_velocity < 0:
                    angular_velocity = -self.MIN_ANGULAR_SPEED
                
                # Send velocity command
                msg = Twist()
                msg.angular.z = angular_velocity
                self.velocity_pub.publish(msg)
                
                time.sleep(0.05)  # 20 Hz control loop
                
        except Exception as e:
            self.get_logger().error(f"❌ Error during turn: {e}")
            self.stop_robot()
            self._publish_result(False)
        
        finally:
            self.is_moving = False
            self._publish_status("idle")
    
    def stop_robot(self):
        """Send zero velocity to stop the robot."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        try:
            self.velocity_pub.publish(msg)
        except Exception:
            pass
    
    def _publish_feedback(self, progress: float):
        """Publish progress percentage (0-100)."""
        msg = Float32()
        msg.data = float(progress)
        self.feedback_pub.publish(msg)
    
    def _publish_result(self, success: bool):
        """Publish final result (True=success, False=failure)."""
        msg = Bool()
        msg.data = success
        self.result_pub.publish(msg)
    
    def _publish_status(self, status: str):
        """Publish current status string."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ClosedLoopMotionController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()

