"""
QBot3 Odometry Node (ROS2 Foxy Compatible)

Fuses wheel encoder ticks and IMU yaw into a standard nav_msgs/Odometry stream
plus the matching odom -> base_link TF transform.

This node exists because qbot3_base.py publishes raw sensor data only; downstream
consumers (RTAB-Map, Nav2, host-side SLAM viewer) expect a proper Odometry topic.

Inputs:
    /qbot3/encoders        std_msgs/Int64MultiArray  -- [left_ticks, right_ticks]
    /qbot3/imu             sensor_msgs/Imu           -- orientation quaternion
    /qbot3/odom/reset      std_msgs/Bool             -- set True to zero pose

Outputs:
    /qbot3/odom            nav_msgs/Odometry         -- 30 Hz pose + twist
    TF                     odom -> base_link         -- broadcast each cycle

Calibration constants must match qbot3_base.py / motion_controller.py:
    TICKS_PER_METER = 2578.0
    WHEEL_BASE_M    = 0.235

==== DISTANCE_SOURCE — IMPORTANT ON QBOT3 #X ====
One of the wheel encoders on this unit is unreliable (it spikes huge tick
deltas, throwing the integrated pose into the kilometres). To work around
the broken encoder we use ONE wheel for distance integration. Yaw is still
taken from the IMU, so straight-line accuracy is unaffected.

Set `DISTANCE_SOURCE` to the side you trust:
    "left"    — use only the LEFT encoder for distance
    "right"   — use only the RIGHT encoder for distance
    "average" — original behaviour (use both, average them)

==== ROTATION-COMPENSATED SINGLE ENCODER ====
A naive single-encoder integration (d_center_m = d_right_m) injects fake
forward motion every time the robot rotates in place, because the chosen
wheel moves linearly even when the robot center does not. The fix below
uses the IMU yaw delta to subtract the rotation contribution:

    v_right = v_center + ω · (b / 2)
    v_left  = v_center − ω · (b / 2)

Per cycle, with Δyaw from the IMU and b = wheel base:
    d_center_m = d_right_m − Δyaw · (b / 2)        # right-wheel source
    d_center_m = d_left_m  + Δyaw · (b / 2)        # left-wheel source

Pure in-place rotation → Δyaw · (b / 2) exactly cancels the wheel delta
and d_center_m ≈ 0, so the SLAM map sees a clean pivot instead of a
phantom forward step on every turn.

==== COMMAND-AWARE TRANSLATION GATE ====
The geometric compensation above is only as good as the wheel-base
calibration. On a real QBot3 the effective wheel base shifts with tyre
wear and floor surface, so the residual after compensation can still
accumulate visible drift over a long spin (the user observed ~1 m of
y-drift on a single 270° pivot).

As a second line of defence we subscribe to /qbot3/cmd_vel and gate the
translation: when the host is NOT commanding forward motion (i.e.
|cmd_vel.linear.x| below a small threshold) we force d_center_m = 0
regardless of what the encoder says. This is robust to any wheel-base
mis-calibration because it uses the operator/AI's intent as ground truth
for "should be translating right now". Yaw integration is untouched, so
the robot still records the rotation correctly.
"""

from __future__ import annotations

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, Int64MultiArray
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped, Twist

from tf2_ros import TransformBroadcaster


# Pick the encoder that is reliable on YOUR robot.
# Edit this string and re-run the node — no other change needed.
DISTANCE_SOURCE = "right"   # "left" | "right" | "average"

# Per-cycle tick budget. At 30 Hz publish rate, 4000 ticks = ~1.55 m per
# cycle = 46 m/s — anything larger is a sensor glitch / rollover, not real
# motion, and is rejected.
MAX_TICK_DELTA_PER_CYCLE = 4000

# Command-aware translation gate — if the most recent /qbot3/cmd_vel has
# |linear.x| below this AND was published within CMD_VEL_FRESHNESS_S, the
# integrator forces d_center_m = 0 (pure rotation). Threshold is well below
# the smallest deliberate forward speed used by any skill (0.04 m/s in
# approach_object), so legitimate motion is never gated out.
CMD_VEL_TRANSLATION_THRESHOLD = 0.02   # m/s
CMD_VEL_FRESHNESS_S = 0.5


class OdometryNode(Node):
    """Encoder + IMU dead-reckoning odometry."""

    # ===== CALIBRATION (must match the rest of the QBot3 stack) =====
    TICKS_PER_METER = 2578.0
    WHEEL_BASE_M = 0.235

    # ===== FRAMES =====
    ODOM_FRAME = 'odom'
    BASE_FRAME = 'base_link'

    # ===== PUBLISH RATE =====
    PUBLISH_HZ = 30.0

    def __init__(self):
        super().__init__('odometry_node')

        # ===== STATE =====
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0  # radians, taken from IMU when available

        # Encoder bookkeeping
        self.left_ticks: int = 0
        self.right_ticks: int = 0
        self.prev_left_ticks: int | None = None
        self.prev_right_ticks: int | None = None

        # Latest IMU readings
        self.imu_yaw: float | None = None
        self.prev_imu_yaw: float | None = None
        self.angular_vel_z: float = 0.0

        # Latest commanded velocity (command-aware translation gate)
        self.cmd_linear_x: float = 0.0
        self.cmd_vel_ts = self.get_clock().now()
        self.cmd_vel_seen: bool = False

        # Twist estimate (filled each publish cycle)
        self.linear_vel_x = 0.0

        # Time of last publish (for dt)
        self.last_pub_time = self.get_clock().now()

        # ===== ROS I/O =====
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            Int64MultiArray, '/qbot3/encoders', self._encoder_cb, sensor_qos
        )
        self.create_subscription(
            Imu, '/qbot3/imu', self._imu_cb, sensor_qos
        )
        # Remote zero-pose: host publishes True here when the operator
        # presses "Reset SLAM" so the Pi resets without restarting.
        self.create_subscription(
            Bool, '/qbot3/odom/reset', self._reset_cb, sensor_qos
        )
        # Command-aware translation gate — listen to the same cmd_vel topic
        # that qbot3_base.py drives the motors from. Best-effort QoS so we
        # don't block the velocity loop when a host packet is delayed.
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            Twist, '/qbot3/cmd_vel', self._cmd_vel_cb, cmd_qos
        )

        self.odom_pub = self.create_publisher(Odometry, '/qbot3/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_timer(1.0 / self.PUBLISH_HZ, self._publish_odom)

        self.get_logger().info('Odometry Node Online')
        self.get_logger().info(
            f'   ticks/m={self.TICKS_PER_METER}  wheel_base={self.WHEEL_BASE_M} m'
        )
        self.get_logger().info(
            f'   Distance source: {DISTANCE_SOURCE}'
            + ('  (single-encoder mode — broken encoder protection)' if DISTANCE_SOURCE != 'average' else '')
        )
        self.get_logger().info(f'   Publishing /qbot3/odom @ {self.PUBLISH_HZ} Hz')

    # ----- Subscriber callbacks -----

    def _encoder_cb(self, msg: Int64MultiArray) -> None:
        if len(msg.data) >= 2:
            self.left_ticks = int(msg.data[0])
            self.right_ticks = int(msg.data[1])

    def _imu_cb(self, msg: Imu) -> None:
        q = msg.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.imu_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.angular_vel_z = float(msg.angular_velocity.z)

    def _cmd_vel_cb(self, msg: Twist) -> None:
        """Track the most-recent commanded forward velocity. The integrator
        consults this to gate phantom translation during pure rotation —
        operator/AI intent overrides a noisy single-encoder reading.
        """
        self.cmd_linear_x = float(msg.linear.x)
        self.cmd_vel_ts = self.get_clock().now()
        self.cmd_vel_seen = True

    def _reset_cb(self, msg: Bool) -> None:
        """Remote zero-pose request from the host."""
        if not bool(msg.data):
            return
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        # Reseed previous ticks so the next cycle's delta is 0
        self.prev_left_ticks = self.left_ticks
        self.prev_right_ticks = self.right_ticks
        self.prev_imu_yaw = self.imu_yaw
        self.linear_vel_x = 0.0
        self.get_logger().info('Pose reset by remote request')

    # ----- Pose integration -----

    def _integrate(self, dt: float) -> None:
        """Update (x, y, yaw) from latest encoder + IMU readings.

        Uses one wheel (configurable via DISTANCE_SOURCE at the top of this
        file) for the distance integration so a broken encoder can't pollute
        pose. Tick deltas above MAX_TICK_DELTA_PER_CYCLE are rejected (sensor
        glitch / 32-bit rollover); we re-seed and skip integration that cycle.
        """
        if self.prev_left_ticks is None or self.prev_right_ticks is None:
            self.prev_left_ticks = self.left_ticks
            self.prev_right_ticks = self.right_ticks
            # Seed prev_imu_yaw in lockstep so the first real cycle's d_yaw
            # spans exactly one period — otherwise the first integration
            # over-subtracts rotation and produces backward drift.
            if self.imu_yaw is not None:
                self.prev_imu_yaw = self.imu_yaw
            return

        d_left_ticks = self.left_ticks - self.prev_left_ticks
        d_right_ticks = self.right_ticks - self.prev_right_ticks

        # ---- Glitch rejection ----
        left_glitch = abs(d_left_ticks) > MAX_TICK_DELTA_PER_CYCLE
        right_glitch = abs(d_right_ticks) > MAX_TICK_DELTA_PER_CYCLE
        if left_glitch or right_glitch:
            self.get_logger().warn(
                f'Encoder glitch rejected (L={d_left_ticks}, R={d_right_ticks}); '
                f're-seeding without integrating'
            )
            self.prev_left_ticks = self.left_ticks
            self.prev_right_ticks = self.right_ticks
            if self.imu_yaw is not None:
                self.prev_imu_yaw = self.imu_yaw
            self.linear_vel_x = 0.0
            return

        # Commit prev for the next cycle
        self.prev_left_ticks = self.left_ticks
        self.prev_right_ticks = self.right_ticks

        d_left_m = d_left_ticks / self.TICKS_PER_METER
        d_right_m = d_right_ticks / self.TICKS_PER_METER

        # ---- Δyaw used for rotation compensation ----
        # Two sources, in order of preference:
        #  1. IMU yaw differencing — most accurate when the bias-corrected
        #     yaw is solid, immune to short integration drift.
        #  2. Angular-velocity × dt — fallback when prev_imu_yaw isn't
        #     seeded yet, or sanity-cross-check.
        d_yaw = 0.0
        if self.imu_yaw is not None and self.prev_imu_yaw is not None:
            d_yaw = self._wrap_pi(self.imu_yaw - self.prev_imu_yaw)
        elif abs(self.angular_vel_z) > 1e-6:
            d_yaw = self.angular_vel_z * dt
        if self.imu_yaw is not None:
            self.prev_imu_yaw = self.imu_yaw

        half_base = 0.5 * self.WHEEL_BASE_M

        # ---- Pick the encoder source (with rotation compensation) ----
        if DISTANCE_SOURCE == 'left':
            d_center_m = d_left_m + d_yaw * half_base
        elif DISTANCE_SOURCE == 'right':
            d_center_m = d_right_m - d_yaw * half_base
        else:
            d_center_m = 0.5 * (d_left_m + d_right_m)

        # ---- Command-aware translation gate ----
        # When the host is not commanding forward motion, force zero
        # translation regardless of what the wheel says. This is the
        # bulletproof fix for wheel-base mis-calibration: the geometric
        # ω·(b/2) compensation above leaves residuals when b is wrong, and
        # those residuals integrate into metres of drift over a long spin.
        # Using operator/AI intent as ground truth side-steps that entirely.
        # Yaw still updates from the IMU below, so pivots are recorded.
        if self.cmd_vel_seen:
            cmd_age_s = (self.get_clock().now() - self.cmd_vel_ts).nanoseconds / 1e9
            if cmd_age_s < CMD_VEL_FRESHNESS_S \
                    and abs(self.cmd_linear_x) < CMD_VEL_TRANSLATION_THRESHOLD:
                d_center_m = 0.0

        # ---- Encoder-zero defensive gate ----
        # Even after cmd_vel goes stale (skill finished, no commands being
        # sent), if both wheels reported zero tick delta this cycle the
        # robot physically did not translate. Without this gate, a
        # non-zero d_yaw (from IMU bias drift, or honest small head
        # wiggle) would inject a fake d_yaw·(b/2) translation per cycle
        # via the rotation-compensation term, and the SLAM trail keeps
        # growing while the robot sits still. If the wheels didn't turn,
        # the centre didn't move — that's geometry, not a heuristic.
        if d_left_ticks == 0 and d_right_ticks == 0:
            d_center_m = 0.0

        if self.imu_yaw is not None:
            self.yaw = self.imu_yaw
        else:
            # Encoder-derived yaw fallback only useful when both encoders work
            d_theta = (d_right_m - d_left_m) / self.WHEEL_BASE_M
            self.yaw = self._wrap_pi(self.yaw + d_theta)

        self.x += d_center_m * math.cos(self.yaw)
        self.y += d_center_m * math.sin(self.yaw)

        self.linear_vel_x = d_center_m / dt if dt > 1e-6 else 0.0

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _yaw_to_quat(yaw: float) -> Quaternion:
        q = Quaternion()
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q

    # ----- Publish cycle -----

    def _publish_odom(self) -> None:
        now = self.get_clock().now()
        dt = (now - self.last_pub_time).nanoseconds / 1e9
        self.last_pub_time = now
        if dt <= 0.0:
            return

        self._integrate(dt)
        stamp = now.to_msg()
        quat = self._yaw_to_quat(self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.ODOM_FRAME
        odom.child_frame_id = self.BASE_FRAME

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = quat

        odom.twist.twist.linear.x = self.linear_vel_x
        odom.twist.twist.angular.z = self.angular_vel_z

        # Diagonal covariance — position trusted moderately, orientation tight (IMU-fused)
        pose_cov = [0.0] * 36
        pose_cov[0] = 0.05   # x
        pose_cov[7] = 0.05   # y
        pose_cov[14] = 1e6   # z (unused, large)
        pose_cov[21] = 1e6   # roll (unused)
        pose_cov[28] = 1e6   # pitch (unused)
        pose_cov[35] = 0.02  # yaw
        odom.pose.covariance = pose_cov

        twist_cov = [0.0] * 36
        twist_cov[0] = 0.05
        twist_cov[7] = 1e6
        twist_cov[14] = 1e6
        twist_cov[21] = 1e6
        twist_cov[28] = 1e6
        twist_cov[35] = 0.05
        odom.twist.covariance = twist_cov

        self.odom_pub.publish(odom)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.ODOM_FRAME
        tf_msg.child_frame_id = self.BASE_FRAME
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = quat
        self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = OdometryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
