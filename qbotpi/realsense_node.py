import time

import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from cv_bridge import CvBridge

class RealSenseNode(Node):
    def __init__(self):
        super().__init__('realsense_node')

        # --- PUBLISHERS ---
        # 1. RGB (Visual)
        self.pub_rgb = self.create_publisher(CompressedImage, '/camera/color/compressed', 10)
        # 2. Heatmap (Visual for Human)
        self.pub_depth_vis = self.create_publisher(CompressedImage, '/camera/depth/visual', 10)
        # 3. Raw Data (Data for Robot Math - Lossless PNG)
        self.pub_depth_raw = self.create_publisher(CompressedImage, '/camera/depth/raw', 10)
        # 4. Diagnostics
        self.pub_diag = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        # 5. Camera Info
        self.pub_info = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)

        # --- STATE ---
        self.pipeline = None
        self.profile = None
        self._started = False        # True only while the pipeline is streaming
        self._consec_errors = 0      # consecutive frame failures (drives recovery)

        # --- SOFTWARE FILTERS (stateless objects, build once) ---
        # Decimation: Lowers resolution to smooth data
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)

        # Spatial: Smooths surface data
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.holes_fill, 0)

        # Hole Filling: Fills the remaining black spots
        self.hole_filling = rs.hole_filling_filter()

        self.align = rs.align(rs.stream.color)
        self.bridge = CvBridge()

        # --- START (with retry + hardware reset on a wedged device) ---
        if not self._open_camera():
            raise RuntimeError(
                "RealSense failed to start after retries. Unplug/replug the "
                "camera or check it is on a USB3 port (lsusb -t should show 5000M)."
            )

        # Timer: 15 FPS limit to save Wi-Fi
        self.timer = self.create_timer(0.066, self.timer_callback)

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    def _open_camera(self, max_attempts: int = 5) -> bool:
        """Start the pipeline, recovering a wedged USB/V4L2 device.

        The classic `xioctl(VIDIOC_S_FMT) failed ... Input/output error`
        means the camera was left in a bad state by a previous crashed run.
        A hardware_reset() forces a USB re-enumeration which clears it; we
        retry a few times with a settle delay between attempts.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                # 640x480 is stable and fast for Pi 4
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.profile = self.pipeline.start(config)
                self._started = True
                self._consec_errors = 0
                self._apply_sensor_options()
                self.get_logger().info(
                    f"RealSense pipeline started (attempt {attempt}/{max_attempts})")
                return True
            except RuntimeError as e:
                self._started = False
                self.get_logger().error(
                    f"Camera start failed (attempt {attempt}/{max_attempts}): {e}")
                self._hardware_reset()
                time.sleep(4.0)   # allow USB to re-enumerate after the reset
        return False

    def _hardware_reset(self):
        """Issue a USB hardware reset on the first connected RealSense."""
        try:
            devices = rs.context().query_devices()
            if len(devices) > 0:
                self.get_logger().warn("Issuing hardware_reset() on RealSense device...")
                devices[0].hardware_reset()
            else:
                self.get_logger().error("No RealSense device found to reset.")
        except Exception as e:
            self.get_logger().error(f"hardware_reset failed: {e}")

    def _apply_sensor_options(self):
        """(Re)apply depth-sensor presets after a (re)start."""
        try:
            depth_sensor = self.profile.get_device().first_depth_sensor()
            # 1. High Density Preset (Fills holes)
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 4)
            # 2. Laser Projector ON (Crucial for white walls)
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
        except Exception as e:
            self.get_logger().error(f"Failed to apply sensor options: {e}")

    def _recover(self):
        """Stream died mid-run — release and reopen instead of spamming errors."""
        self.get_logger().warn("Stream appears dead — attempting pipeline restart...")
        self.stop()
        if not self._open_camera():
            self.get_logger().error("Pipeline restart failed — will retry on next tick.")

    def timer_callback(self):
        # If the camera isn't streaming, try to bring it back rather than
        # hammering wait_for_frames() (which raises "cannot be called before
        # start()" every tick and floods the log).
        if not self._started:
            self._recover()
            return
        try:
            # 1. Capture
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            self._consec_errors = 0

            # 2. Align Depth to Color
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame: return

            # 3. Apply Filters
            depth_frame = self.decimation.process(depth_frame)
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)

            # 4. Convert to Numpy
            depth_raw = np.asanyarray(depth_frame.get_data())
            color_img = np.asanyarray(color_frame.get_data())
            
            # Resize depth back to 640x480 (Decimation shrinks it)
            if depth_raw.shape != color_img.shape[:2]:
                depth_raw = cv2.resize(depth_raw, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # --- PUBLISH 1: RGB ---
            self.pub_rgb.publish(self.bridge.cv2_to_compressed_imgmsg(color_img))
            
            # --- PUBLISH 2: SMART HEATMAP ---
            # A. Clip to 3 meters (3000mm)
            depth_clipped = np.clip(depth_raw, 0, 3000)
            
            # B. Normalize 0-3000 to 0-255
            depth_norm = cv2.convertScaleAbs(depth_clipped, alpha=255.0/3000.0)
            
            # C. Invert (So 0m/Close is High Value)
            depth_inverted = 255 - depth_norm
            
            # D. Apply JET Colormap (High=Red, Low=Blue)
            # Result: Close = Red, Far = Blue
            depth_colormap = cv2.applyColorMap(depth_inverted, cv2.COLORMAP_JET)
            
            # E. Make invalid pixels black
            mask = (depth_raw == 0)
            depth_colormap[mask] = [0, 0, 0] 

            self.pub_depth_vis.publish(self.bridge.cv2_to_compressed_imgmsg(depth_colormap))

            # --- PUBLISH 3: RAW DATA ---
            # Send raw 16-bit integers using PNG (Lossless)
            raw_msg = self.bridge.cv2_to_compressed_imgmsg(depth_raw, dst_format='png')
            self.pub_depth_raw.publish(raw_msg)
            
            # --- PUBLISH 5: CAMERA INFO ---
            self._publish_camera_info(color_frame)

            # --- PUBLISH 4: DIAGNOSTICS ---
            # self._publish_depth_diagnostics(depth_raw)

        except Exception as e:
            self._consec_errors += 1
            # Throttle: log once, then every 30th failure (~2 s) to avoid the flood.
            if self._consec_errors == 1 or self._consec_errors % 30 == 0:
                self.get_logger().error(
                    f"Frame Error (x{self._consec_errors}): {e}")
            # Sustained failures mean the stream is dead — reopen the device.
            if self._consec_errors >= 30:
                self._started = False   # next tick runs _recover()

    def stop(self):
        """Release the camera. Called on shutdown AND during recovery.

        The previous version was missing this method entirely, so the
        Ctrl+C cleanup path (main()) crashed before pipeline.stop() ran,
        leaving the camera in a wedged USB state that made the NEXT launch
        fail with `xioctl(VIDIOC_S_FMT) failed ... Input/output error`.
        """
        try:
            if self._started and self.pipeline is not None:
                self.pipeline.stop()
        except Exception as e:
            self.get_logger().error(f"Error stopping pipeline: {e}")
        finally:
            self._started = False

    def _publish_depth_diagnostics(self, depth_raw: np.ndarray):
        """Publish depth quality diagnostics."""
        # Calculate quality metrics
        valid_mask = (depth_raw > 100) & (depth_raw < 5000)
        valid_count = np.sum(valid_mask)
        total_pixels = depth_raw.size
        valid_percentage = (valid_count / total_pixels) * 100.0
        
        if valid_count > 0:
            valid_depths = depth_raw[valid_mask] / 1000.0
            mean_distance = float(np.mean(valid_depths))
        else:
            mean_distance = 0.0
        
        # Calculate coverage (grid-based)
        h, w = depth_raw.shape
        grid_size = 4
        covered = 0
        for i in range(grid_size):
            for j in range(grid_size):
                cell_h, cell_w = h // grid_size, w // grid_size
                cell = valid_mask[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                if np.sum(cell) > (cell_h * cell_w * 0.3):
                    covered += 1
        coverage = (covered / (grid_size * grid_size)) * 100.0
        
        # Create diagnostic message
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        status = DiagnosticStatus()
        status.name = "Depth Camera"
        status.hardware_id = "realsense_d435i"
        
        if valid_percentage > 70 and coverage > 60:
            status.level = DiagnosticStatus.OK
            status.message = "Depth quality excellent"
        elif valid_percentage > 50 and coverage > 40:
            status.level = DiagnosticStatus.WARN
            status.message = "Depth quality fair"
        else:
            status.level = DiagnosticStatus.ERROR
            status.message = "Depth quality poor"
        
        status.values.append(KeyValue(key="valid_percentage", value=f"{valid_percentage:.1f}"))
        status.values.append(KeyValue(key="coverage_percentage", value=f"{coverage:.1f}"))
        status.values.append(KeyValue(key="mean_distance_m", value=f"{mean_distance:.2f}"))
        
        msg.status.append(status)
        self.pub_diag.publish(msg)

    def _publish_camera_info(self, frame):
        """Publish standard CameraInfo message."""
        intrinsics = frame.profile.as_video_stream_profile().get_intrinsics()
        
        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.width = intrinsics.width
        msg.height = intrinsics.height
        msg.distortion_model = "plumb_bob"
        msg.d = [float(c) for c in intrinsics.coeffs]
        
        # Intrinsic Matrix K
        msg.k = [float(intrinsics.fx), 0.0, float(intrinsics.ppx),
                 0.0, float(intrinsics.fy), float(intrinsics.ppy),
                 0.0, 0.0, 1.0]
        
        # Projection Matrix P (Assume identity rotation / no translation for single cam)
        msg.p = [float(intrinsics.fx), 0.0, float(intrinsics.ppx), 0.0,
                 0.0, float(intrinsics.fy), float(intrinsics.ppy), 0.0,
                 0.0, 0.0, 1.0, 0.0]
                 
        self.pub_info.publish(msg)

    # _publish_depth_diagnostics disabled to prevent rmw_cyclonedds serialization errors
    def _publish_depth_diagnostics_DISABLED(self, depth_raw: np.ndarray):
        pass

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
