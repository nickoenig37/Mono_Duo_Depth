#!/usr/bin/env python3
import os
import re
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np  # Added for .npy saving
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from datetime import datetime


class DatasetRecorderNode(Node):
    def __init__(self):
        super().__init__('dataset_recorder_node')

        # Parameters
        self.declare_parameter('dataset_dir', 'dataset')
        self.declare_parameter('capture_period_sec', 1.0)
        self.declare_parameter('warmup_seconds', 1.0)
        self.declare_parameter('jpeg_quality', 95)
        self.declare_parameter('png_compression', 3)
        self.declare_parameter('depth_save_millimeters', True)

        # Topics
        self.declare_parameter('left_topic', 'camera/left/image_raw')
        self.declare_parameter('right_topic', 'camera/right/image_raw')
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')

        # Sync strategy (time-window search instead of message_filters)
        self.declare_parameter('sync_time_slop', 0.1)  # seconds, allowed skew between topics
        self.declare_parameter('buffer_size', 60)      # frames to keep per topic
        self.declare_parameter('max_wait_sec', 0.5)    # wait window for missing frames on tick
        self.declare_parameter('require_all', True)    # require all topics to save
        self.declare_parameter('adaptive_sync', True)  # try tight sync first, fall back if needed
        self.declare_parameter('remove_used_frames', True)  # clear consumed frames from buffers

        # Read params
        self.dataset_dir = os.path.abspath(self.get_parameter('dataset_dir').value)
        self.capture_period_sec = float(self.get_parameter('capture_period_sec').value)
        self.warmup_seconds = float(self.get_parameter('warmup_seconds').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.png_compression = int(self.get_parameter('png_compression').value)
        self.depth_save_mm = bool(self.get_parameter('depth_save_millimeters').value)

        self.left_topic = str(self.get_parameter('left_topic').value)
        self.right_topic = str(self.get_parameter('right_topic').value)
        self.color_topic = str(self.get_parameter('color_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)

        self.sync_time_slop = float(self.get_parameter('sync_time_slop').value)
        self.buffer_size = int(self.get_parameter('buffer_size').value)
        self.max_wait_sec = float(self.get_parameter('max_wait_sec').value)
        self.require_all = bool(self.get_parameter('require_all').value)
        self.adaptive_sync = bool(self.get_parameter('adaptive_sync').value)
        self.remove_used_frames = bool(self.get_parameter('remove_used_frames').value)

        # Prepare
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.bridge = CvBridge()
        
        # Create a run-level folder with timestamp
        run_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.dataset_dir, run_timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Frame counter for this run (always starts at 1)
        self.frame_index = 1
        
        # Sync statistics
        self.sync_stats = {
            'total_attempts': 0,
            'successful_syncs': 0,
            'tight_syncs': 0,  # within half of slop
            'max_skew_ms': 0.0
        }
        
        self.start_time_ns = self.get_clock().now().nanoseconds
        self.last_save_ns = 0

        # Deques for recent messages per topic
        self.buf_left = deque(maxlen=self.buffer_size)
        self.buf_right = deque(maxlen=self.buffer_size)
        self.buf_color = deque(maxlen=self.buffer_size)
        self.buf_depth = deque(maxlen=self.buffer_size)

        self.sub_left = self.create_subscription(Image, self.left_topic, self._cb_left, 10)
        self.sub_right = self.create_subscription(Image, self.right_topic, self._cb_right, 10)
        self.sub_color = self.create_subscription(Image, self.color_topic, self._cb_color, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self._cb_depth, 10)

        # Timer to trigger captures periodically
        self.timer = self.create_timer(self.capture_period_sec, self._on_tick)

        self.get_logger().info(
            f"Recording session started: {run_timestamp}"
        )
        self.get_logger().info(
            f"Saving to '{self.run_dir}' every {self.capture_period_sec}s (warmup {self.warmup_seconds}s)."
        )
        self.get_logger().info(
            f"Topics: left={self.left_topic}, right={self.right_topic}, color={self.color_topic}, depth={self.depth_topic}"
        )

    # ---------- Callbacks to store recent messages ----------
    def _cb_left(self, msg: Image):
        self.buf_left.append(msg)
        if len(self.buf_left) == 1:
            self.get_logger().info(f"✓ Receiving left images on {self.left_topic}")

    def _cb_right(self, msg: Image):
        self.buf_right.append(msg)
        if len(self.buf_right) == 1:
            self.get_logger().info(f"✓ Receiving right images on {self.right_topic}")

    def _cb_color(self, msg: Image):
        self.buf_color.append(msg)
        if len(self.buf_color) == 1:
            self.get_logger().info(f"✓ Receiving color images on {self.color_topic}")

    def _cb_depth(self, msg: Image):
        self.buf_depth.append(msg)
        if len(self.buf_depth) == 1:
            self.get_logger().info(f"✓ Receiving depth images on {self.depth_topic}")

    # ---------- Timer tick: find best-matching set and save ----------
    def _on_tick(self):
        now_ns = self.get_clock().now().nanoseconds
        # Enforce warmup period
        if (now_ns - self.start_time_ns) < int(self.warmup_seconds * 1e9):
            return

        self.sync_stats['total_attempts'] += 1

        # Log buffer status every tick for debugging
        self.get_logger().debug(
            f"Buffer status: left={len(self.buf_left)}, right={len(self.buf_right)}, "
            f"color={len(self.buf_color)}, depth={len(self.buf_depth)}"
        )

        deadline = time.time() + self.max_wait_sec
        msgs = None
        sync_quality = None
        
        # Adaptive sync: try progressively looser slop values
        if self.adaptive_sync:
            # Try tight sync first (half slop), then configured slop, then 2x slop
            slop_attempts = [
                self.sync_time_slop * 0.5,  # Tight: 50% of configured
                self.sync_time_slop,         # Normal: configured value
                self.sync_time_slop * 2.0,   # Loose: 2x configured (fallback)
            ]
            
            for attempt_slop in slop_attempts:
                while time.time() < deadline:
                    result = self._pick_synced_set(attempt_slop)
                    if result is not None:
                        msgs, sync_quality = result
                        break
                    time.sleep(0.01)
                
                if msgs is not None:
                    if attempt_slop == slop_attempts[0]:
                        self.sync_stats['tight_syncs'] += 1
                    break
        else:
            # Non-adaptive: just use configured slop
            while time.time() < deadline:
                result = self._pick_synced_set(self.sync_time_slop)
                if result is not None:
                    msgs, sync_quality = result
                    break
                time.sleep(0.01)

        if msgs is None:
            self.get_logger().warn(
                f"No synchronized set found (buffers: L={len(self.buf_left)} R={len(self.buf_right)} "
                f"C={len(self.buf_color)} D={len(self.buf_depth)}). "
                f"Try increasing sync_time_slop (now {self.sync_time_slop}s) or check topics are publishing."
            )
            
            # Clear old frames if buffers are full to prevent clogging
            if self.remove_used_frames:
                self._clear_old_frames()
            return

        self.sync_stats['successful_syncs'] += 1
        left_msg, right_msg, color_msg, depth_msg = msgs

        # Convert
        try:
            left_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
            right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
            color_cv = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Color conversion failed: {e}")
            return

        try:
            if depth_msg.encoding in ('16UC1', 'mono16'):
                depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
            else:
                depth_f32 = self.bridge.imgmsg_to_cv2(depth_msg)  # likely 32FC1 meters
                depth_cv = (depth_f32 * 1000.0).clip(0, 65535).astype('uint16') if self.depth_save_mm else depth_f32
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")
            return

        # Create frame folder with zero-padded index (000001, 000002, etc.)
        folder_name = f"{self.frame_index:06d}"
        folder_path = os.path.join(self.run_dir, folder_name)
        
        if os.path.exists(folder_path):
            self.get_logger().error(f"Folder already exists: {folder_path}")
            return
        os.makedirs(folder_path, exist_ok=False)

        # Paths (changed to .npy only per new requirement)
        left_path = os.path.join(folder_path, 'left.npy')
        right_path = os.path.join(folder_path, 'right.npy')
        color_path = os.path.join(folder_path, 'color.npy')
        depth_path = os.path.join(folder_path, 'depth.npy')
        meta_path = os.path.join(folder_path, 'meta.txt')

        # Save as raw arrays (.npy).
        # Convert BGR (cv2) to RGB before saving to avoid blue-tinted visualization when loaded with libraries expecting RGB.
        # Depth saved either as uint16 (millimeters) or float32 (meters) depending on depth_save_mm.
        try:
            left_rgb = cv2.cvtColor(left_cv, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_cv, cv2.COLOR_BGR2RGB)
            color_rgb = cv2.cvtColor(color_cv, cv2.COLOR_BGR2RGB)

            np.save(left_path, left_rgb)
            np.save(right_path, right_rgb)
            np.save(color_path, color_rgb)
            np.save(depth_path, depth_cv)
        except Exception as e:
            self.get_logger().error(f"Failed to save .npy arrays: {e}")
            return

        # Meta
        try:
            def ns(stamp):
                return stamp.sec * 1_000_000_000 + stamp.nanosec

            with open(meta_path, 'w') as f:
                f.write(f"Timestamp (UTC): {datetime.utcnow().isoformat()}Z\n")
                f.write(f"Frame index: {self.frame_index}\n")
                f.write(f"Frame folder: {folder_name}\n")
                f.write(f"Left stamp (ns): {ns(left_msg.header.stamp)}\n")
                f.write(f"Right stamp (ns): {ns(right_msg.header.stamp)}\n")
                f.write(f"Color stamp (ns): {ns(color_msg.header.stamp)}\n")
                f.write(f"Depth stamp (ns): {ns(depth_msg.header.stamp)}\n")
                
                # Add sync quality metrics
                if sync_quality:
                    f.write(f"Max skew (ms): {sync_quality['max_skew_ms']:.2f}\n")
                    f.write(f"Avg skew (ms): {sync_quality['avg_skew_ms']:.2f}\n")
                    f.write(f"Sync quality: {sync_quality['quality']}\n")
                # New metadata for .npy arrays
                f.write(f"Left shape (RGB): {left_rgb.shape}, dtype: {left_rgb.dtype}\n")
                f.write(f"Right shape (RGB): {right_rgb.shape}, dtype: {right_rgb.dtype}\n")
                f.write(f"Color shape (RGB): {color_rgb.shape}, dtype: {color_rgb.dtype}\n")
                f.write(f"Depth shape: {depth_cv.shape}, dtype: {depth_cv.dtype}\n")
                f.write(f"Depth units: {'millimeters (uint16)' if self.depth_save_mm else 'meters (float32)'}\n")
        except Exception as e:
            self.get_logger().warn(f"Failed to write meta: {e}")

        # Update stats
        if sync_quality:
            self.sync_stats['max_skew_ms'] = max(
                self.sync_stats['max_skew_ms'], 
                sync_quality['max_skew_ms']
            )

        # Remove used frames from buffers to prevent clogging
        if self.remove_used_frames:
            self._remove_frames_before(left_msg, right_msg, color_msg, depth_msg)

        self.last_save_ns = now_ns
        self.frame_index += 1
        
        # Log with sync quality
        quality_str = f" (skew: {sync_quality['max_skew_ms']:.1f}ms, {sync_quality['quality']})" if sync_quality else ""
        self.get_logger().info(f"Saved frame {self.frame_index - 1:06d}{quality_str} → {folder_path}")
        
        # Periodic stats
        if self.frame_index % 50 == 1:
            success_rate = 100.0 * self.sync_stats['successful_syncs'] / max(1, self.sync_stats['total_attempts'])
            tight_rate = 100.0 * self.sync_stats['tight_syncs'] / max(1, self.sync_stats['successful_syncs'])
            self.get_logger().info(
                f"Sync stats: {success_rate:.1f}% success, {tight_rate:.1f}% tight, "
                f"max skew: {self.sync_stats['max_skew_ms']:.1f}ms"
            )

    # ---------- Helper: choose best-matching messages within slop ----------
    def _pick_synced_set(self, slop_sec: float) -> Optional[Tuple[Tuple[Image, Image, Image, Image], dict]]:
        """
        Pick best synchronized set of messages.
        Returns: ((left, right, color, depth), sync_quality) or None
        """
        # Make local copies to avoid mutation during selection
        left = list(self.buf_left)
        right = list(self.buf_right)
        color = list(self.buf_color)
        depth = list(self.buf_depth)

        if self.require_all and (not left or not right or not color or not depth):
            return None
        if not left or not right or not color or not depth:
            # If partial allowed, bail early (we require all by default)
            return None

        def to_ns(msg: Image) -> int:
            return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # Build candidate times (use median of latest stamps to reduce outliers)
        latest = [to_ns(left[-1]), to_ns(right[-1]), to_ns(color[-1]), to_ns(depth[-1])]

        # In dataset_recorder_node.py inside _pick_synced_set
        # TRYING FOR CONFIG TO BOTTLENECKING VALS
        target_ns = min(latest)
        # target_ns = sorted(latest)[len(latest)//2]
        slop_ns = int(slop_sec * 1e9)

        def nearest_within(buf, t_ns):
            best = None
            best_diff = None
            for m in buf:
                d = abs(to_ns(m) - t_ns)
                if best is None or d < best_diff:
                    best, best_diff = m, d
            if best is not None and best_diff <= slop_ns:
                return best
            return None

        l = nearest_within(left, target_ns)
        r = nearest_within(right, target_ns)
        c = nearest_within(color, target_ns)
        d = nearest_within(depth, target_ns)

        if any(x is None for x in (l, r, c, d)):
            return None
        
        # Calculate sync quality
        stamps_ns = [to_ns(l), to_ns(r), to_ns(c), to_ns(d)]
        min_stamp = min(stamps_ns)
        max_stamp = max(stamps_ns)
        skews_ns = [abs(s - target_ns) for s in stamps_ns]
        
        max_skew_ms = (max_stamp - min_stamp) / 1e6
        avg_skew_ms = sum(skews_ns) / len(skews_ns) / 1e6
        
        # Quality rating
        if max_skew_ms < slop_sec * 500:  # < 50% of slop
            quality = "excellent"
        elif max_skew_ms < slop_sec * 1000:  # < 100% of slop
            quality = "good"
        else:
            quality = "acceptable"
        
        sync_quality = {
            'max_skew_ms': max_skew_ms,
            'avg_skew_ms': avg_skew_ms,
            'quality': quality,
            'slop_used_ms': slop_sec * 1000
        }
        
        return ((l, r, c, d), sync_quality)

    def _remove_frames_before(self, left_msg, right_msg, color_msg, depth_msg):
        """Remove all frames older than the ones we just used to prevent buffer clogging"""
        def to_ns(msg: Image) -> int:
            return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        
        # Find oldest timestamp we're keeping
        oldest_ns = min(to_ns(left_msg), to_ns(right_msg), to_ns(color_msg), to_ns(depth_msg))
        
        # Remove frames older than this
        def filter_buffer(buf):
            return deque([m for m in buf if to_ns(m) >= oldest_ns], maxlen=buf.maxlen)
        
        self.buf_left = filter_buffer(self.buf_left)
        self.buf_right = filter_buffer(self.buf_right)
        self.buf_color = filter_buffer(self.buf_color)
        self.buf_depth = filter_buffer(self.buf_depth)

    def _clear_old_frames(self):
        """Clear oldest 50% of frames from all buffers when sync fails to prevent clogging"""
        for buf in [self.buf_left, self.buf_right, self.buf_color, self.buf_depth]:
            if len(buf) > 10:
                # Keep only the newest half
                keep_count = len(buf) // 2
                items = list(buf)
                buf.clear()
                for item in items[-keep_count:]:
                    buf.append(item)
        
        self.get_logger().debug("Cleared old frames from buffers to prevent clogging")


def main(args=None):
    rclpy.init(args=args)
    node = DatasetRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
