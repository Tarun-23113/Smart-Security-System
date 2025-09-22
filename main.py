import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from collections import defaultdict, deque
import json
from datetime import datetime
import threading
import queue
import logging

logging.getLogger('ultralytics').setLevel(logging.ERROR)

class SmartSecuritySystem:
    """
    Advanced Computer Vision Security System
    Features: Real-time object detection, tracking, and anomaly detection
    """

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        """Initialize the security system with YOLO model and tracking"""
        print("Initializing Smart Security System...")

        self.model = YOLO(model_path)
        self.model.verbose = False
        self.confidence_threshold = confidence_threshold

        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.object_counts = defaultdict(int)
        self.detection_log = []

        self.fps_counter = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = time.time()

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=100
        )

        self.alert_queue = queue.Queue()
        self.alert_threshold = 7  # Changed to # people for alert
        self.frame_count = 0

        # Summary statistics
        self.total_persons_detected = 0
        self.total_vehicles_detected = 0
        self.max_people_at_once = 0
        self.max_vehicles_at_once = 0
        self.total_alerts_generated = 0

        print("System initialized successfully!")

    def detect_objects(self, frame):
        """
        Perform object detection using YOLO
        Returns detection results with bounding boxes, classes, and confidence scores
        """
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            persist=True,
            classes=[0, 1, 2, 3, 5, 7],  # [person, bicycle, car, motorcycle, bus, truck]
            verbose=False
        )
        return results

    def process_detections(self, results, frame):
        """
        Process YOLO detection results and update tracking information
        """
        current_detections = {
            'persons': 0,
            'vehicles': 0,
            'total_objects': 0
        }

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.float().cpu().tolist()

            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                # Update tracking history
                x, y, w, h = box
                center = (int(x), int(y))
                self.track_history[track_id].append(center)

                # Count objects by type
                if cls == 0:
                    current_detections['persons'] += 1
                    self.total_persons_detected = max(self.total_persons_detected, current_detections['persons'])
                elif cls in [1, 2, 3, 4, 5, 7]:
                    current_detections['vehicles'] += 1
                    self.total_vehicles_detected = max(self.total_vehicles_detected, current_detections['vehicles'])

                current_detections['total_objects'] += 1

                # Draw tracking trail
                points = np.array(self.track_history[track_id]).reshape((-1, 1, 2))
                if len(points) > 1:
                    cv2.polylines(frame, [points], isClosed=False,
                                color=(0, 255, 0), thickness=2)

                # Draw detection box
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)

                color = (0, 255, 0) if cls == 0 else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{self.model.names[cls]} ID:{track_id} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update max counts for summary
        self.max_people_at_once = max(self.max_people_at_once, current_detections['persons'])
        self.max_vehicles_at_once = max(self.max_vehicles_at_once, current_detections['vehicles'])

        return current_detections

    def motion_detection(self, frame):
        """
        Detect motion using background subtraction
        Returns motion mask and motion percentage
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Calculate motion percentage
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_percentage = (motion_pixels / total_pixels) * 100

        return fg_mask, motion_percentage

    def check_anomalies(self, detections, motion_percentage):
        """
        Check for anomalous situations and generate alerts
        """
        alerts = []

        # Too many people detected
        if detections['persons'] > self.alert_threshold:
            alerts.append(f"Crowd Alert: {detections['persons']} people detected")

        # Log alerts (but don't print during video)
        for alert in alerts:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            alert_data = {
                'timestamp': timestamp,
                'alert': alert,
                'detections': detections,
                'motion': motion_percentage
            }
            self.detection_log.append(alert_data)
            self.total_alerts_generated += 1

        return alerts

    def calculate_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        if hasattr(self, 'last_time'):
            fps = 1.0 / (current_time - self.last_time)
            self.fps_counter.append(fps)
        self.last_time = current_time

        if self.fps_counter:
            return np.mean(self.fps_counter)
        return 0

    def draw_analytics(self, frame, detections, motion_percentage, fps, alerts):
        """
        Draw analytics overlay on frame with clean alert box
        """
        height, width = frame.shape[:2]

        alert_height = 100
        alert_overlay = frame.copy()
        cv2.rectangle(alert_overlay, (10, 10), (width-10, alert_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, alert_overlay, 0.2, 0)

        cv2.rectangle(frame, (10, 10), (width-10, alert_height), (0, 255, 255), 2)
        cv2.putText(frame, "ALERTS & STATUS", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if alerts:
            alert_text = f"🚨 {alerts[-1]}"
            cv2.putText(frame, alert_text, (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "✅ All Clear - No alerts", (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        stats_text = f"👥 Persons: {detections['persons']} | 🚗 Vehicles: {detections['vehicles']} | 📊 Motion: {motion_percentage:.1f}% | ⚡ FPS: {fps:.1f}"
        cv2.putText(frame, stats_text, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (width-250, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run_security_system(self, video_source=0, save_video=False):
        """
        Main loop for running the security system
        """
        print("Starting security system...")

        cap = cv2.VideoCapture(video_source)

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps > 0 and original_fps < 100:
            frame_delay = max(1, int(1000 / original_fps))
        else:
            frame_delay = 33

        print(f"Video FPS: {original_fps}, Using frame delay: {frame_delay}ms")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        self.output_video_saved = False
        if save_video:
            out = cv2.VideoWriter('security_output.mp4', fourcc, 20.0, (1280, 720))
            self.output_video_saved = True

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (1280, 720))
                original_frame = frame.copy()
                results = self.detect_objects(frame)
                detections = self.process_detections(results, frame)

                # Motion detection
                motion_mask, motion_percentage = self.motion_detection(original_frame)

                # Check for anomalies
                alerts = self.check_anomalies(detections, motion_percentage)

                # Calculate FPS
                fps = self.calculate_fps()

                self.draw_analytics(frame, detections, motion_percentage, fps, alerts)

                if save_video and out:
                    out.write(frame)

                cv2.imshow('Smart Security System - Video Analysis', frame)

                self.frame_count += 1

                if isinstance(video_source, str):
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nShutting down security system...")

        finally:

            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

            # Save detection log
            self.save_detection_log()

            # Print comprehensive summary
            self.print_final_summary()

    def save_detection_log(self):
        """Save detection log to JSON file"""
        with open(f'detection_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(self.detection_log, f, indent=2)

    def print_final_summary(self):
        """Print comprehensive summary of detection results"""
        print("\n" + "="*60)
        print("🎯 SMART SECURITY SYSTEM - FINAL SUMMARY")
        print("="*60)

        # Processing statistics
        total_time = time.time() - self.start_time
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0

        print(f"📊 PROCESSING STATISTICS:")
        print(f"   • Total Frames Processed: {self.frame_count}")
        print(f"   • Total Processing Time: {total_time:.1f} seconds")
        print(f"   • Average FPS: {avg_fps:.1f}")

        # Detection summary
        print(f"\n👥 DETECTION SUMMARY:")
        print(f"   • Maximum People Detected (at once): {self.max_people_at_once}")
        print(f"   • Maximum Vehicles Detected (at once): {self.max_vehicles_at_once}")
        print(f"   • Total Unique Objects Tracked: {len(self.track_history)}")

        # Alert summary
        print(f"\n🚨 ALERT SUMMARY:")
        print(f"   • Total Alerts Generated: {self.total_alerts_generated}")
        if self.detection_log:
            print(f"   • Alert Types:")
            alert_types = {}
            for log in self.detection_log:
                alert_type = log['alert'].split(':')[0]
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

            for alert_type, count in alert_types.items():
                print(f"     - {alert_type}: {count} times")

        print(f"\n⏰ TIMELINE OF MAJOR EVENTS:")
        if self.detection_log:
            for i, log in enumerate(self.detection_log[-5:], 1):
                print(f"   {i}. [{log['timestamp']}] {log['alert']}")
        else:
            print("   • No alerts generated - All clear!")

        print(f"\n💾 LOGS SAVED:")
        print(f"   • Detection log saved with {len(self.detection_log)} entries")
        if hasattr(self, 'output_video_saved') and self.output_video_saved:
            print(f"   • Processed video saved as 'security_output.mp4'")

        print("="*60)
        print("🎉 Analysis Complete - Thank you for using Smart Security System!")
        print("="*60)

def main():
    """
    Main function to run the Smart Security System
    """
    print("🚀 Smart Security System with Computer Vision")
    print("=" * 50)

    # Initialize system
    security_system = SmartSecuritySystem(
        model_path="yolov8n.pt",
        confidence_threshold=0.6
    )

    # Run the system
    security_system.run_security_system(
        video_source="Test Videos/test_video_4.mp4",
        save_video=True
    )

if __name__ == "__main__":
    main()