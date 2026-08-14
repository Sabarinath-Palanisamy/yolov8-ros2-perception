import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import json
import time
from ultralytics import YOLO

class IntentionPerceptionNode(Node):
    def __init__(self):
        super().__init__('intention_perception_node')
        
        self.publisher_ = self.create_publisher(String, '/env/intention_updates', 10)
        self.get_logger().info("🚀 Initializing VDI Brain: Scale-Aware Proximity Tracking Stack...")
        
        model_path = "/home/Sabari/projects/ObjectDetection/roadwork_perception_module/runs/segment/train/weights/best.pt"
        video_path = "/home/Sabari/projects/ObjectDetection/roadwork_perception_module/videos/Simulation_test.mp4"
        
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        
        self.window_name = "VDI Stack - Perception Engine"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 450) 
        
        self.agent_history = {}
        
        # Danger zones optimized for an 800px wide layout footprint
        self.DANGER_ZONE_MIN = 250
        self.DANGER_ZONE_MAX = 550
        self.VRU_BUFFER = 80 
        
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return

        frame = cv2.resize(frame, (800, 450))
        current_time = time.time()
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        detected_agents = []
        current_ids = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            class_ids = results[0].boxes.cls.cpu().numpy().tolist()
            track_ids = results[0].boxes.id.cpu().numpy().tolist()

            for box, cls_id, trk_id in zip(boxes, class_ids, track_ids):
                trk_id = int(trk_id)
                current_ids.add(trk_id)
                
                cls_name = self.model.names[int(cls_id)].lower()
                
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                box_width = box[2] - box[0]
                
                raw_vx, vy = 0.0, 0.0 
                vx = 0.0
                
                if trk_id in self.agent_history:
                    prev_cx, prev_cy, prev_time, prev_vx = self.agent_history[trk_id]
                    dt = current_time - prev_time
                    if dt > 0:
                        raw_vx = (cx - prev_cx) / dt
                        vy = (cy - prev_cy) / dt
                        # Low-pass filter to smooth out pixel tracking jitter
                        vx = 0.7 * prev_vx + 0.3 * raw_vx
                
                self.agent_history[trk_id] = (cx, cy, current_time, vx)

                intention = "SAFE_STATIONARY"
                ttc = 99.9 

                # Base Kinematic Intention Calculations
                if abs(vx) > 15.0: 
                    if cx < self.DANGER_ZONE_MIN and vx > 0: 
                        distance = self.DANGER_ZONE_MIN - cx
                        ttc = distance / vx
                    elif cx > self.DANGER_ZONE_MAX and vx < 0: 
                        distance = cx - self.DANGER_ZONE_MAX
                        ttc = distance / abs(vx)
                    
                    if ttc < 2.0:
                        intention = "DANGEROUS_CROSSING"
                    else:
                        intention = "SAFE_MOVING"
                else:
                    if self.DANGER_ZONE_MIN < cx < self.DANGER_ZONE_MAX:
                        intention = "OBSTACLE_IN_PATH"

                # 🛠️ TWEAK 1: LONG-DISTANCE VEHICLE FALSE-POSITIVE FILTER
                # Identify standard vehicle categories
                is_vehicle = any(v in cls_name for v in ["car", "truck", "bus", "vehicle", "van"])
                # If a vehicle is far down the road (width < 60px on our 800px frame), suppress threat levels
                if is_vehicle and box_width < 60:
                    if intention == "DANGEROUS_CROSSING":
                        intention = "SAFE_MOVING"          # Downgrade to cautious yellow
                    elif intention == "OBSTACLE_IN_PATH":
                        intention = "SAFE_STATIONARY"      # Downgrade to safe green

                # 🛠️ TWEAK 2: VULNERABLE ROAD USER PROXIMITY ELEVATION
                # Ensure humans/workers/cyclists near our driving corridor stay strictly protected
                is_vru = any(h in cls_name for h in ["person", "worker", "bicycle", "motorcycle", "rider"])
                vru_in_buffer = (self.DANGER_ZONE_MIN - self.VRU_BUFFER) < cx < (self.DANGER_ZONE_MAX + self.VRU_BUFFER)
                
                if is_vru and vru_in_buffer:
                    if intention != "DANGEROUS_CROSSING":
                        intention = "CRITICAL_VRU_ZONE"

                # Build data payload
                agent = {
                    "id": trk_id,
                    "class": cls_name,
                    "bbox": [round(val, 2) for val in box],
                    "velocity": [round(vx, 2), round(vy, 2)],
                    "ttc": round(ttc, 2),
                    "intention": intention
                }
                detected_agents.append(agent)
                
                # Dynamic Color Assignment
                if intention in ["DANGEROUS_CROSSING", "CRITICAL_VRU_ZONE"]:
                    id_color = (0, 0, 255)       # Red: Immediate hazards/workers close by
                elif intention in ["SAFE_MOVING", "OBSTACLE_IN_PATH"]:
                    id_color = (0, 255, 255)     # Yellow: Moderate caution required
                else:
                    id_color = (0, 255, 0)       # Green: Completely safe/distant

                # Draw clean label text
                cv2.putText(frame, f"{cls_name.upper()} ID: {trk_id}", (int(box[0]), int(box[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color, 2)

        self.agent_history = {k: v for k, v in self.agent_history.items() if k in current_ids}

        if detected_agents:
            msg = String()
            msg.data = json.dumps(detected_agents)
            self.publisher_.publish(msg)

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = IntentionPerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'cap') and node.cap.isOpened():
            node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()