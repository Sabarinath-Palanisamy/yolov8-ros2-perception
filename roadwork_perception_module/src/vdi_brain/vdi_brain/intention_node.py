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
        self.get_logger().info("🚀 Initializing VDI Brain: Intention State Machine...")
        
        self.model = YOLO("/home/Sabari/projects/ObjectDetection/roadwork_perception_module/runs/segment/train/weights/best.pt")
        self.cap = cv2.VideoCapture(0)
        
        # Memory dictionary
        self.agent_history = {}
        
        # Vehicle Path Configuration (Assuming 640px camera width)
        self.DANGER_ZONE_MIN = 200
        self.DANGER_ZONE_MAX = 440
        
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        current_time = time.time()
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        detected_agents = []
        current_ids = set()
        
        # Draw the virtual path of the vehicle on the camera frame
        cv2.line(frame, (self.DANGER_ZONE_MIN, 0), (self.DANGER_ZONE_MIN, 480), (0, 0, 255), 2)
        cv2.line(frame, (self.DANGER_ZONE_MAX, 0), (self.DANGER_ZONE_MAX, 480), (0, 0, 255), 2)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            class_ids = results[0].boxes.cls.cpu().numpy().tolist()
            track_ids = results[0].boxes.id.cpu().numpy().tolist()

            for box, cls_id, trk_id in zip(boxes, class_ids, track_ids):
                trk_id = int(trk_id)
                current_ids.add(trk_id)
                
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                vx, vy = 0.0, 0.0 
                
                if trk_id in self.agent_history:
                    prev_cx, prev_cy, prev_time = self.agent_history[trk_id]
                    dt = current_time - prev_time
                    if dt > 0:
                        vx = (cx - prev_cx) / dt
                        vy = (cy - prev_cy) / dt
                
                self.agent_history[trk_id] = (cx, cy, current_time)

                # --- NEW: Intention State Machine ---
                intention = "SAFE_STATIONARY"
                ttc = 99.9 # Default safe value
                
                # Ignore micro-jitters from the ByteTrack bounding box
                if abs(vx) > 15.0: 
                    if cx < self.DANGER_ZONE_MIN and vx > 0: # Left side, moving right
                        distance = self.DANGER_ZONE_MIN - cx
                        ttc = distance / vx
                    elif cx > self.DANGER_ZONE_MAX and vx < 0: # Right side, moving left
                        distance = cx - self.DANGER_ZONE_MAX
                        ttc = distance / abs(vx)
                    
                    if ttc < 2.0:
                        intention = "DANGEROUS_CROSSING"
                    else:
                        intention = "SAFE_MOVING"
                else:
                    if self.DANGER_ZONE_MIN < cx < self.DANGER_ZONE_MAX:
                        intention = "OBSTACLE_IN_PATH"

                # Build the final payload
                agent = {
                    "id": trk_id,
                    "class": self.model.names[int(cls_id)],
                    "bbox": [round(val, 2) for val in box],
                    "velocity": [round(vx, 2), round(vy, 2)],
                    "ttc": round(ttc, 2),
                    "intention": intention
                }
                detected_agents.append(agent)
                
                # Overlay the intention on the live video
                cv2.putText(frame, intention, (int(box[0]), int(box[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        self.agent_history = {k: v for k, v in self.agent_history.items() if k in current_ids}

        if detected_agents:
            msg = String()
            msg.data = json.dumps(detected_agents)
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published: {msg.data}')

        # Note: We plot the manual frame here to keep our custom red lines and text
        cv2.imshow("VDI Brain - Intention Tracker", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = IntentionPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()