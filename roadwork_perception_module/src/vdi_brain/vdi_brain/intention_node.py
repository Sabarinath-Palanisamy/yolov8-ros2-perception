import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import json
from ultralytics import YOLO

class IntentionPerceptionNode(Node):
    def __init__(self):
        super().__init__('intention_perception_node')
        
        # 1. Create the ROS2 Publisher
        self.publisher_ = self.create_publisher(String, '/env/intention_updates', 10)
        self.get_logger().info("🚀 Initializing VDI Brain Perception Node...")
        
        # 2. Load the compiled YOLOv8 Model
        # Using the absolute path to ensure ROS2 finds it from any directory
        self.model = YOLO("/home/Sabari/projects/ObjectDetection/roadwork_perception_module/runs/segment/train/weights/best.pt")
        
        # 3. Initialize Camera (0 = default webcam)
        self.cap = cv2.VideoCapture(0)
        
        # 4. Create a timer to process frames at 10 Hz (0.1 seconds)
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Camera frame dropped!")
            return

        # Execute ByteTrack ID Tracking
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        detected_agents = []
        
        # Extract Box Coordinates, Class IDs, and Track IDs
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            class_ids = results[0].boxes.cls.cpu().numpy().tolist()
            track_ids = results[0].boxes.id.cpu().numpy().tolist()

            # Package the data into a structured dictionary
            for box, cls_id, trk_id in zip(boxes, class_ids, track_ids):
                agent = {
                    "id": int(trk_id),
                    "class": self.model.names[int(cls_id)],
                    "bbox": [round(val, 2) for val in box] # [x1, y1, x2, y2]
                }
                detected_agents.append(agent)

        # Publish the data payload to the ROS2 Network
        if detected_agents:
            msg = String()
            msg.data = json.dumps(detected_agents)
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published Payload: {msg.data}')

        # Show the live visual feed (optional, good for debugging)
        annotated_frame = results[0].plot()
        cv2.imshow("VDI Brain - Intention Tracker", annotated_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = IntentionPerceptionNode()
    
    try:
        rclpy.spin(node) # Keep the node running infinitely
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
