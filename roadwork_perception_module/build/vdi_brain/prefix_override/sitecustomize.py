import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/Sabari/projects/ObjectDetection/roadwork_perception_module/install/vdi_brain'
