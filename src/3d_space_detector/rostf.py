import cv2
import math
import rospy
import ros_numpy
import statistics
import collections

import numpy as np
import tensorflow as tf
import sensor_msgs.point_cloud2 as pc2

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class rostf():
    def __init__(self):
        self.buffer_size = 10
        self.bridge = CvBridge()
        self.rgb_frame_buffer = []
        self.depth_frame_buffer = []

        rospy.init_node('zed_mass_node', anonymous=False)
        raw_image_topic = "/zed/zed_node/rgb/image_rect_color"
        depht_image_topic = "/zed/zed_node/point_cloud/cloud_registered"

        rospy.Subscriber(raw_image_topic, Image, self.rgb_image_callback)
        rospy.Subscriber(depht_image_topic, PointCloud2, self.depht_image_callback)
        self.recognize()
        rospy.spin()

    def rgb_image_callback(self, image):
        rgb_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        # resize na imagem
        # rgb_image = cv2.resize(rgb_image, (416,704))
        # rgb_image = rgb_image.astype(np.uint8)

        rgb_image = rgb_image[:, :, 0:3]
        self.rgb_frame_buffer.append(rgb_image)

        if len(self.rgb_frame_buffer) > self.buffer_size:
            self.rgb_frame_buffer.pop(0)

    def depht_image_callback(self, pointcloud):        
        test_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud)

        # usar a mesma resolucao da rgb image
        # np.reshape(test_array, (416, 704))
        
        # depht_image = depht_image.astype(np.uint8)        
        # depht_image = depht_image.astype(np.uint8)
        # depht_image = depht_image[:, :, 0:3]
        
        self.depth_frame_buffer.append(test_array)

        if len(self.depth_frame_buffer) > self.buffer_size:
            self.depth_frame_buffer.pop(0)

    def recognize(self):
        PATH_TO_FROZEN_GRAPH = 'model/frozen_inference_graph.pb'
        PATH_TO_LABELS = 'model/labelmap.pbtxt'
        NUM_CLASSES = 1

        width = 1280 #704
        height = 720 #416

        # Load a (frozen) Tensorflow model into memory.
        print("Loading model")
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        # Loading label map
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Perform the inference
        exit_signal = False

        with detection_graph.as_default() and tf.Session(config=config, graph=detection_graph) as sess:
            print("INITIALIZING NEW SESSION")
            while not exit_signal:
                if (len(self.rgb_frame_buffer) > 0 and len(self.depth_frame_buffer) > 0):
                    # aquisicao de imagens
                    frame = self.rgb_frame_buffer[0]
                    self.rgb_frame_buffer.pop(0)

                    # aquisicao de imagens de profundidade 
                    depth_frame = self.depth_frame_buffer[0] 
                    self.depth_frame_buffer.pop(0)
                   
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(frame, axis=0)

                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})

                    # Visualization of the results of a detu,vction and storing targets positions
                    box_to_display_str_map = collections.defaultdict(list)
                    box_to_color_map = collections.defaultdict(str)

                    research_distance_box = 30
                    targets_pos = []
                    image_np = frame
                    num_detections_ = num_detections.astype(int)[0]
                    boxes_ = np.squeeze(boxes)
                    classes_ = np.squeeze(classes).astype(np.int32)
                    scores_ = np.squeeze(scores)
                    
                    for i in range(num_detections_):
                        confidence = 0.8
                        if scores_[i] > confidence:
                            box = tuple(boxes_[i].tolist())
                            if classes_[i] in category_index.keys():
                                class_name = category_index[classes_[i]]['name']

                            display_str = str(class_name)
                            if not display_str:
                                display_str = '{}%'.format(int(100 * scores_[i]))
                            else:
                                display_str = '{}: {}%'.format(display_str, int(100 * scores_[i]))

                            # Find object distance
                            ymin, xmin, ymax, xmax = box
                            x_center = int(xmin * width + (xmax - xmin) * width * 0.5)
                            y_center = int(ymin * height + (ymax - ymin) * height * 0.5)
                            x_vect = []
                            y_vect = []
                            z_vect = []

                            # the points used for calculating distances are at most 30 pixels from the center
                            min_y_r = max(int(ymin * height), int(y_center - research_distance_box))
                            min_x_r = max(int(xmin * width), int(x_center - research_distance_box))
                            max_y_r = min(int(ymax * height), int(y_center + research_distance_box))
                            max_x_r = min(int(xmax * width), int(x_center + research_distance_box))

                            if min_y_r < 0: min_y_r = 0
                            if min_x_r < 0: min_x_r = 0
                            if max_y_r > height: max_y_r = height
                            if max_x_r > width: max_x_r = width

                            for j_ in range(min_y_r, max_y_r):
                                for i_ in range(min_x_r, max_x_r):
                                    x = depth_frame[j_, i_][0]
                                    y = depth_frame[j_, i_][1]
                                    z = depth_frame[j_, i_][2]

                                    if not np.isnan(z) and not np.isinf(z):
                                        if not np.isnan(y) and not np.isinf(y):
                                            if not np.isnan(x) and not np.isinf(x):
                                                x_vect.append(x)
                                                y_vect.append(y)
                                                z_vect.append(z)

                            if len(x_vect) > 0:
                                aux = []

                                x = statistics.median(x_vect)
                                y = statistics.median(y_vect)
                                z = statistics.median(z_vect)

                                # Getting the position of the targets detected
                                aux.append(round(x, 2))
                                aux.append(round(y, 2))
                                aux.append(round(z, 2))

                                targets_pos.append(aux)
                                
                                # Calculating distances
                                distance = math.sqrt(x * x + y * y + z * z)

                                display_str = display_str + " " + str('% 6.2f' % distance) + " m "
                                box_to_display_str_map[box].append(display_str)
                                box_to_color_map[box] = vis_util.STANDARD_COLORS[classes_[i] % len(vis_util.STANDARD_COLORS)]

                    print(targets_pos)
                    for box, color in box_to_color_map.items():
                        ymin, xmin, ymax, xmax = box
                        vis_util.draw_bounding_box_on_image_array(frame, ymin, xmin, ymax, xmax, color=color, thickness=4, display_str_list=box_to_display_str_map[box], use_normalized_coordinates=True)

                    cv2.imshow('ZED object detection', frame )
                   
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        exit_signal = True
            
            sess.close()

def main():
    node = rostf()

if __name__ == '__main__':
    main()