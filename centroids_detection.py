import numpy as np
import cv2
import paho.mqtt.client as mqtt
from sklearn.cluster import DBSCAN


# Global variables
right_foot_centroids = []
left_foot_centroids = []
last_left_foot = None
last_right_foot = None
right_foot_detected = False
left_foot_detected = False
DISTANCE_THRESHOLD = 35  # Define a threshold for foot re-detection
PROXIMITY_THRESHOLD = 15  # Define a threshold for proximity check to avoid swapping

def preprocess_frame(frame):
    if np.max(frame) > 0:
        frame_normalized = (frame / np.max(frame) * 255).astype(np.uint8)
    else:
        frame_normalized = np.zeros_like(frame, dtype=np.uint8)
    
    blurred = cv2.GaussianBlur(frame_normalized, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 1, 255, cv2.THRESH_BINARY)
    return thresh

def cluster_foot_regions(frame):
    nonzero_points = np.argwhere(frame > 0)
    if len(nonzero_points) == 0:
        return None, None
    
    clustering = DBSCAN(eps=1, min_samples=5).fit(nonzero_points)
    labels = clustering.labels_
    
    unique_labels = np.unique(labels)
    
    if len(unique_labels) == 1 and unique_labels[0] == -1:
        return None, None
    
    clusters = [nonzero_points[labels == label] for label in unique_labels if label != -1]
    
    if len(clusters) == 1:
        return clusters[0], None
    elif len(clusters) >= 2:
        return clusters[0], clusters[1]
    else:
        return None, None

def update_foot_centroids(frame):
    global last_left_foot, last_right_foot, right_foot_detected, left_foot_detected
    processed_frame = preprocess_frame(frame)
    cluster_1, cluster_2 = cluster_foot_regions(processed_frame)

    if cluster_1 is not None:
        centroid_1 = np.mean(cluster_1, axis=0)
    else:
        centroid_1 = None

    if cluster_2 is not None:
        centroid_2 = np.mean(cluster_2, axis=0)
    else:
        centroid_2 = None

    if not right_foot_detected and centroid_1 is not None and centroid_2 is None:
        # Right foot detected first
        right_foot_centroids.append(centroid_1)
        left_foot_centroids.append([np.nan, np.nan])
        last_right_foot = centroid_1
        right_foot_detected = True
        return centroid_1, [np.nan, np.nan]  # Right foot detected, no left foot yet

    elif right_foot_detected and not left_foot_detected:
        # Now left foot is detected after right foot
        if centroid_1 is not None and centroid_2 is not None:
            dist_to_right_1 = np.linalg.norm(centroid_1 - last_right_foot) if last_right_foot is not None else np.inf
            dist_to_right_2 = np.linalg.norm(centroid_2 - last_right_foot) if last_right_foot is not None else np.inf

            # The closer centroid to the last right foot position is the right foot
            if dist_to_right_1 < dist_to_right_2:
                right_foot_centroids.append(centroid_1)
                left_foot_centroids.append(centroid_2)
                last_right_foot, last_left_foot = centroid_1, centroid_2
            else:
                right_foot_centroids.append(centroid_2)
                left_foot_centroids.append(centroid_1)
                last_right_foot, last_left_foot = centroid_2, centroid_1
            
            left_foot_detected = True
            return last_right_foot, last_left_foot  # Return both centroids

        elif centroid_1 is not None:
            dist_to_right = np.linalg.norm(centroid_1 - last_right_foot) if last_right_foot is not None else np.inf
            if dist_to_right < np.inf:
                right_foot_centroids.append(centroid_1)
                left_foot_centroids.append([np.nan, np.nan])
                last_right_foot = centroid_1
                return centroid_1, [np.nan, np.nan]  # Right foot detected, no left foot yet

    elif right_foot_detected and left_foot_detected:
        # Both feet already detected, now classify based on last known positions and distance threshold
        if centroid_1 is not None and centroid_2 is not None:
            dist_to_left_1 = np.linalg.norm(centroid_1 - last_left_foot) if last_left_foot is not None else np.inf
            dist_to_right_1 = np.linalg.norm(centroid_1 - last_right_foot) if last_right_foot is not None else np.inf

            dist_to_left_2 = np.linalg.norm(centroid_2 - last_left_foot) if last_left_foot is not None else np.inf
            dist_to_right_2 = np.linalg.norm(centroid_2 - last_right_foot) if last_right_foot is not None else np.inf

            # Check distance threshold to avoid misclassification when a foot leaves and re-enters
            if np.linalg.norm(centroid_1 - centroid_2) < PROXIMITY_THRESHOLD:
                # Feet are too close, assume no swap, keep current classification
                return last_right_foot, last_left_foot

            if dist_to_right_1 < dist_to_left_1 and dist_to_right_1 < DISTANCE_THRESHOLD:
                right_foot_centroids.append(centroid_1)
                left_foot_centroids.append(centroid_2)
                last_right_foot, last_left_foot = centroid_1, centroid_2
            elif dist_to_right_2 < dist_to_left_2 and dist_to_right_2 < DISTANCE_THRESHOLD:
                right_foot_centroids.append(centroid_2)
                left_foot_centroids.append(centroid_1)
                last_right_foot, last_left_foot = centroid_2, centroid_1
            else:
                # When both feet exceed the threshold, use previous positions to classify
                if dist_to_left_1 < dist_to_right_1:
                    left_foot_centroids.append(centroid_1)
                    right_foot_centroids.append(centroid_2)
                    last_left_foot, last_right_foot = centroid_1, centroid_2
                else:
                    left_foot_centroids.append(centroid_2)
                    right_foot_centroids.append(centroid_1)
                    last_left_foot, last_right_foot = centroid_2, centroid_1

            return last_right_foot, last_left_foot  # Return both centroids

        elif centroid_1 is not None:
            dist_to_left = np.linalg.norm(centroid_1 - last_left_foot) if last_left_foot is not None else np.inf
            dist_to_right = np.linalg.norm(centroid_1 - last_right_foot) if last_right_foot is not None else np.inf

            if dist_to_right < dist_to_left and dist_to_right < DISTANCE_THRESHOLD:
                right_foot_centroids.append(centroid_1)
                left_foot_centroids.append([np.nan, np.nan])
                last_right_foot = centroid_1
                return centroid_1, [np.nan, np.nan]  # Right foot detected, no left foot
            elif dist_to_left < DISTANCE_THRESHOLD:
                left_foot_centroids.append(centroid_1)
                right_foot_centroids.append([np.nan, np.nan])
                last_left_foot = centroid_1
                return [np.nan, np.nan], centroid_1  # Left foot detected, no right foot

    # If no feet are detected in any branch, return (nan, nan)
    return [np.nan, np.nan], [np.nan, np.nan]