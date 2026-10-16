#!/usr/bin/env python3
"""
Generate ArUco Marker for Eye-in-Hand Calibration.
"""
import os
import cv2
import numpy as np

def generate_marker(marker_id=0, size_pixels=600, output_path=None):
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, f"marker_{marker_id}.png")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    if hasattr(cv2.aruco, 'generateImageMarker'):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)
    else:
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, size_pixels)
    
    # Add white margin
    margin = 50
    bordered_img = np.full((size_pixels + 2 * margin, size_pixels + 2 * margin), 255, dtype=np.uint8)
    bordered_img[margin:margin + size_pixels, margin:margin + size_pixels] = marker_img
    
    cv2.imwrite(output_path, bordered_img)
    print(f"Generated ArUco marker ID={marker_id} saved to {output_path}")

if __name__ == '__main__':
    generate_marker()
