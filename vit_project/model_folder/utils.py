"""
Utility functions for room detection and visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import time

# Define beautiful colors and icons for room types
ROOM_CONFIG = {
    "Living Room": {
        "color": "#8DD3C7",  # Teal
        "icon": "🛋️",
        "features": ["Sofa", "TV", "Coffee Table"],
        "description": "Main living space for family activities"
    },
    "Bedroom": {
        "color": "#FFFFB3",  # Light yellow
        "icon": "🛏️",
        "features": ["Bed", "Wardrobe", "Nightstand"],
        "description": "Private sleeping area"
    },
    "Bathroom": {
        "color": "#BEBADA",  # Lavender
        "icon": "🚿",
        "features": ["Shower", "Toilet", "Sink"],
        "description": "Personal hygiene space"
    },
    "Kitchen": {
        "color": "#FB8072",  # Salmon pink
        "icon": "🍳",
        "features": ["Stove", "Fridge", "Sink"],
        "description": "Cooking and food preparation area"
    },
    "Room": {
        "color": "#80B1D3",  # Light blue
        "icon": "🚪",
        "features": ["Multi-purpose", "Generic space"],
        "description": "General purpose room"
    }
}

def detect_rooms_and_walls(image_path, output_dir=None, min_area=500, max_area=50000):
    """
    Detect walls and rooms in a floorplan image
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save visualization (optional)
        min_area: Minimum room area to consider (increased to filter out text)
        max_area: Maximum room area to consider
        
    Returns:
        dict: Detection results including room data and visualization path
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Resize large images
    max_dimension = 1200
    height, width = img.shape[:2]
    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Process the image for wall detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply OCR preprocessing - this can help identify text regions to ignore
    # Use morphological operations to emphasize actual walls and rooms
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)  # Close gaps in walls
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)   # Remove small noise
    
    # Detect walls using Hough transform with stricter parameters
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 
        threshold=50,  # Higher threshold to filter out noise
        minLineLength=40,  # Longer lines for walls
        maxLineGap=10
    )
    
    # Detect rooms using contours - filter by size to exclude text
    binary = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size - increase min_area to filter out text
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Additional check for text-like shapes (very elongated)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if aspect_ratio < 10:  # Not an extremely elongated shape like text
                filtered_contours.append(contour)
    
    # Count walls and rooms
    wall_count = len(lines) if lines is not None else 0
    room_count = len(filtered_contours)
    
    # Initialize room data structures
    room_data = []
    room_type_counts = {room_type: 0 for room_type in ROOM_CONFIG.keys()}
    
    # Sort contours by area (largest first)
    filtered_contours.sort(key=cv2.contourArea, reverse=True)
    
    # Create visualization
    plt.figure(figsize=(12, 10), dpi=120)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    
    # Draw walls
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            plt.plot([x1, x2], [y1, y2], color='red', linewidth=1.5, alpha=0.7)
    
    # Process each room contour
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Enhanced room classification with more precise criteria
        if area < 1000 and 0.7 < aspect_ratio < 1.3:
            room_type = "Bathroom"
        elif area > 5000 and aspect_ratio > 1.3:
            room_type = "Living Room"
        elif 2000 < area < 5000 and 0.7 < aspect_ratio < 1.5:
            room_type = "Bedroom"
        elif 1000 < area < 2000 and aspect_ratio > 1.5:
            room_type = "Kitchen"
        else:
            room_type = "Room"
        
        # Get room configuration
        config = ROOM_CONFIG[room_type]
        color = config["color"]
        icon = config["icon"]
        
        # Update room count
        room_type_counts[room_type] += 1
        room_id = room_type_counts[room_type]
        
        # Draw room with enhanced styling
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor=color + "60"  # 60% transparency
        )
        plt.gca().add_patch(rect)
        
        # Add room label with icon
        label = f"{icon} {room_type} {room_id}"
        plt.text(x+5, y+20, label, color='black', fontsize=10,
                 fontweight='bold', bbox=dict(facecolor='white', alpha=0.85))
        
        # Store detailed room info
        room_data.append({
            "id": i+1,
            "type": room_type,
            "type_id": room_id,
            "label": label,
            "icon": icon,
            "area": float(area),
            "bounds": [int(x), int(y), int(w), int(h)],
            "aspect_ratio": float(aspect_ratio),
            "features": config["features"],
            "description": config["description"],
            "color": color
        })
    
    # Add title with stats
    plt.title(f"Wall and Room Detection\nWalls: {wall_count} | Rooms: {room_count}")
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization if output directory provided
    vis_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        filename = f'detection_{timestamp}.png'
        vis_path = os.path.join(output_dir, filename)
        plt.savefig(vis_path, format='png', dpi=120, bbox_inches='tight')
    
    # Organize rooms by type
    rooms_by_type = {room_type: [] for room_type in ROOM_CONFIG.keys()}
    for room in room_data:
        room_type = room["type"]
        if room_type in rooms_by_type:
            rooms_by_type[room_type].append(room)
    
    # Return detection results
    return {
        'wall_count': wall_count,
        'room_count': room_count,
        'visualization': vis_path,
        'rooms': room_data,
        'rooms_by_type': rooms_by_type,
        'room_type_counts': room_type_counts,
        'room_config': ROOM_CONFIG
    }
