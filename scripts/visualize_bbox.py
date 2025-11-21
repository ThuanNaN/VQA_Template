"""Visualize the bounding boxes of detected objects using Gradio interface."""
import os
import sys

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import gradio as gr
from utils.dataset_utils import load_obj_tsv
import time

print("Starting Object Detection Visualization Interface...")

# Get the absolute path to the project root directory (already defined above)

# Load class vocabularies
vocab_dir = os.path.join(PROJECT_ROOT, "data", "1600-400-20")
with open(os.path.join(vocab_dir, "objects_vocab.txt"), "r") as f:
    obj_classes = f.read().split("\n")[:-1]
obj_id2class = {i: obj_classes[i] for i in range(len(obj_classes))}

with open(os.path.join(vocab_dir, "attributes_vocab.txt"), "r") as f:
    attr_classes = f.read().split("\n")[:-1]
attr_id2class = {i: attr_classes[i] for i in range(len(attr_classes))}

# Define the mapping between TSV files and their corresponding image directories (absolute paths)
IMG_DIR = {
    # ViVQA dataset
    "vivqa_obj36.tsv": os.path.join(PROJECT_ROOT, "data", "vivqa", "images"),

    # OpenViVQA dataset
    "openvivqa_train_obj36.tsv": os.path.join(PROJECT_ROOT, "data", "openvivqa", "training-images"),
    "openvivqa_dev_obj36.tsv": os.path.join(PROJECT_ROOT, "data", "openvivqa", "dev-images"),
    "openvivqa_test_obj36.tsv": os.path.join(PROJECT_ROOT, "data", "openvivqa", "test-images"),

    # ViTextVQA dataset
    "vitextvqa_obj36.tsv": os.path.join(PROJECT_ROOT, "data", "vitextvqa", "images"),

    # ViOCRVQA dataset
    "viocrvqa_obj36.tsv": os.path.join(PROJECT_ROOT, "data", "viocrvqa", "images"),

    # EVJVQA dataset
    "evjvqa_obj36.tsv": os.path.join(PROJECT_ROOT, "data", "evjvqa", "images"),

    # ViVQA-X dataset
    # "vivqax_obj36.tsv": os.path.join(PROJECT_ROOT, "data", "vivqax", "images"),
}

# Cache for loaded TSV data
data_cache = {}

def find_tsv_file_paths():
    """Find the full paths of the TSV files specified in IMG_DIR"""
    tsv_file_paths = {}
    
    # Recursively search for the TSV files in data directory
    data_dir = os.path.join(PROJECT_ROOT, "data")
    for root, dirs, files in os.walk(data_dir):
        for tsv_filename in IMG_DIR.keys():
            if tsv_filename in files:
                tsv_file_paths[tsv_filename] = os.path.join(root, tsv_filename)
    
    return tsv_file_paths

def get_image_path(tsv_filepath, img_id, remove_prefix=True):
    """
    Get the image path based on the TSV file and image ID
    
    Args:
        tsv_filepath: Path to the TSV file
        img_id: Image ID
        
    Returns:
        Path to the image if found, None otherwise
    """
    # Get the TSV filename from the path
    tsv_filename = os.path.basename(tsv_filepath)
    
    # Find the corresponding image directory
    img_dir = IMG_DIR.get(tsv_filename)
    
    if img_dir is None:
        return None
    
    if remove_prefix:
        # Remove any prefix from img_id if present (e.g., "COCO_val2014_000000123456.jpg" -> "000000123456")
        img_id = os.path.splitext(os.path.basename(img_id))[0].split('_')[-1]
    
    # Try different extensions in the image directory
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = os.path.join(img_dir, f"{img_id}{ext}")
        if os.path.exists(img_path):
            return img_path
        else:
            print(f"Image not found: {img_path}")
    
    return None

def load_data_with_cache(tsv_filepath):
    """
    Load data from TSV file with caching
    
    Args:
        tsv_filepath: Path to the TSV file
        
    Returns:
        Loaded data
    """
    # Check if data is already in cache
    if tsv_filepath in data_cache:
        print(f"Using cached data for {os.path.basename(tsv_filepath)}")
        return data_cache[tsv_filepath]
    
    # Load data from file
    start_time = time.time()
    data = load_obj_tsv(tsv_filepath)
    elapsed = time.time() - start_time
    print(f"Loaded data from {os.path.basename(tsv_filepath)} in {elapsed:.2f} seconds")
    
    # Store in cache
    data_cache[tsv_filepath] = data
    
    return data

def visualize_detection(image, detection_data, confidence_threshold=0.0):
    """
    Visualize object detection results on an image
    
    Args:
        image: PIL Image or path to image
        detection_data: Dictionary containing detection data
        confidence_threshold: Minimum confidence score to display
        
    Returns:
        Matplotlib figure with visualization
    """
    # Load the image if path is provided
    if isinstance(image, str):
        try:
            img = Image.open(image)
        except Exception as e:
            print(f"Error loading image: {e}")
            img_width = detection_data.get('img_w', 600)
            img_height = detection_data.get('img_h', 400)
            img = Image.new('RGB', (img_width, img_height), color='white')
    elif image is None:
        # Create a blank image if no image is provided
        img_width = detection_data.get('img_w', 600)
        img_height = detection_data.get('img_h', 400)
        img = Image.new('RGB', (img_width, img_height), color='white')
    else:
        img = image
    
    img_width, img_height = img.size
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(np.array(img))
    
    # Get detection data
    boxes = detection_data.get('boxes', [])
    objects_id = detection_data.get('objects_id', [])
    objects_conf = detection_data.get('objects_conf', [])
    attrs_id = detection_data.get('attrs_id', [])
    attrs_conf = detection_data.get('attrs_conf', [])
    
    # Generate random colors for each class
    np.random.seed(42)  # For reproducibility
    colors = np.random.rand(1000, 3)
    
    # Draw bounding boxes
    for i, (box, obj_id, obj_conf, attr_id, attr_conf) in enumerate(zip(boxes, objects_id, objects_conf, attrs_id, attrs_conf)):
        # Skip if confidence is below threshold
        if obj_conf < confidence_threshold:
            continue
            
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within image boundaries
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        width = x2 - x1
        height = y2 - y1
        
        # Get color for this class
        color = colors[int(obj_id) % len(colors)]
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Create label with object class, attribute, and confidence
        obj_name = obj_id2class[int(obj_id)]
        attr_name = attr_id2class[int(attr_id)]
        label = f"{obj_name}: {attr_name} ({obj_conf:.2f})"

        plt.text(x1, y1-5, label, bbox=dict(facecolor=color, alpha=0.5))
    
    plt.title(f"Object Detection Results - {detection_data.get('img_id', 'Unknown')}")
    plt.axis('off')
    plt.tight_layout()
    
    return fig

def process_detection_data(tsv_filename, image, confidence_threshold=0.0, index=0, cached_data=None):
    """
    Process detection data and visualize results
    
    Args:
        tsv_filename: Basename of the TSV file
        image: Image to visualize detections on
        confidence_threshold: Minimum confidence score
        index: Index of detection to use from the file
        cached_data: Previously loaded data (if available)
        
    Returns:
        Matplotlib figure with visualization, status message, current index, max index, and cached data
    """
    try:
        # Get full path of the TSV file
        tsv_file_paths = find_tsv_file_paths()
        tsv_filepath = tsv_file_paths.get(tsv_filename)
        
        if not tsv_filepath:
            return None, f"Error: TSV file '{tsv_filename}' not found", 0, 0, None
        
        # Use cached data if available, otherwise load from file
        if cached_data is not None:
            data = cached_data
        else:
            data = load_data_with_cache(tsv_filepath)
        
        # Check if data is a list
        if isinstance(data, list):
            if len(data) == 0:
                return None, "Error: No detection data found in the file", 0, 0, data
                
            max_index = len(data) - 1
            if 0 <= index <= max_index:
                detection_data = data[index]
                total_entries = len(data)
            else:
                # If index out of range, use the first entry
                index = 0
                detection_data = data[0]
                total_entries = len(data)
                return None, f"Error: Index out of range. Using index 0.", 0, max_index, data
        else:
            # If it's a single dictionary
            detection_data = data
            total_entries = 1
            max_index = 0
            
        # Update the image if img_id is available and no image was provided
        img_id = detection_data.get('img_id')
        if img_id and not image:
            img_path = get_image_path(tsv_filepath, img_id)
            if img_path:
                image = img_path
            
        # Visualize detection
        fig = visualize_detection(image, detection_data, confidence_threshold)
        
        # Add information about the image location to the status message
        status_msg = f"Showing detection {index+1} of {total_entries} entries from {tsv_filename}."
        if image and isinstance(image, str):
            status_msg += f"\nImage loaded from: {image}"
        elif not image:
            status_msg += "\nNo image found. Using blank canvas."
            
        return fig, status_msg, index, max_index, data
        
    except Exception as e:
        return None, f"Error processing data: {str(e)}", 0, 0, None

def next_detection(tsv_filename, image, confidence_threshold, current_index, max_index, cached_data):
    """Handle next button click"""
    if current_index < max_index:
        next_index = current_index + 1
    else:
        next_index = 0  # Wrap around to the beginning
    
    return process_detection_data(tsv_filename, image, confidence_threshold, next_index, cached_data)

def prev_detection(tsv_filename, image, confidence_threshold, current_index, max_index, cached_data):
    """Handle previous button click"""
    if current_index > 0:
        prev_index = current_index - 1
    else:
        prev_index = max_index  # Wrap around to the end
    
    return process_detection_data(tsv_filename, image, confidence_threshold, prev_index, cached_data)

def get_dataset_info(tsv_filename):
    """Get information about the dataset size"""
    try:
        # Get full path of the TSV file
        tsv_file_paths = find_tsv_file_paths()
        tsv_filepath = tsv_file_paths.get(tsv_filename)
        
        if not tsv_filepath:
            return "No dataset selected", None
            
        data = load_data_with_cache(tsv_filepath)
        if isinstance(data, list):
            return f"Dataset contains {len(data)} images", data
        else:
            return "Dataset contains 1 image", data
    except Exception as e:
        return f"Error getting dataset info: {e}", None
    
# Gradio interface
def create_interface():
    # Find TSV files that match the keys in IMG_DIR
    tsv_file_paths = find_tsv_file_paths()
    available_tsv_files = list(tsv_file_paths.keys())
    
    with gr.Blocks(title="Object Detection Visualization") as app:
        gr.Markdown("# Object Detection Visualization")
        gr.Markdown("Select a detection data file to visualize object detection results.")
        
        # State variables to track current index, max index, and cached data
        current_index = gr.State(0)
        max_index = gr.State(0)
        cached_data = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                tsv_dropdown = gr.Dropdown(
                    choices=available_tsv_files, 
                    label="Select Detection Data File",
                    info="Only pre-defined TSV files are available"
                )
                dataset_info = gr.Textbox(label="Dataset Information", lines=1)
                
                # New slider for directly selecting image index
                index_slider = gr.Slider(
                    minimum=0, 
                    maximum=100,  # Will be updated based on dataset size
                    value=0, 
                    step=1, 
                    label="Image Index",
                    info="Directly select image by index"
                )
                
                image_input = gr.Image(type="pil", label="Upload Custom Image (Optional)")
                conf_slider = gr.Slider(
                    minimum=0.0, 
                    maximum=1.0, 
                    value=0.0, 
                    step=0.05, 
                    label="Confidence Threshold"
                )
                
                with gr.Row():
                    prev_btn = gr.Button("← Previous", variant="secondary")
                    jump_btn = gr.Button("Go to Index", variant="primary")
                    next_btn = gr.Button("Next →", variant="secondary")
                
                visualize_btn = gr.Button("Visualize Detection", variant="primary")
                clear_cache_btn = gr.Button("Clear Cache", variant="secondary")
                
            with gr.Column(scale=2):
                # Output components
                output_plot = gr.Plot(label="Detection Visualization")
                status_output = gr.Textbox(label="Status Messages", lines=3)
                navigation_info = gr.Textbox(label="Navigation", lines=1)
        
        # Update dataset info, preload data, and update slider max when TSV file changes
        def update_dataset_and_controls(tsv_filename):
            info, data = get_dataset_info(tsv_filename)
            if data and isinstance(data, list):
                max_index = len(data) - 1
                # Return updated slider maximum and value
                return info, data, 0, gr.update(maximum=max_index, value=0)
            else:
                max_index = 0
                return info, data, 0, gr.update(maximum=0, value=0)
            
        tsv_dropdown.change(
            fn=update_dataset_and_controls,
            inputs=[tsv_dropdown],
            outputs=[dataset_info, cached_data, current_index, index_slider]
        )
        
        # Update current index when slider changes
        index_slider.change(
            fn=lambda idx: idx,
            inputs=[index_slider],
            outputs=[current_index]
        )
        
        # Set up event handler for visualization
        visualize_btn.click(
            fn=process_detection_data,
            inputs=[tsv_dropdown, image_input, conf_slider, current_index, cached_data],
            outputs=[output_plot, status_output, current_index, max_index, cached_data]
        )
        
        # Set up event handler for jump to index button
        jump_btn.click(
            fn=process_detection_data,
            inputs=[tsv_dropdown, image_input, conf_slider, index_slider, cached_data],
            outputs=[output_plot, status_output, current_index, max_index, cached_data]
        )
        
        # Set up event handlers for navigation buttons
        next_btn.click(
            fn=next_detection,
            inputs=[tsv_dropdown, image_input, conf_slider, current_index, max_index, cached_data],
            outputs=[output_plot, status_output, current_index, max_index, cached_data]
        )
        
        prev_btn.click(
            fn=prev_detection,
            inputs=[tsv_dropdown, image_input, conf_slider, current_index, max_index, cached_data],
            outputs=[output_plot, status_output, current_index, max_index, cached_data]
        )
        
        # Update navigation info and slider when current index changes
        def update_navigation_info(idx, max_idx):
            return f"Image {idx+1} of {max_idx+1}", gr.update(value=idx)
            
        current_index.change(
            fn=update_navigation_info,
            inputs=[current_index, max_index],
            outputs=[navigation_info, index_slider]
        )
        
        # Clear cache button
        def clear_cache():
            global data_cache
            cache_size = len(data_cache)
            data_cache = {}
            return f"Cache cleared ({cache_size} datasets removed)"
            
        clear_cache_btn.click(
            fn=clear_cache,
            inputs=[],
            outputs=[status_output]
        )
        
        # Add search input for object classes
        with gr.Accordion("Advanced Features", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    search_input = gr.Textbox(
                        label="Search for Object Class",
                        placeholder="Enter object class (e.g., person, car)",
                        info="Search will show images containing this object class"
                    )
                    
                    search_btn = gr.Button("Search", variant="primary")
                    
                with gr.Column(scale=1):
                    search_results = gr.Textbox(
                        label="Search Results",
                        lines=3,
                        placeholder="Results will appear here"
                    )
                    
                    jump_to_result = gr.Number(
                        label="Jump to Result Index",
                        value=0,
                        precision=0,
                        info="Enter index from search results to jump directly to that image"
                    )
                    
                    jump_to_result_btn = gr.Button("Jump to Result", variant="primary")
        
        # Add a section showing the TSV-to-image directory mapping
        mapping_info = "## Image Directory Mapping\n"
        for tsv, img_dir in IMG_DIR.items():
            mapping_info += f"- **{tsv}**: {img_dir}\n"
        
        with gr.Accordion("Image Directory Mapping", open=True):
            gr.Markdown(mapping_info)
        
        gr.Markdown("""
        ## Instructions
        1. Select a detection data file from the dropdown
        2. The system will automatically load and cache the dataset
        3. Use the slider to jump directly to a specific image index
        4. Use the "Go to Index" button to load the image at the current slider position
        5. The system will automatically load images from the corresponding directory
        6. You can upload a custom image to override automatic image loading
        7. Adjust confidence threshold to filter low-confidence detections
        8. Use the Previous and Next buttons to navigate between detections
        9. Click "Clear Cache" to free up memory if needed
        10. Use the Advanced Features section to search for specific object classes
        """)
        
        # Function to search for object class
        def search_object_class(search_term, tsv_filename, cached_data):
            if not search_term or not tsv_filename or cached_data is None:
                return "Please select a dataset and enter a search term"
            
            search_term = search_term.lower()
            found_indices = []
            
            if isinstance(cached_data, list):
                for idx, item in enumerate(cached_data):
                    objects_id = item.get('objects_id', [])
                    
                    for obj_id in objects_id:
                        obj_name = obj_id2class.get(int(obj_id), "").lower()
                        if search_term in obj_name:
                            found_indices.append(idx)
                            break
                
                if found_indices:
                    return f"Found {len(found_indices)} images with '{search_term}'. Indices: {', '.join(map(str, found_indices[:20]))}{'...' if len(found_indices) > 20 else ''}"
                else:
                    return f"No images found containing '{search_term}'"
            else:
                return "Dataset is not a list or is empty"
        
        search_btn.click(
            fn=search_object_class,
            inputs=[search_input, tsv_dropdown, cached_data],
            outputs=[search_results]
        )
        
        # Jump to result button handler
        jump_to_result_btn.click(
            fn=process_detection_data,
            inputs=[tsv_dropdown, image_input, conf_slider, jump_to_result, cached_data],
            outputs=[output_plot, status_output, current_index, max_index, cached_data]
        )
        
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visuzlize the bounding boxes of detected objects")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    args = parser.parse_args()
    
    app = create_interface()
    app.launch(server_name=args.host, server_port=args.port)