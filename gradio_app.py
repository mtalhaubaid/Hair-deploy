# gradio_app.py
import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import os
import io
import base64
import gdown
from scipy import ndimage

from networks import get_network  # your own import

# ========== Global Settings ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the model
model_path = r"models\pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth"

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Google Drive file ID
file_id = "1_IFr5ke26a1MATn2XzxF6ysF6qJJIH8t"
gdown_url = f"https://drive.google.com/uc?id={file_id}"

# Download model if it doesn't exist
if not os.path.exists(model_path):
    print("Model not found. Downloading from Google Drive...")
    gdown.download(gdown_url, model_path, quiet=False)
else:
    print("Model already exists.")

network_name = "pspnet_resnet101"
os.makedirs("results", exist_ok=True)

# ========== Hair Colors ==========
HAIR_COLORS = {
    "Jet Black": "#0A0A0A", "Soft Black": "#2D2D2D", "Dark Brown": "#3B2F2F",
    "Medium Brown": "#5A3825", "Light Brown": "#A0522D", "Chocolate Brown": "#381819",
    "Chestnut Brown": "#4E3629", "Golden Brown": "#996633", "Caramel": "#D19C62",
    "Honey Blonde": "#E6BE8A", "Dark Blonde": "#C9B18E", "Medium Blonde": "#F5DEB3",
    "Light Blonde": "#FAF0BE", "Platinum Blonde": "#F0EAD6", "Ash Blonde": "#D4C5A9",
    "Rose Gold": "#B76E79", "Copper": "#B87333", "Auburn": "#A52A2A", "Burgundy": "#800020",
    "Mahogany": "#582F2F", "Wine Red": "#722F37", "Deep Plum": "#580F41", "Violet": "#8F00FF",
    "Lavender": "#B57EDC", "Lilac": "#C8A2C8", "Pastel Pink": "#FFD1DC", "Hot Pink": "#FF69B4",
    "Bubblegum Pink": "#FF85B2", "Coral": "#FF7F50", "Red": "#FF0000",
    "Silver": "#C0C0C0", "Graphite Grey": "#6E6E6E", "Ice Blue": "#AFEEEE",
    "Teal": "#008080", "Mint Green": "#98FF98", "Forest Green": "#228B22", "Electric Blue": "#0080FF"
}

# ========== Utilities ==========
def hex_to_rgb(hex_color): 
    return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

# Cache the model loading
model_cache = None

def load_model():
    global model_cache
    if model_cache is None:
        net = get_network(network_name).to(device)
        state = torch.load(model_path, map_location=device)
        net.load_state_dict(state['weight'])
        net.eval()
        model_cache = net
    return model_cache

def predict_mask(model, image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        pred = torch.sigmoid(logit.cpu())[0][0].numpy()
    return (pred >= 0.5).astype(np.uint8)

def create_soft_mask(mask, blur_radius=5, feather_distance=10):
    """
    Create a soft mask with blended edges for more natural color application
    
    Args:
        mask: Binary mask (0s and 1s)
        blur_radius: Gaussian blur radius for edge softening
        feather_distance: Distance for edge feathering effect
    
    Returns:
        Soft mask with values between 0 and 1
    """
    # Convert to float for processing
    soft_mask = mask.astype(np.float32)
    
    # Method 1: Gaussian blur for basic edge softening
    if blur_radius > 0:
        soft_mask = cv2.GaussianBlur(soft_mask, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
    
    # Method 2: Distance-based feathering for more natural edges
    if feather_distance > 0:
        # Calculate distance from edges
        edges = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
        distance_from_edge = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # Create feather effect
        feather_mask = np.clip(distance_from_edge / feather_distance, 0, 1)
        
        # Apply feathering only to mask regions
        soft_mask = soft_mask * feather_mask
    
    # Method 3: Morphological operations for better edge handling
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    soft_mask = cv2.morphologyEx(soft_mask, cv2.MORPH_CLOSE, kernel)
    
    return np.clip(soft_mask, 0, 1)

def apply_color_replace_enhanced(image, mask, color_rgb, alpha=0.7, preserve=True, 
                               blur_radius=3, feather_distance=8, edge_smoothing=True):
    """
    Enhanced color replacement with soft edge blending
    
    Args:
        image: Input image
        mask: Binary mask
        color_rgb: Target color in RGB
        alpha: Color intensity
        preserve: Whether to preserve original lighting
        blur_radius: Gaussian blur radius for edges
        feather_distance: Distance for edge feathering
        edge_smoothing: Whether to apply additional edge smoothing
    
    Returns:
        Image with applied hair color and soft edges
    """
    image = np.array(image)[:, :, ::-1].copy()
    
    # Create soft mask with blended edges
    soft_mask = create_soft_mask(mask, blur_radius, feather_distance)
    
    # Additional edge smoothing if requested
    if edge_smoothing:
        # Apply bilateral filter to preserve edges while smoothing
        soft_mask = cv2.bilateralFilter(soft_mask.astype(np.float32), 9, 75, 75)
        soft_mask = np.clip(soft_mask, 0, 1)
    
    result = image.copy().astype(np.float32)
    
    # Create a 3-channel version of the soft mask
    soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)
    
    if preserve:
        # Preserve natural lighting and texture
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate luminance factor with smoother transitions
        luminance = gray.astype(np.float32) / 255.0
        
        # Apply luminance preservation with soft mask
        for c in range(3):
            # Create base color layer
            color_layer = np.full_like(result[:, :, c], color_rgb[2-c], dtype=np.float32)
            
            # Apply luminance factor
            luminance_factor = 0.3 + 0.7 * luminance
            color_layer = color_layer * luminance_factor
            
            # Blend with soft mask
            result[:, :, c] = (1 - soft_mask_3ch[:, :, c]) * result[:, :, c] + \
                             soft_mask_3ch[:, :, c] * color_layer
    else:
        # Simple color replacement with soft blending
        color_layer = np.full_like(result, color_rgb[::-1], dtype=np.float32)
        result = (1 - soft_mask_3ch) * result + soft_mask_3ch * color_layer
    
    # Final blending with original image
    final_result = cv2.addWeighted(
        image.astype(np.float32), 1 - alpha, 
        result, alpha, 0
    )
    
    # Additional post-processing for more natural look
    final_result = apply_color_enhancement(final_result, soft_mask)
    
    return np.clip(final_result, 0, 255).astype(np.uint8)

def apply_color_enhancement(image, mask):
    """
    Apply subtle color enhancement for more natural results
    """
    # Slight saturation boost in hair regions
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Boost saturation slightly in masked regions
    saturation_boost = 1.1
    hsv[:, :, 1] = hsv[:, :, 1] * (1 + (saturation_boost - 1) * mask)
    
    # Slight brightness adjustment
    brightness_factor = 1.05
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + (brightness_factor - 1) * mask * 0.5)
    
    # Convert back to BGR
    hsv = np.clip(hsv, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    return enhanced

def resize_image_for_display(image, max_width=800):
    """Resize image for display while maintaining aspect ratio"""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size
    
    if w > max_width:
        ratio = max_width / w
        new_w = max_width
        new_h = int(h * ratio)
        
        if isinstance(image, np.ndarray):
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return image

def process_hair_color(image, color_choice, alpha, preserve_lighting, blur_radius, feather_distance, edge_smoothing, show_mask):
    """
    Main processing function for Gradio interface
    """
    if image is None:
        return None, None, "Please upload an image first."
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Resize for processing (optional, depends on your model requirements)
        original_image = resize_image_for_display(image)
        
        # Load model and predict mask
        model = load_model()
        mask = predict_mask(model, original_image)
        
        # Get color RGB values
        rgb_color = hex_to_rgb(HAIR_COLORS[color_choice])
        
        # Apply enhanced color replacement
        result = apply_color_replace_enhanced(
            original_image, mask, rgb_color, 
            alpha=alpha, preserve=preserve_lighting,
            blur_radius=blur_radius, feather_distance=feather_distance,
            edge_smoothing=edge_smoothing
        )
        
        # Convert result back to RGB for display
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Create mask visualization if requested
        mask_visualization = None
        if show_mask:
            soft_mask = create_soft_mask(mask, blur_radius, feather_distance)
            mask_visualization = (soft_mask * 255).astype(np.uint8)
            mask_visualization = cv2.cvtColor(mask_visualization, cv2.COLOR_GRAY2RGB)
        
        # Save result
        output_file = os.path.join("results", f"result_enhanced_{datetime.now().strftime('%H%M%S')}.jpg")
        cv2.imwrite(output_file, result)
        
        success_message = f"✅ Hair color successfully applied! Result saved as {output_file}"
        
        return result_rgb, mask_visualization, success_message
        
    except Exception as e:
        error_message = f"❌ Processing failed: {str(e)}"
        return None, None, error_message

# ========== Custom CSS ==========
custom_css = """
#title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

#title h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    font-weight: 700;
}

#title p {
    font-size: 1.1rem;
    opacity: 0.9;
}

.gradio-container {
    max-width: 1200px;
    margin: 0 auto;
}

.color-preview {
    display: flex;
    align-items: center;
    gap: 10px;
}

.color-swatch {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 2px solid #fff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Custom styling for better appearance */
.gradio-row {
    gap: 1rem;
}

.gradio-column {
    padding: 1rem;
}

#component-0 {
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}
"""

# ========== Gradio Interface ==========
def create_interface():
    with gr.Blocks(css=custom_css, title="Hair Color Simulation Studio") as demo:
        # Header
        gr.HTML("""
        <div id="title">
            <h1>🎨 Hair Color Simulation Studio</h1>
            <p>Transform your look with AI-powered hair color simulation & natural edge blending</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image Upload
                image_input = gr.Image(
                    label="📸 Upload Your Photo",
                    type="pil",
                    height=400
                )
                
                # Color Selection
                color_choice = gr.Dropdown(
                    choices=list(HAIR_COLORS.keys()),
                    value=list(HAIR_COLORS.keys())[0],
                    label="🎨 Choose Hair Color",
                    info="Select your desired hair color"
                )
                
                # Color preview
                def update_color_preview(color_name):
                    hex_color = HAIR_COLORS[color_name]
                    return gr.HTML(f"""
                    <div class="color-preview">
                        <div class="color-swatch" style="background-color: {hex_color};"></div>
                        <span>Selected: {color_name}</span>
                    </div>
                    """)
                
                color_preview = gr.HTML()
                color_choice.change(update_color_preview, inputs=[color_choice], outputs=[color_preview])
                
                # Initialize color preview
                demo.load(lambda: update_color_preview(list(HAIR_COLORS.keys())[0]), outputs=[color_preview])
                
            with gr.Column(scale=2):
                # Results
                with gr.Row():
                    result_image = gr.Image(
                        label="✨ Hair Color Result",
                        type="numpy",
                        height=400
                    )
                    
                    mask_image = gr.Image(
                        label="🎯 Edge Blending Mask",
                        type="numpy",
                        height=400,
                        visible=False
                    )
                
                # Status message
                status_message = gr.Textbox(
                    label="Status",
                    value="Upload an image and click 'Apply Hair Color' to get started!",
                    interactive=False
                )
        
        # Advanced Controls
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                alpha = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.01,
                    label="Color Intensity",
                    info="Higher values apply more intense color"
                )
                
                preserve_lighting = gr.Checkbox(
                    value=True,
                    label="Natural Lighting",
                    info="Preserve original lighting and texture"
                )
        
        # Edge Blending Controls
        with gr.Accordion("🎯 Edge Blending Controls", open=False):
            with gr.Row():
                blur_radius = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Edge Softness",
                    info="Higher values create softer, more blended edges"
                )
                
                feather_distance = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=8,
                    step=1,
                    label="Feather Distance",
                    info="Distance over which edges gradually fade"
                )
            
            with gr.Row():
                edge_smoothing = gr.Checkbox(
                    value=True,
                    label="Advanced Edge Smoothing",
                    info="Apply additional smoothing for ultra-natural results"
                )
                
                show_mask = gr.Checkbox(
                    value=False,
                    label="Show Edge Blending Mask",
                    info="Display the soft mask used for blending"
                )
        
        # Process Button
        process_btn = gr.Button(
            "🚀 Apply Hair Color",
            variant="primary",
            size="lg"
        )
        
        # Event handlers
        process_btn.click(
            fn=process_hair_color,
            inputs=[
                image_input, color_choice, alpha, preserve_lighting,
                blur_radius, feather_distance, edge_smoothing, show_mask
            ],
            outputs=[result_image, mask_image, status_message]
        )
        
        # Show/hide mask image based on checkbox
        def toggle_mask_visibility(show):
            return gr.update(visible=show)
        
        show_mask.change(
            fn=toggle_mask_visibility,
            inputs=[show_mask],
            outputs=[mask_image]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; opacity: 0.7;">
            <p>✨ Powered by AI • Enhanced Edge Blending • Made with Gradio</p>
        </div>
        """)
    
    return demo

# ========== Launch Application ==========
if __name__ == "__main__":
    demo = create_interface()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        debug=True,             # Enable debug mode
        show_error=True         # Show errors in the interface
    )

# To run the app:
# python gradio_app.py