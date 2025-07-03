# streamlit_app.py
import streamlit as st
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
    "jet_black": "#0A0A0A", "soft_black": "#2D2D2D", "dark_brown": "#3B2F2F",
    "medium_brown": "#5A3825", "light_brown": "#A0522D", "chocolate_brown": "#381819",
    "chestnut_brown": "#4E3629", "golden_brown": "#996633", "caramel": "#D19C62",
    "honey_blonde": "#E6BE8A", "dark_blonde": "#C9B18E", "medium_blonde": "#F5DEB3",
    "light_blonde": "#FAF0BE", "platinum_blonde": "#F0EAD6", "ash_blonde": "#D4C5A9",
    "rose_gold": "#B76E79", "copper": "#B87333", "auburn": "#A52A2A", "burgundy": "#800020",
    "mahogany": "#582F2F", "wine_red": "#722F37", "deep_plum": "#580F41", "violet": "#8F00FF",
    "lavender": "#B57EDC", "lilac": "#C8A2C8", "pastel_pink": "#FFD1DC", "hot_pink": "#FF69B4",
    "bubblegum_pink": "#FF85B2", "coral": "#FF7F50", "red": "#FF0000",
    "silver": "#C0C0C0", "graphite_grey": "#6E6E6E", "ice_blue": "#AFEEEE",
    "teal": "#008080", "mint_green": "#98FF98", "forest_green": "#228B22", "electric_blue": "#0080FF"
}

# ========== Custom CSS ==========
def load_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }               
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        text-align: center;
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin: 0;
    }
    
    .control-panel {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .color-option {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin: 2px 0;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .color-option:hover {
        background-color: #f8f9fa;
        transform: translateX(2px);
    }
    
    .color-swatch {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        margin-right: 12px;
        border: 2px solid #fff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .color-name {
        font-weight: 500;
        color: #2c3e50;
        text-transform: capitalize;
    }
    
    .stDownloadButton button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .processing-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    .image-caption {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

# ========== Utilities ==========
def hex_to_rgb(hex_color): 
    return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def create_color_option_html(color_name, hex_color):
    rgb = hex_to_rgb(hex_color)
    display_name = color_name.replace('_', ' ').title()
    return f"""
    <div class="color-option">
        <div class="color-swatch" style="background-color: {hex_color};"></div>
        <span class="color-name">{display_name}</span>
    </div>
    """

@st.cache_resource
def load_model():
    net = get_network(network_name).to(device)
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state['weight'])
    net.eval()
    return net

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

def resize_image_for_display(image, max_width=400):
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

# ========== Streamlit UI ==========
st.set_page_config(
    page_title="Hair Color Simulation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
load_css()

# Header
st.markdown("""
<div class="main-header">
    <h1>🎨 Hair Color Simulation Studio</h1>
    <p>Transform your look with AI-powered hair color simulation & natural edge blending</p>
</div>
""", unsafe_allow_html=True)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Control Panel
    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📸 Upload Your Photo",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo showing your hair"
        )
        
        # Color selection with visual preview
        st.markdown("### 🎨 Choose Hair Color")
        
        # Create color grid for better visualization
        color_cols = st.columns(4)
        selected_color = None
        
        color_items = list(HAIR_COLORS.items())
        for i, (color_name, hex_color) in enumerate(color_items):
            col_idx = i % 4
            with color_cols[col_idx]:
                display_name = color_name.replace('_', ' ').title()
                if st.button(
                    f"⚪ {display_name}",
                    key=f"color_{color_name}",
                    help=f"Click to select {display_name}"
                ):
                    selected_color = color_name
                
                # Show color swatch
                st.markdown(
                    f'<div style="width:100%; height:20px; background-color:{hex_color}; '
                    f'border-radius:10px; margin-bottom:10px; border: 1px solid #ddd;"></div>',
                    unsafe_allow_html=True
                )
        
        # If no color selected yet, default to first one
        if selected_color is None:
            selected_color = list(HAIR_COLORS.keys())[0]
        
        # Store selected color in session state
        if 'selected_color' not in st.session_state:
            st.session_state.selected_color = selected_color
        else:
            st.session_state.selected_color = selected_color
        
        # Advanced controls
        st.markdown("### ⚙️ Advanced Settings")
        
        col_alpha, col_preserve = st.columns(2)
        with col_alpha:
            alpha = st.slider("Color Intensity", 0.0, 1.0, 0.7, 0.01)
        with col_preserve:
            preserve_lighting = st.checkbox("Natural Lighting", value=True)
        
        # Edge blending controls
        st.markdown("### 🎯 Edge Blending Controls")
        col_blur, col_feather = st.columns(2)
        
        with col_blur:
            blur_radius = st.slider(
                "Edge Softness", 0, 10, 3, 1,
                help="Higher values create softer, more blended edges"
            )
        
        with col_feather:
            feather_distance = st.slider(
                "Feather Distance", 0, 20, 8, 1,
                help="Distance over which edges gradually fade"
            )
        
        edge_smoothing = st.checkbox(
            "Advanced Edge Smoothing", value=True,
            help="Apply additional smoothing for ultra-natural results"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

# Image processing and display
if uploaded_file is not None:
    # Create two columns for side-by-side display
    img_col1, img_col2 = st.columns(2)
    
    # Load and resize original image
    original_image = Image.open(uploaded_file).convert('RGB')
    display_original = resize_image_for_display(original_image)
    
    with img_col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown('<p class="image-caption">📸 Original Photo</p>', unsafe_allow_html=True)
        st.image(display_original, use_column_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with img_col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        
        with st.spinner("🔄 Processing your image with natural edge blending..."):
            try:
                # Load model and process
                model = load_model()
                mask = predict_mask(model, original_image)
                rgb_color = hex_to_rgb(HAIR_COLORS[st.session_state.selected_color])
                
                # Apply enhanced color replacement with soft edges
                result = apply_color_replace_enhanced(
                    original_image, mask, rgb_color, 
                    alpha=alpha, preserve=preserve_lighting,
                    blur_radius=blur_radius, feather_distance=feather_distance,
                    edge_smoothing=edge_smoothing
                )
                
                # Resize result for display
                display_result = resize_image_for_display(result[:, :, ::-1])
                
                color_name = st.session_state.selected_color.replace('_', ' ').title()
                st.markdown(f'<p class="image-caption">✨ {color_name} Hair (Enhanced)</p>', unsafe_allow_html=True)
                st.image(display_result, use_column_width=False)
                
                # Show edge blending preview
                if st.checkbox("Show Edge Blending Preview", value=False):
                    soft_mask = create_soft_mask(mask, blur_radius, feather_distance)
                    mask_preview = (soft_mask * 255).astype(np.uint8)
                    mask_preview = resize_image_for_display(mask_preview)
                    st.image(mask_preview, caption="Soft Edge Mask", use_column_width=False)
                
                # Download section
                st.markdown("---")
                output_file = os.path.join("results", f"result_enhanced_{datetime.now().strftime('%H%M%S')}.jpg")
                cv2.imwrite(output_file, result)
                
                with open(output_file, "rb") as file:
                    st.download_button(
                        "📥 Download Enhanced Result",
                        file.read(),
                        file_name=f"hair_color_enhanced_{color_name.lower().replace(' ', '_')}.jpg",
                        mime="image/jpeg"
                    )
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                st.info("Please make sure the model file is available and try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Welcome message when no image is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: white;">
        <h3>👆 Upload a photo to get started</h3>
        <p>Choose a clear photo where your hair is visible for the best results</p>
        <p>✨ Now with enhanced edge blending for ultra-realistic results!</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: white; opacity: 0.7;">
    <p>✨ Powered by AI • Enhanced Edge Blending • Made with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# streamlit run app.py