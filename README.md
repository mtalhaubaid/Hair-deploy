# Hair Color Simulation API
A powerful API and application suite for realistic hair color simulation using deep learning. This project provides tools to detect hair in images and apply various hair colors with natural-looking results.

## Features
- Advanced Hair Detection : Uses PSPNet with ResNet101 backbone for accurate hair segmentation
- 37 Predefined Colors : Choose from natural shades, fashion colors, and unique bold options
- Smart Blending Modes : Automatically selects the best blending technique based on color properties
- Advanced Settings : Adjust color intensity, preserve natural lighting, and smooth edges
- Multiple Interfaces : Streamlit web application, command-line scripts, and API endpoints
## Installation
```
# Clone the repository
git clone https://github.com/mtalhaubaid/hair-color-api.
git
cd hair-color-api

# Install dependencies
pip install -r requirements.txt
```
The application will automatically download the required model files on first run.

## Usage
### Streamlit Web Application
The easiest way to use the application is through the Streamlit interface:

```
streamlit run streamlit_app.py
```
This will open a web interface where you can:

1. Upload your photo
2. Choose from 37 hair colors
3. Adjust settings like color intensity and lighting preservation
4. Download the result
### Command-line Usage
For batch processing or integration into other workflows, use the Python scripts directly:


```
## Color Options
### Natural Shades
- jet_black, soft_black, dark_brown, medium_brown, light_brown
- chocolate_brown, chestnut_brown, golden_brown, caramel
- honey_blonde, dark_blonde, medium_blonde, light_blonde, platinum_blonde, ash_blonde
### Fashion Colors
- rose_gold, copper, auburn, burgundy, mahogany, wine_red
- deep_plum, violet, lavender, lilac
- pastel_pink, hot_pink, bubblegum_pink, coral, red
### Unique & Bold Colors
- silver, graphite_grey, ice_blue, teal, mint_green, forest_green, electric_blue
## Technical Details
This project uses a PSPNet (Pyramid Scene Parsing Network) with a ResNet101 backbone for hair segmentation. The model achieves 91.8% IoU (Intersection over Union) on test data, providing accurate hair masks even with complex hairstyles.

The color application process uses multiple blending techniques:

- Overlay : Good for most colors, especially vibrant ones
- Multiply : Ideal for natural dark colors
- Replace : Best for very dark or black hair colors
- Soft Light : Creates the most natural look for light colors
The system automatically selects the best blending mode based on color properties.

## Requirements
- Python 3.7+
- PyTorch
- OpenCV
- Streamlit (for web interface)
- FastAPI (for API endpoints)
- Other dependencies in requirements.txt
## Project Structure
- streamlit_app.py : Web interface using Streamlit
- FInal_change_hair_color_ICC_37_colors.py : Advanced script with 37 colors and smart blending
- change_hair_color.py : Basic implementation
- networks/ : Neural network architecture definitions
- models/ : Pre-trained model weights
- results/ : Output directory for processed images
- test_images/ : Sample images for testing
## License
MIT

## Acknowledgements
- PSPNet implementation based on paper
- Hair color palette designed for realistic and fashion-forward options