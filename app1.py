import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2

# --- MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Lung Diagnostic AI", layout="wide")

# ==========================================
# 1. DEFINE YOUR MODEL ARCHITECTURE HERE
# ==========================================
class FusionModel(nn.Module):
    def __init__(self, num_classes=4): 
        super(FusionModel, self).__init__()
        
        # 1. IMAGE BRANCH
        self.cnn = models.resnet50(weights=None)
        self.cnn.fc = nn.Identity() 
        
        # 2. METADATA BRANCH
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 16),    
            nn.ReLU(),           
            nn.Linear(16, 8)     # <-- FIXED: Your trained model outputs 8 features here
        )
        
        # 3. FUSION & CLASSIFIER
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 8, 128),  # <-- FIXED: 2048 (image) + 8 (metadata) = 2056. Outputs 128.
            nn.ReLU(),                 
            nn.Linear(128, num_classes)# <-- FIXED: Takes the 128 features and outputs your 4 classes.
        )

    def forward(self, image, metadata):
        img_features = self.cnn(image)
        meta_features = self.meta_fc(metadata)
        
        fused_features = torch.cat((img_features, meta_features), dim=1)
        
        output = self.classifier(fused_features)
        return output

# ==========================================
# 2. LOAD THE TRAINED MODEL (Caching for speed)
# ==========================================
@st.cache_resource
def load_model():
    model = FusionModel()
    model.load_state_dict(torch.load('day4_fusion_model.pth', map_location=torch.device('cpu')))
    model.eval() 
    return model

# Initialize the model
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: Details: {e}")

# ==========================================
# 3. GRAD-CAM FUNCTION (Tuned Version)
# ==========================================
def generate_gradcam(model, image_tensor, metadata_tensor, target_class_idx):
    # Target the final convolutional block of the ResNet50
    target_layer = model.cnn.layer4[-1]

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    def forward_hook(module, input, output):
        activations.append(output)

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    output = model(image_tensor, metadata_tensor)

    model.zero_grad()
    target = output[0, target_class_idx]
    target.backward()

    hook_f.remove()
    hook_b.remove()

    # Global average pooling of gradients
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    
    # Weight the channels by corresponding gradients
    activation = activations[0].squeeze()
    for i in range(activation.size(0)):
        activation[i, :, :] *= pooled_gradients[i]

    # Average the channels of the activation
    heatmap = torch.mean(activation, dim=0).squeeze()
    
    # ReLU on top of the heatmap to only keep features that have a positive influence
    heatmap = F.relu(heatmap)
    
    # Strict Normalization between 0 and 1
    if torch.max(heatmap) > 0:
        heatmap /= torch.max(heatmap)

    return heatmap.detach().cpu().numpy()

# ==========================================
# 4. BUILD THE STREAMLIT USER INTERFACE
# ==========================================
st.title("🫁 Lung Cancer Multimodal Diagnostic System")
st.write("Upload a CT scan and provide patient metadata for automated classification.")

# --- SIDEBAR ---
st.sidebar.header("Patient Metadata")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=65)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
smoking_years = st.sidebar.number_input("Smoking History (Years)", min_value=0, max_value=100, value=20)

gender_val = 0 if gender == "Male" else 1

# --- MAIN AREA ---
uploaded_file = st.file_uploader("Upload CT Scan Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded CT Scan", use_column_width=True)

    if st.button("Run Diagnostic Analysis") and model_loaded:
        with st.spinner("Processing image and metadata..."):
            
            # --- PREPARE DATA ---
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            input_image = transform(image).unsqueeze(0) 
            input_metadata = torch.tensor([[age, gender_val, smoking_years]], dtype=torch.float32)

            # --- PREDICTION ---
            with torch.no_grad():
                output = model(input_image, input_metadata)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class_idx].item()

            # --- DISPLAY RESULTS ---
            classes = ["Adenocarcinoma", "Benign Tumor", "Malignant Tumor", "normal"]
            prediction_label = classes[predicted_class_idx]
            
            with col2:
                st.subheader("Diagnostic Results")
                
                if prediction_label == "Normal":
                    st.success(f"**Primary Prediction:** {prediction_label}")
                elif prediction_label == "Benign Tumor":
                    st.warning(f"**Primary Prediction:** {prediction_label}")
                else:
                    st.error(f"**Primary Prediction:** {prediction_label}")
                
                st.write(f"**Confidence Score:** {confidence * 100:.2f}%")
                
                st.write("---")
                
                # --- GRAD-CAM ---
                st.write("**Explainable AI (Grad-CAM)**")
                
                try:
                    # Generate the raw heatmap
                    heatmap = generate_gradcam(model, input_image, input_metadata, predicted_class_idx)
                    
                    # Convert original PIL image to NumPy array
                    original_img_np = np.array(image)
                    
                    # Resize heatmap to perfectly match the original image size
                    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
                    
                    # Apply Color Map (JET is standard for medical imaging heatmaps)
                    heatmap_colored = np.uint8(255 * heatmap_resized)
                    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    
                    # Blend the images: 60% original image, 40% heatmap
                    alpha = 0.4 
                    superimposed_img = cv2.addWeighted(original_img_np, 1 - alpha, heatmap_colored, alpha, 0)
                    
                    # Display in Streamlit
                    st.image(superimposed_img, caption=f"Grad-CAM Localization ({prediction_label})", use_column_width=True)
                    st.info("Red/Yellow areas indicate the exact structural features the CNN focused on.")
                    
                except Exception as e:
                    st.error(f"Could not generate Grad-CAM visualization. Error: {e}")