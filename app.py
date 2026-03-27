import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
from datetime import datetime

# --- 1. PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="Lung Diagnostic AI", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

# Inject custom CSS to make it look like a modern website AND format the print layout
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A8A; 
            text-align: center;
            margin-bottom: 0px;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #6B7280;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* HIGHLIGHT THE RUN BUTTON */
        button[kind="primary"] {
            background-color: #2563EB !important; /* Rich Blue */
            color: white !important;
            border-radius: 8px !important;
            height: 60px !important;
            font-size: 20px !important;
            font-weight: bold !important;
            border: none !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            transition: all 0.3s ease !important;
        }
        button[kind="primary"]:hover {
            background-color: #1D4ED8 !important; /* Darker Blue on Hover */
            transform: scale(1.02) !important;
        }

        /* =========================================================
           PRINT-SPECIFIC CSS: Hides UI elements when saving to PDF
           ========================================================= */
        @media print {
            /* Hide Streamlit Input Widgets & Buttons */
            [data-testid="stFileUploader"], 
            [data-testid="stNumberInput"], 
            [data-testid="stSelectbox"], 
            button, 
            iframe { 
                display: none !important; 
            }
            
            /* Hide everything tagged with our custom 'no-print' class */
            .no-print { display: none !important; }
            .sub-header { display: none !important; }
            
            /* Force exact background colors to print */
            * {
                -webkit-print-color-adjust: exact !important;
                color-adjust: exact !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINE MODEL ARCHITECTURE & LOAD
# ==========================================
class FusionModel(nn.Module):
    def __init__(self, num_classes=4): 
        super(FusionModel, self).__init__()
        self.cnn = models.resnet50(weights=None)
        self.cnn.fc = nn.Identity() 
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 16),    
            nn.ReLU(),           
            nn.Linear(16, 8)     
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 8, 128),  
            nn.ReLU(),                 
            nn.Linear(128, num_classes)
        )

    def forward(self, image, metadata):
        img_features = self.cnn(image)
        meta_features = self.meta_fc(metadata)
        fused_features = torch.cat((img_features, meta_features), dim=1)
        output = self.classifier(fused_features)
        return output

@st.cache_resource
def load_model():
    model = FusionModel()
    model.load_state_dict(torch.load('day4_fusion_model.pth', map_location=torch.device('cpu')))
    model.eval() 
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading AI Model. Please check the server logs. Details: {e}")

# ==========================================
# 3. GRAD-CAM EXPLAINABILITY ENGINE
# ==========================================
def generate_gradcam(model, image_tensor, metadata_tensor, target_class_idx):
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

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    activation = activations[0].squeeze()
    for i in range(activation.size(0)):
        activation[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activation, dim=0).squeeze()
    heatmap = F.relu(heatmap)
    
    if torch.max(heatmap) > 0:
        heatmap /= torch.max(heatmap)

    return heatmap.detach().cpu().numpy()

# Function to convert image for HTML preview
def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# ==========================================
# 4. MODERN WEB INTERFACE
# ==========================================

# Hero Section
st.markdown('<div class="main-header">🫁 PulmoAI Diagnostic Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Multimodal Lung Cancer Detection System</div>', unsafe_allow_html=True)

# Main Content Layout
left_pane, right_pane = st.columns([1, 1], gap="large")

with left_pane:
    # Card 1: Patient Data
    with st.container(border=True):
        # We use HTML for the header so we can hide it during print with class="no-print"
        st.markdown('<div class="no-print"><h3 style="margin-top:0px;">👤 1. Patient Profile</h3><p style="color:gray;">Enter the patient\'s clinical metadata below.</p></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (Years)", min_value=1, max_value=120, value=65)
        with col2:
            gender = st.selectbox("Biological Sex", ["Male", "Female"])
        
        smoking_years = st.number_input("Smoking History (Pack-Years)", min_value=0, max_value=100, value=20)
        gender_val = 0 if gender == "Male" else 1

with right_pane:
    # Card 2: Image Upload & Preview
    with st.container(border=True):
        st.markdown('<div class="no-print"><h3 style="margin-top:0px;">🩻 2. Radiological Scan</h3><p style="color:gray;">Upload a high-resolution CT Scan (Axial view).</p></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drag and drop file here", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        
        # UI PREVIEW (Hidden in Print)
        if uploaded_file is not None:
            image_ready = Image.open(uploaded_file).convert('RGB')
            img_b64 = get_image_base64(image_ready)
            # This HTML image is tagged "no-print" so it shows on the website but vanishes on the PDF
            st.markdown(f"""
                <div class="no-print" style="text-align: center; margin-top: 15px;">
                    <img src="data:image/jpeg;base64,{img_b64}" style="max-width: 60%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"/>
                    <p style="color: gray; font-size: 13px; margin-top: 5px;">Uploaded Scan Preview</p>
                </div>
            """, unsafe_allow_html=True)

# --- ACTION SECTION ---
st.write("") 
if uploaded_file is not None:
    
    # Center the button 
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button("🚀 Run AI Diagnostic Analysis", type="primary")

    # --- RESULTS DASHBOARD (THIS IS WHAT PRINTS) ---
    if analyze_button and model_loaded:
        st.divider()
        st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🩺 Clinical Analysis Report</h2>", unsafe_allow_html=True)
        
        # --- PATIENT SUMMARY BANNER (Prints perfectly on PDF) ---
        current_date = datetime.now().strftime("%B %d, %Y - %H:%M")
        st.markdown(f"""
        <div style="background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; margin-bottom: 25px;">
            <h4 style="margin-top: 0px; color: #1E3A8A; font-size: 18px; border-bottom: 1px solid #CBD5E1; padding-bottom: 10px;">📋 Patient & Scan Details</h4>
            <div style="display: flex; justify-content: space-between; color: #374151; font-size: 16px; margin-top: 15px;">
                <div><strong>Age:</strong> {age} Years</div>
                <div><strong>Sex:</strong> {gender}</div>
                <div><strong>Smoking History:</strong> {smoking_years} Pack-Years</div>
                <div><strong>Report Date:</strong> {current_date}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Processing deep visual features and integrating clinical metadata..."):
            
            # Prepare Data
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            input_image = transform(image_ready).unsqueeze(0) 
            input_metadata = torch.tensor([[age, gender_val, smoking_years]], dtype=torch.float32)

            # Prediction
            with torch.no_grad():
                output = model(input_image, input_metadata)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class_idx].item()

            classes = ["Adenocarcinoma", "Benign Tumor", "Malignant Tumor", "Normal"]
            prediction_label = classes[predicted_class_idx]
            
            # --- DASHBOARD LAYOUT ---
            metric_col, status_col = st.columns([1, 3])
            
            with metric_col:
                with st.container(border=True):
                    st.metric(label="AI Confidence Score", value=f"{confidence * 100:.1f}%")
            
            with status_col:
                with st.container(border=True):
                    st.write("**Primary AI Finding:**")
                    if prediction_label == "Normal":
                        st.success(f"### ✅ {prediction_label} Tissue Detected")
                    elif prediction_label == "Benign Tumor":
                        st.warning(f"### ⚠️ {prediction_label} Detected")
                    else:
                        st.error(f"### 🛑 {prediction_label} Detected")
            
            st.write("")
            
            # --- SIDE-BY-SIDE IMAGE OUTPUT ---
            st.markdown("### 🔍 Radiological Findings & AI Explainability")
            
            try:
                heatmap = generate_gradcam(model, input_image, input_metadata, predicted_class_idx)
                original_img_np = np.array(image_ready)
                heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
                
                heatmap_colored = np.uint8(255 * heatmap_resized)
                heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                alpha = 0.45 
                superimposed_img = cv2.addWeighted(original_img_np, 1 - alpha, heatmap_colored, alpha, 0)
                
                # Render the original scan next to the Grad-CAM scan
                out_col1, out_col2 = st.columns(2, gap="large")
                with out_col1:
                    st.markdown("**Original Patient CT Scan**")
                    st.image(image_ready, use_column_width=True)
                with out_col2:
                    st.markdown("**AI Localization Heatmap (Grad-CAM)**")
                    st.image(superimposed_img, caption=f"Regions triggering '{prediction_label}'", use_column_width=True)
            except Exception as e:
                st.error(f"Could not generate Grad-CAM visualization. Error: {e}")
                
            st.caption("Disclaimer: This tool is for research and decision-support purposes only and does not replace professional medical diagnosis.")

            # --- ISOLATED PRINT BUTTON ---
            st.write("")
            components.html(
                """
                <div style="text-align: center; margin-top: 10px;">
                    <button onclick="window.parent.print()" style="background-color: #10B981; color: white; padding: 12px 24px; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family: sans-serif; transition: 0.3s;">
                        🖨️ Print Clinical Report to PDF
                    </button>
                </div>
                """,
                height=80
            )

# ==========================================
# 5. EDUCATIONAL RESOURCE SECTION (PURE HTML)
# ==========================================
# Built purely in HTML so we can wrap the whole thing in <div class="no-print">
st.markdown("""
<div class="no-print">
    <hr>
    <h2 style='text-align: center; color: #1E3A8A; margin-bottom: 20px;'>📚 Lung Health & Clinical Guide</h2>
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 300px; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px; background-color: #ffffff;">
            <h3 style="margin-top: 0; color: #1E3A8A;">🔬 Understanding Lung Tumors</h3>
            <p>Lung cancer is one of the most critical diseases worldwide. Early detection via CT scans drastically improves survival rates. Our AI classifies scans into four primary categories:</p>
            <ul style="line-height: 1.6;">
                <li>🟢 <b>Normal Tissue:</b> Healthy lung structure with no signs of abnormal cell growth.</li>
                <li>🟡 <b>Benign Tumor:</b> A non-cancerous growth. Does not spread to other parts of the body.</li>
                <li>🔴 <b>Malignant Tumor:</b> A cancerous mass that grows aggressively and can metastasize.</li>
                <li>🟣 <b>Adenocarcinoma:</b> The most common subtype of Non-Small Cell Lung Cancer (NSCLC).</li>
            </ul>
        </div>
        <div style="flex: 1; min-width: 300px; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px; background-color: #ffffff;">
            <h3 style="margin-top: 0; color: #1E3A8A;">🛡️ Lung Health Maintenance</h3>
            <p>Proactive lung care is essential for preventing respiratory diseases and minimizing cancer risks.</p>
            <ul style="line-height: 1.6;">
                <li>🚭 <b>Eliminate Tobacco Use:</b> Smoking is the leading cause of lung cancer.</li>
                <li>🌫️ <b>Avoid Toxins:</b> Limit exposure to secondhand smoke, asbestos, and radon.</li>
                <li>🏃 <b>Exercise:</b> Regular aerobic activities improve lung capacity.</li>
                <li>🩺 <b>Screenings:</b> High-risk individuals should undergo annual LDCT scans.</li>
                <li>🥦 <b>Diet:</b> Antioxidant-rich diets protect lung tissue from cellular damage.</li>
            </ul>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)