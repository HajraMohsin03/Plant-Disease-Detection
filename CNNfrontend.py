import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from Architecture import ResidualNN

device = torch.device('cpu')

model_path = 'plant-disease-model-AI-Shortened.pth'  
num_classes = 32
model = ResidualNN(3, num_classes,p=0.25, weight_decay=1e-5) 
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device) 
model.eval()  

disease_list = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Cherry_(including_sour)___healthy',
    'Cherry_(including_sour)___Powdery_mildew','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___healthy','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Peach___Bacterial_spot','Peach___healthy','Potato___Early_blight','Potato___healthy','Potato___Late_blight',
    'Raspberry___healthy','Strawberry___healthy','Strawberry___Leaf_scorch','Tomato___Bacterial_spot','Tomato___Early_blight',
    'Tomato___healthy','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus'
]

def to_device(data):

    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(img):
    img = img.resize((256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).to(device)
    xb = to_device(img_tensor.unsqueeze(0))
    with torch.no_grad():
        yb = model(xb)
    _, preds  = torch.max(yb, dim=1)

    return disease_list[preds[0].item()]

def main():
    st.set_page_config(page_title="Plant Disease Detection", page_icon=":seedling:")
    st.markdown("""
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    </head>
    <h1 style="text-align:center;"><i class="fa fa-leaf" style="color:green;" ></i> Plant Disease Detection</h1>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button('Predict'):
            prediction = predict_image(image)
            st.success(f"The predicted disease is {prediction}")
if __name__ == '__main__':
    main()

