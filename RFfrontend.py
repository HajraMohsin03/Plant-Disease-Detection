import RFfeatureExtraction
import streamlit as st
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
            feature_vector = create_feature_space(image)
            prediction = predict_rf(feature_vector)
            st.success(f"The predicted disease is {prediction}")
if __name__ == '__main__':
    main()

