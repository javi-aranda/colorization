import streamlit as st
from PIL import Image
import numpy as np
import cv2

@st.cache_resource()
def load_model():
    prototxt_path = 'model/colorization_deploy_v2.prototxt'
    model_path = 'model/colorization_release_v2.caffemodel'
    kernel_path = 'model/pts_in_hull.npy'
    
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)
    
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype='float32')]
    
    return net

def colorize_image(image, net):
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    return (colorized * 255).astype("uint8")

def main():
    st.set_page_config(page_title='Colorize It!', page_icon="🖌️", layout="centered")
    st.title("Black and White Image Colorizer")
    
    uploaded_file = st.file_uploader("Choose a black and white image...", type=["jpg", "jpeg", "png"], )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Colorize Image'):
            net = load_model()
            
            # Convert PIL Image to numpy array
            image_array = np.array(image.convert('RGB'))
            
            colorized_image = colorize_image(image_array, net)
            
            st.image(colorized_image, caption='Colorized Image', use_column_width=True)

if __name__ == "__main__":
    main()