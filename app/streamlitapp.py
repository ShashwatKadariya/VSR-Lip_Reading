# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.title('LipNet Replication')
    st.info('This model is a modification of actual LipNet model.')

st.title('LipNet Application') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info(f'This is all the machine learning model sees when making a prediction')
        sample = load_data(tf.convert_to_tensor(file_path))
        out = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in [sample[1]]]

            
            # Drop the last dimension (channel dimension)
        sample_frames = np.squeeze(sample[0], axis=-1)

        # imageio.mimsave('animation.gif', sample_frames, fps=10)
        st.image('animation.gif', width=400) 

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(sample[0], axis=0))
        decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
        st.text(decoded)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')
        st.text(converted_prediction)


        # actual text
        st.info('Predicted Text')
        st.text(out)
        
