import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

# Function to load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('weights-improvement-10-0.91.hdf5', custom_objects={'KerasLayer': hub.KerasLayer})
    return model

# Set the page title and description
st.set_page_config(
    page_title="Time of the Day Classification",
    page_icon="🌞",
    layout="wide"
)

# Create a navigation widget
page = st.radio("Select a page:", ("Home", "Upload", "About Team"))
if page == "Home":
    # Sample content for the "Home" page
    st.title("Welcome to Time of the Day Classification")
    st.markdown("This is the home page of our web app. We help you classify the time of day as Day Time, Night Time, or Sunrise.")
    st.header("Key Features:")
    st.markdown("1. Upload your image and get instant time-of-day classification.")
    st.markdown("2. Receive safety tips based on the time of day.")
    st.markdown("3. Explore the 'About Team' page to learn more about us.")
    
if page == "Upload":
    # Add a title and description for the "Upload" page
    st.title("Upload an Image for Time of the Day Classification")
    st.markdown("Upload an image, and this website will classify the time of day as Day Time, Night Time, or Sunrise.")

    # File uploader
    file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    # Function to preprocess and predict the image
    def import_and_predict(image_data, model):
        size = (224, 224)
        image = image_data.resize(size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        img_reshape = tf.image.resize(image, size)
        img_reshape = tf.expand_dims(img_reshape, axis=0)
        prediction = model.predict(img_reshape)
        return prediction

    # Load the pre-trained model
    model = load_model()
    class_names = ['Day Time', 'Night Time', 'Sunrise']

    # Display prediction results and safety tips
    if file is None:
        st.warning("Please upload an image file")
    else:
        try:
            image = Image.open(file)

            # Style and display the uploaded image
            st.image(image, use_column_width=True, caption="Uploaded Image")

            prediction = import_and_predict(image, model)
            class_index = tf.argmax(prediction, axis=1).numpy()[0]
            class_label = class_names[class_index]

            # Display the prediction with a custom style
            st.markdown("## Prediction:")
            prediction_text = f"**{class_label}**"
            st.markdown(prediction_text)

            # Add some spacing
            st.markdown("---")

            # Display safety tips based on the time of day
            st.markdown("## Safety Tips:")
            if class_label == 'Day Time':
                st.info("1. Use sunscreen to protect your skin from UV rays.")
                st.info("2. Stay hydrated, especially during outdoor activities.")
                st.info("3. Wear sunglasses to shield your eyes from the sun.")
            elif class_label == 'Night Time':
                st.info("1. Be cautious while driving at night; reduced visibility may be a concern.")
                st.info("2. Keep outdoor areas well-lit to avoid accidents.")
            else:
                st.info("1. Watch the beautiful sunrise safely and enjoy the fresh morning air.")
                st.info("2. Consider taking morning walks to start your day positively.")

        except Exception as e:
            st.error("An error occurred while processing the image. Please make sure you've uploaded a valid image file.")
            st.error(str(e))
if page == "About Team":
    # Sample content for the "About Team" page
    st.title("About Our Team")
    st.markdown("Learn more about the team behind this web app.")
    st.header("Our Team Members:")
    st.markdown("1. Musni, Christian Marc")
    st.markdown("2. Ocampo, Jane Blanca")
    st.markdown("3. Paningbatan, Ryan")
    st.markdown("4. Popes, Mikaela Faye")
    st.markdown("5. Roque, Alexia Jose M. ")
    # ... (You can add more content about the team)

# Add some spacing at the end
st.markdown("---")
