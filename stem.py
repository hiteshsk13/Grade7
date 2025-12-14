import google.generativeai as genai
import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Crop Guard AI",
    page_icon="🍎",
)

st.title("🍎 Crop Guard AI")
st.markdown("""
<b><u>About:</b></u><br> This is built to detect pesticides, wax, rot and ripe.<br>
""", unsafe_allow_html=True)
st.markdown("""<b><u>instructions:</b></u><br> upload 4 images of a fruit or vegetable, each one side of the fruit,
then click the analyze button when it is green and wait till you recieve your answer.<br>
""", unsafe_allow_html=True)


genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #b2f7b2, #add8e6, #4a7c4a);
    background-attachment: fixed;
    color: black;  /* sets all text to black */
}

h1, h2, h3, h4, h5, h6, p, span, div {
    color: black !important;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# Upload multiple images
uploaded_images = st.file_uploader(
    "Upload 4 images of your fruit",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Only enable submit if exactly 12 images are uploaded
submit = st.button(
   "🔍Analyze Images",
   disabled=(len(uploaded_images) != 4 if uploaded_images else True)
)


#justs lets the user know they have to add more pictures, and the if and eles is for plural and singular
if uploaded_images:
    image = Image.open(uploaded_images[0])
    st.image(image, caption="Uploaded Image",width=300)
    if len(uploaded_images) == 4:
        st.success ("✅Your all set")
    else:
        if len(uploaded_images) == 1 - 3:
            st.error(f"⚠️You have uploaded 1 image. Upload exactly { 4-len(uploaded_images)} more images to enable analysis.⚠️")
        else:
            st.error(f"⚠️You have uploaded {len(uploaded_images)} images. Upload exactly { 4-len(uploaded_images)} more images to enable analysis.⚠️")

def analyze_image_cv(image_cv):
    avg_color = np.mean(image_cv, axis=(0,1))
    #this looks at every single pixel and averages them, like this (220, 200, 432)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    #this defines gray as the color gray
    shine = np.sum(gray > 200) / gray.size * 100
    #this looks at the picture and shows if it is shiny by seeing the value of gray, it looks at every single pixel
    # and sees how much of the apples pixels are not bright and shiny, 200 and above is very bright, grey is 100
    # also says how much percent of the shiny
    dark_spots = np.sum(gray < 50) / gray.size * 100
    #this does the same thins, but it looks at every pixel but this time, it looks at dark spots, like really dark color spots
    #and tells how much percent of the pixels are dark
    color_uniformity = 100 - np.mean(np.std(image_cv, axis=(0,1)))
    # it avredges all the colors and sees how even the colors are, 0 is patchy with some mistakes, 100 is perfect
    return avg_color, shine, dark_spots, color_uniformity
    # this sets the variables so it can be used in the prompts: average color, shine %, dark spots %, color uniformity %


if submit:
    with st.spinner("Analyzing all images together..."):
        all_images_data = []

        for uploaded_image in uploaded_images:
            image = Image.open(uploaded_image)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            avg_color, shine, dark_spots, color_uniformity = analyze_image_cv(image_cv)

            all_images_data.append({
                "avg_color": avg_color,
                "shine": shine,
                "dark_spots": dark_spots,
                "color_uniformity": color_uniformity
            })


        all_images_data.append({
            "avg_color": avg_color.tolist(),
            "shine": float(shine),
            "dark_spots": float(dark_spots),
            "color_uniformity": float(color_uniformity)
        })



    # the prompt is the most important piece of code in the whole AI
    prompt_filled_all = ""
    for idx, d in enumerate(all_images_data, start=1):
        prompt_filled_all += ("""
You are a helpful assistant that analyzes fruits and vegetables for health, ripeness, wax, and pesticide presence based on both visual clues and numeric data from image analysis.

Use the following indicators:

Pesticide/Wax Indicators:
- Shiny or glossy surface
- Perfectly even color
- Very large size
- Smooth surface with no blemishes

Low-Pesticide / Natural Indicators:
- Dull or matte surface
- Uneven color
- Small imperfections or blemishes
- Smaller size

Numeric Data (from OpenCV analysis):
- Average color (BGR): {avg_color}
- Shine percentage: {shine:.2f}
- Dark spot percentage: {dark_spots:.2f}
- Color uniformity: {color_uniformity:.2f}

Tasks:
1. Determine if the fruit is likely treated with pesticides, waxed, ripe, or rotten.
2. Provide precise percentages for Rotten, Ripe, Pesticides, and Wax.
3. Write a 30-word paragraph explaining how healthy or unhealthy the fruit appears, using both the numeric data and visual clues.
4. Use the numeric values to support your answer for higher accuracy.

if the image is uploaded wrong, then tell them this exactly: 
The image you have provided does not appear to be a food item. 
This system is designed to analyze fruits and vegetables for ripeness, wax, and pesticide presence. 
Since the provided image is a graphic and not edible, it cannot be scanned.
Please submit an image of a food item for analysis.
answer like this, Exactly, but change to situation same formate though: 
This food item seems to be fresh and wholesome. You can safely eat it, and it appears to be a very healthy choice. Enjoy it as part of a balanced diet, and it should taste delicious!
if its bad then the oppiste. add suggestions, even if its going to cross 30 words, on what to do with it to eat it safly
DONT HAVE ANYTHING LIKE THIS: The very low shine percentage (1.09%) and poor color uniformity 
(39.83) suggest minimal wax or pesticides, while the rich, uneven coloration indicates good ripeness.
dont give me details on the shine, and other factors i gave you
if the rottens high, that means that there is less pesticides
avredge each picture then show outcome
use this formate, but change to situation:
Here is the analysis of the fruit:
Percentages:
Rotten: 10%
Ripe: 85%
Pesticides: 5%
Wax: 0%
Health Analysis: This fruit seems fresh and wholesome. With 0.00% shine and 54.72% color uniformity, it has minimal pesticide or wax presence.
The low 0.82% dark spots confirm it is ripe, healthy, and safe to enjoy.The fruit is likely ripe and natural, with very low indications of pesticide 
treatment or wax application. There are minimal signs of rot.
""".
        format(
            avg_color=d['avg_color'],
            shine=d['shine'],
            dark_spots=d['dark_spots'],
            color_uniformity=d['color_uniformity']
        ))

    # Converts the first image to bytes
    first_image_bytes = uploaded_images[0].getvalue()

    # Prepare all images list
    all_images = [
        prompt_filled_all,
        {"mime_type": uploaded_images[0].type, "data": first_image_bytes},
    ]

    # Sends everything to Gemini
    response = model.generate_content(all_images)

    # shows the AI result
    st.write("AI Analysis:")
    st.markdown(response.text)









