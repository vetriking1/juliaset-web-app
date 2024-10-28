import streamlit as st
import numpy as np
import plotly.graph_objects as go

# page config title and icon
st.set_page_config(
    page_title="Julia Set Plot",
    page_icon="./icon.png"  
)

# the title of the page
st.title("Interactive Julia Set Visualization")
# the header of the sidebar
st.sidebar.header("Adjust Parameters")

# the subheader of the sidebar
st.sidebar.subheader("Enter Julia Set Parameters")

# sliders for real and imaginary parts with default values
real_slider = st.sidebar.slider("Real Part of c", -2.0, 2.0, -0.79, step=0.01)
img_slider = st.sidebar.slider("Imaginary Part of c", -2.0, 2.0, 0.15, step=0.01)

# text inputs for manually entering real and imaginary parts
real_input = st.sidebar.text_input("Real Part of c input (range -2 to 2)", value=str(real_slider))
img_input = st.sidebar.text_input("Imaginary Part of c input (range -2 to 2)", value=str(img_slider))

# Validate the input and if not valid use the default values from the sliders
try:
    realX = float(real_input)
except ValueError:
    realX = real_slider

try:
    imgY = float(img_input)
except ValueError:
    imgY = img_slider

# Julia set parameters
width, height = 800, 800  # Resolution of the image
max_iter = 150  # total number of iterations

# function to compute the julia set
def julia_set(width, height, x_min, x_max, y_min, y_max, c, max_iter):
    # creating the julia set points using width and height
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    # converting the points into complex numbers
    z = np.array([[complex(re, im) for re in x] for im in y])
    output = np.zeros(z.shape, dtype=int)

    # computing the julia set
    for i in range(height):
        for j in range(width):
            n = 0
            z_ij = z[i, j]
            while abs(z_ij) <= 2 and n < max_iter:
                # formula for the julia set
                z_ij = z_ij * z_ij + c
                n += 1
            output[i, j] = n
    return output

# Creating the Julia set with current parameters
c = complex(realX, imgY)
julia_image = julia_set(width, height, -2.0, 2.0, -2.0, 2.0, c, max_iter)

# Plotting the julia set using plotly library
fig = go.Figure(data=go.Heatmap( 
    # heatmap is used to represent the number of iterations
    z=julia_image,
    colorscale="Viridis",
    colorbar=dict(title="Iterations"),
))

# updating the layout configuration for the plot
fig.update_layout(
    title=f"Julia Set for c = {c}",
    # removing the axis
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False),
    xaxis_title="Re",
    yaxis_title="Im",
    width=700,
    height=700,
)

# this code shows the plot in streamlit app using st.plotly_chart
st.plotly_chart(fig, use_container_width=True)
