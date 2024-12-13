import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go



def plot_rgb_curves(red_curve, green_curve, blue_curve):
    x_values = np.linspace(0, 1, len(red_curve))
    fig = go.Figure()

    # Add traces for each curve
    fig.add_trace(go.Scatter(x=x_values, y=red_curve, mode='lines+markers', name='Red Curve', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_values, y=green_curve, mode='lines+markers', name='Green Curve', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x_values, y=blue_curve, mode='lines+markers', name='Blue Curve', line=dict(color='blue')))

    # Update layout
    fig.update_layout(
        title="RGB Curves",
        xaxis_title="Input Intensity",
        yaxis_title="Output Intensity",
        template="plotly_white",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


class AdvancedColorGrading:
    def apply_brightness(self, image, exposure):
        exposure_factor = 2 ** exposure
        adjusted_image = image * exposure_factor
        return np.clip(adjusted_image, 0, 1)

    def compute_gradient_vector(self, image):
        gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        max_val = gradient_magnitude.max()
        if max_val > 0:
            gradient_magnitude /= max_val
        return np.clip(gradient_magnitude, 0, 1), dx, dy

    def apply_linearization(self, image, contrast):
        tangent_plane = lambda x: x + contrast*(x - 0.5)
        linearized = tangent_plane(image)
        return np.clip(linearized, 0, 1)

    def apply_smoothing(self, image, sharpness):
        if sharpness == 0.5:
            return image
        if sharpness < 0.5:
            sigma = max(0.1, 20*(1 - sharpness))
            adjusted_image = cv2.GaussianBlur(image, (5, 5), sigmaX=sigma)
        else:
            alpha = (sharpness - 0.5)*2
            kernel = np.array([[0, -1, 0],
                               [-1, 5 + alpha, -1],
                               [0, -1, 0]], dtype=np.float32)
            adjusted_image = cv2.filter2D(image, -1, kernel)
        return np.clip(adjusted_image, 0, 1)

    def optimize_saturation_with_lagrange(self, image, saturation):
        if saturation == 1.0:
            return image
        luminance = np.dot(image, [0.299,0.587,0.114])
        adjusted_image = luminance[:,:,np.newaxis] + (image - luminance[:,:,np.newaxis])*saturation
        return np.clip(adjusted_image,0,1)

    def apply_exposure_adjustment(self, image, brightness):
        if brightness == 0.0:
            return image
        adjusted = image + brightness
        return np.clip(adjusted,0,1)

    def apply_halation(self, image, halation_amount):
        if halation_amount == 0.0:
            return image
        luminance = np.dot(image, [0.299,0.587,0.114])
        threshold = 0.8
        bright_mask = (luminance > threshold).astype(np.float32)
        glow = cv2.GaussianBlur(bright_mask, (21,21), sigmaX=10)
        glow_3d = np.stack([glow]*3, axis=-1)
        adjusted = image + halation_amount * glow_3d
        return np.clip(adjusted,0,1)

    def apply_vignette(self, image, vignette_amount):
        if vignette_amount == 0.0:
            return image
        h, w, _ = image.shape
        cx, cy = w/2.0, h/2.0
        sigma = w*0.5 * 0.5
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x,y)
        Xc = X - cx
        Yc = Y - cy
        r = np.sqrt(Xc**2 + Yc**2)
        vignette_mask = np.exp(-(r**2)/(sigma**2))
        vignette_mask_3d = np.stack([vignette_mask]*3, axis=-1)
        adjusted = image*(1 - vignette_amount) + (image * vignette_mask_3d)*vignette_amount
        return np.clip(adjusted,0,1)

    def apply_temperature_tint(self, image, temperature, tint):
        R = image[:,:,0]
        G = image[:,:,1]
        B = image[:,:,2]

        if temperature > 0:
            factor_t = 1 + temperature*0.3
            R = R * factor_t
            B = B / factor_t
        elif temperature < 0:
            factor_t = 1 + abs(temperature)*0.3
            B = B * factor_t
            R = R / factor_t

        if tint > 0:
            factor_i = 1 + tint*0.3
            G = G * factor_i
            R = R / factor_i
        elif tint < 0:
            factor_i = 1 + abs(tint)*0.3
            R = R * factor_i
            G = G / factor_i

        adjusted = np.stack([R,G,B], axis=-1)
        return np.clip(adjusted,0,1)

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2],16) for i in (0,2,4))

    def apply_hue_adjustment(self, image, target_color, hue_strength):
        if hue_strength == 0.0:
            return image
        target_rgb = np.array(self.hex_to_rgb(target_color), dtype=np.float32)/255.0
        target_yuv = cv2.cvtColor((target_rgb[np.newaxis,np.newaxis,:]*255).astype(np.uint8), cv2.COLOR_RGB2YUV)
        target_yuv = target_yuv.astype(np.float32)/255.0
        target_U, target_V = target_yuv[0,0,1], target_yuv[0,0,2]

        yuv = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_RGB2YUV)
        yuv = yuv.astype(np.float32)/255.0
        U = yuv[:,:,1]
        V = yuv[:,:,2]

        U_final = (1 - hue_strength)*U + hue_strength*target_U
        V_final = (1 - hue_strength)*V + hue_strength*target_V
        yuv[:,:,1] = U_final
        yuv[:,:,2] = V_final

        adjusted = cv2.cvtColor((yuv*255).astype(np.uint8), cv2.COLOR_YUV2RGB).astype(np.float32)/255.0
        return np.clip(adjusted,0,1)

    def apply_gradient_mapping(self, image, gradient_intensity, gradient_color1, gradient_color2):
        if gradient_intensity == 0.0:
            return image, None, None

        # Convert gradient colors to RGB arrays
        color1_rgb = np.array(self.hex_to_rgb(gradient_color1), dtype=np.float32) / 255.0
        color2_rgb = np.array(self.hex_to_rgb(gradient_color2), dtype=np.float32) / 255.0

        # Compute normalized intensity map based on grayscale values
        gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        intensity_map = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min())

        # Create an alpha map directly from the intensity map
        alpha = np.power(intensity_map, 1.5)  # Adjust power to emphasize mid-tones

        # Blend the two colors based on the alpha map
        highlight = color1_rgb * (1 - alpha[:, :, np.newaxis]) + color2_rgb * alpha[:, :, np.newaxis]
        weight = gradient_intensity * alpha[:, :, np.newaxis]

        # Apply the gradient mapping to the image
        adjusted = image * (1 - weight) + highlight * weight
        return np.clip(adjusted, 0, 1), intensity_map, alpha

    def apply_rgb_curves(self, image, red_curve, green_curve, blue_curve):
        """
        Apply RGB curves to adjust individual channel intensities.
        """
        # Apply curves to each channel
        red_channel = np.interp(image[:, :, 0], np.linspace(0, 1, len(red_curve)), red_curve)
        green_channel = np.interp(image[:, :, 1], np.linspace(0, 1, len(green_curve)), green_curve)
        blue_channel = np.interp(image[:, :, 2], np.linspace(0, 1, len(blue_curve)), blue_curve)

        # Merge adjusted channels
        adjusted_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        return np.clip(adjusted_image, 0, 1)

    def color_grade_image(self, image, params):
        image_float = image.astype(np.float32)/255.0

        exposed_image = self.apply_brightness(image_float, params['brightness'])
        smoothed_image = self.apply_smoothing(exposed_image, params['sharpness'])
        linearized_image = self.apply_linearization(smoothed_image, params['contrast'])
        saturated_image = self.optimize_saturation_with_lagrange(linearized_image, params['saturation'])

        grad_mag, dx, dy = self.compute_gradient_vector(saturated_image)

        bright_image = self.apply_exposure_adjustment(saturated_image, params['exposure'])
        halation_image = self.apply_halation(bright_image, params['halation'])
        vignette_image = self.apply_vignette(halation_image, params['vignette'])
        temp_tint_image = self.apply_temperature_tint(vignette_image, params['temperature'], params['tint'])
        hue_image = self.apply_hue_adjustment(temp_tint_image, params['hue_color'], params['hue_strength'])
        gradient_mapped_image, grad_norm_map, alpha_map = self.apply_gradient_mapping(
            hue_image, params['gradient_intensity'], params['gradient_color1'], params['gradient_color2'])

        # Apply RGB curves
        rgb_curved_image = self.apply_rgb_curves(
            gradient_mapped_image, params['red_curve'], params['green_curve'], params['blue_curve'])

        final_image = np.clip(rgb_curved_image, 0, 1)

        return (final_image * 255).astype(np.uint8), grad_mag, smoothed_image, grad_norm_map, alpha_map


def main():
    st.title("Multivar Calculus Color Grader")
    st.sidebar.header("Upload and Adjust")
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        grader = AdvancedColorGrading()


        presets = {
            "None": {  # Original values when no preset is selected
                "brightness": 0.0, "contrast": 0.0, "saturation": 1.0, "sharpness": 0.5, "exposure": 0.0,
                "halation": 0.0, "vignette": 0.0, "temperature": 0.0, "tint": 0.0, "hue_color": "#00ffff",
                "hue_strength": 0.0, "gradient_color1": "#ff0000", "gradient_color2": "#0000ff", "gradient_intensity": 0.0,
                "red_curve": [0.0, 0.5, 1.0], "green_curve": [0.0, 0.5, 1.0], "blue_curve": [0.0, 0.5, 1.0]
            },
            "Vintage": {
                "brightness": 0.2, "contrast": -0.1, "saturation": 0.9, "sharpness": 0.0, "exposure": 0.05,
                "halation": 0.01, "vignette": 0.00, "temperature": 0.25, "tint": 0.0, "hue_color": "#dec8a5",
                "hue_strength": 0.3, "gradient_color1": "#bda170", "gradient_color2": "#a06b38", "gradient_intensity": 0.3,
                "red_curve": [0.09, 0.58, 1.00], "green_curve": [0.0, 0.57, 1], "blue_curve": [0.0, 0.4, 0.95]
            },
            "Moody": {
                "brightness": 0.3, "contrast": 0.2, "saturation": 1.1, "sharpness": 0.5, "exposure": -0.1,
                "halation": 0.02, "vignette": 0.1, "temperature": -0.15, "tint": 0.0, "hue_color": "#2400ff",
                "hue_strength": 0.05, "gradient_color1": "#7d97c7", "gradient_color2": "#28334e", "gradient_intensity": 0.4,
                "red_curve": [0.0, 0.43, 0.94], "green_curve": [0.05, 0.46, 0.96], "blue_curve": [0.04, 0.54, 1]
            },
            "Grunge": {
                "brightness": 0.3, "contrast": 0.2, "saturation": 1.1, "sharpness": 0.6, "exposure": -0.1,
                "halation": 0.02, "vignette": 0.1, "temperature": -0.15, "tint": 0.0, "hue_color": "#00ff3f",
                "hue_strength": 0.15, "gradient_color1": "#7d97c7", "gradient_color2": "#284e2d", "gradient_intensity": 0.5,
                "red_curve": [0.0, 0.43, 0.94], "green_curve": [0.05, 0.51, 0.96], "blue_curve": [0.04, 0.54, 1]
            },
            "Nostalgia": {
                "brightness": 0.3, "contrast": 0.3, "saturation": 1.2, "sharpness": 0.5, "exposure": -0.05,
                "halation": 0.02, "vignette": 0.1, "temperature": -0.15, "tint": 0.0, "hue_color": "#ffc500",
                "hue_strength": 0.15, "gradient_color1": "#e538aa", "gradient_color2": "#e2983b", "gradient_intensity": 0.4,
                "red_curve": [0.17, 0.43, 0.94], "green_curve": [0.05, 0.51, 0.96], "blue_curve": [0.17, 0.54, 1]
            }
        }
        
        # Select preset from sidebar
        preset = st.sidebar.selectbox("Choose a preset", options=["None", "Vintage", "Moody", "Grunge", "Nostalgia"])
        
        # Get the preset values
        preset_values = presets[preset]
        
        # Update sliders and color pickers based on preset values
        with st.sidebar.expander("Basic Adjustments"):
            brightness = st.slider("Brightness", -2.0, 2.0, preset_values["brightness"], 0.1)
            contrast = st.slider("Contrast", -1.0, 1.0, preset_values["contrast"], 0.1)
            saturation = st.slider("Saturation", 0.0, 2.0, preset_values["saturation"], 0.1)
            sharpness = st.slider("Sharpness", 0.0, 1.0, preset_values["sharpness"], 0.05)
        
        with st.sidebar.expander("Creative Adjustments"):
            exposure = st.slider("Exposure", -0.5, 0.5, preset_values["exposure"], 0.05)
            halation = st.slider("Halation", 0.0, 0.2, preset_values["halation"], 0.01)
            vignette = st.slider("Vignette", 0.0, 1.0, preset_values["vignette"], 0.05)
        
        with st.sidebar.expander("Color Adjustments"):
            temperature = st.slider("Temperature", -0.5, 0.5, preset_values["temperature"], 0.05)
            tint = st.slider("Tint", -0.5, 0.5, preset_values["tint"], 0.05)
            hue_color = st.color_picker("Select a Hue Color", preset_values["hue_color"])
            hue_strength = st.slider("Hue Strength", 0.0, 1.0, preset_values["hue_strength"], 0.05)
        
        with st.sidebar.expander("Gradient Mapping"):
            gradient_color1 = st.color_picker("Gradient Color 1", preset_values["gradient_color1"])
            gradient_color2 = st.color_picker("Gradient Color 2", preset_values["gradient_color2"])
            gradient_intensity = st.slider("Gradient Intensity", 0.0, 1.0, preset_values["gradient_intensity"], 0.1)
        
        # Update RGB Curves to handle each point separately
        with st.sidebar.expander("RGB Curves"):
            red_curve_points = [
                st.slider("Red Curve Point 1", 0.0, 1.0, preset_values["red_curve"][0], 0.01),
                st.slider("Red Curve Point 2", 0.0, 1.0, preset_values["red_curve"][1], 0.01),
                st.slider("Red Curve Point 3", 0.0, 1.0, preset_values["red_curve"][2], 0.01)
            ]
            green_curve_points = [
                st.slider("Green Curve Point 1", 0.0, 1.0, preset_values["green_curve"][0], 0.01),
                st.slider("Green Curve Point 2", 0.0, 1.0, preset_values["green_curve"][1], 0.01),
                st.slider("Green Curve Point 3", 0.0, 1.0, preset_values["green_curve"][2], 0.01)
            ]
            blue_curve_points = [
                st.slider("Blue Curve Point 1", 0.0, 1.0, preset_values["blue_curve"][0], 0.01),
                st.slider("Blue Curve Point 2", 0.0, 1.0, preset_values["blue_curve"][1], 0.01),
                st.slider("Blue Curve Point 3", 0.0, 1.0, preset_values["blue_curve"][2], 0.01)
            ]



        
        params = {
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'sharpness': sharpness,
            'exposure': exposure,
            'halation': halation,
            'vignette': vignette,
            'temperature': temperature,
            'tint': tint,
            'hue_color': hue_color,
            'hue_strength': hue_strength,
            'gradient_color1': gradient_color1,
            'gradient_color2': gradient_color2,
            'gradient_intensity': gradient_intensity,
            'red_curve': red_curve_points,
            'green_curve': green_curve_points,
            'blue_curve': blue_curve_points
        }


        graded_image, gradient_image, smoothed_image, grad_norm_map, alpha_map = grader.color_grade_image(original_image, params)


        gradient_vis = (gradient_image * 255).astype(np.uint8)

        if grad_norm_map is not None:
            grad_norm_vis = (grad_norm_map * 255).astype(np.uint8)

        if alpha_map is not None:
            alpha_vis = (alpha_map * 255).astype(np.uint8)

        col1, col2 = st.columns(2)
        with col1:
            st.header("Original")
            st.image(original_image, use_container_width=True)
        with col2:
            st.header("Graded")
            st.image(graded_image, use_container_width=True)


        # Display RGB curves
        st.plotly_chart(plot_rgb_curves(params['red_curve'], params['green_curve'], params['blue_curve']), use_container_width=True)


if __name__ == "__main__":
    main()

