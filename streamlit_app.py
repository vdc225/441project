
import os
import base64
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Vision-to-Brain",
    page_icon="🧠",
    layout="wide"
)

# Make the top hero wider and closer to the top
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
            max-width: 100%;
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }
    </style>
    """,
    unsafe_allow_html=True
)

DATASET_URL = "https://openneuro.org/datasets/ds004192/versions/1.0.7"

TEAM_MEMBERS = [
    "Vincent Caruso",
    "Krina Shukal"
]

GITHUB_URL = "https://github.com/vdc225/441project"

INPUT_IMAGE_PATH = "input_x_diagram.png"
OUTPUT_IMAGE_PATH = "output_y_diagram.png"
VIDEO_PATH = "assets/brain_loop.mp4"

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("Vision-to-Brain")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Background",
        "Research Question",
        "Data",
        "Preprocessing",
        "Model Performance",
        "Conclusion"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Team")
for member in TEAM_MEMBERS:
    st.sidebar.write(f"- {member}")

st.sidebar.markdown(f"[GitHub Repository]({GITHUB_URL})")

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset")
st.sidebar.markdown(f"[THINGS-fMRI Dataset]({DATASET_URL})")

st.sidebar.markdown("---")

st.sidebar.subheader("Project Variables")

with st.sidebar.expander("Input X: CLIP embeddings"):
    st.caption("X matrix: 8,003 × 512 image-embedding matrix")
    if os.path.exists(INPUT_IMAGE_PATH):
        st.image(INPUT_IMAGE_PATH, use_container_width=True)
    else:
        st.warning(
            f"Image not found: {INPUT_IMAGE_PATH}. "
            "Save your input-X image in the project folder with this filename."
        )

with st.sidebar.expander("Output Y: fMRI voxel responses"):
    st.caption("Y matrix: 8,003 × ~200,000 voxel-response matrix")
    if os.path.exists(OUTPUT_IMAGE_PATH):
        st.image(OUTPUT_IMAGE_PATH, use_container_width=True)
    else:
        st.warning(
            f"Image not found: {OUTPUT_IMAGE_PATH}. "
            "Save your output-Y image in the project folder with this filename."
        )

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def show_video_title_hero(video_path, playback_speed=0.22):
    """
    Show a slow-motion looping video behind the main project title.
    """

    title_text = "Predicting fMRI BOLD responses based on semantic image representation"

    if not os.path.exists(video_path):
        st.title(title_text)
        st.info(
            f"Background video not found: {video_path}. "
            "Save your video as assets/brain_loop.mp4."
        )
        return

    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()

    encoded_video = base64.b64encode(video_bytes).decode()

    components.html(
        f"""
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
            }}

            .hero-full {{
                position: relative;
                width: 100%;
                height: 485px;
                overflow: hidden;
                margin-top: -0.4rem;
                margin-bottom: 0;
                padding: 0;
                background: #020617;
            }}

            .hero-full video {{
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                object-position: center 42%;
                filter: brightness(40%) saturate(92%);
            }}

            .hero-overlay {{
                position: absolute;
                inset: 0;
                background:
                    linear-gradient(
                        180deg,
                        rgba(2, 6, 23, 0.12) 0%,
                        rgba(2, 6, 23, 0.34) 42%,
                        rgba(2, 6, 23, 0.86) 100%
                    ),
                    linear-gradient(
                        90deg,
                        rgba(2, 6, 23, 0.78) 0%,
                        rgba(2, 6, 23, 0.35) 55%,
                        rgba(2, 6, 23, 0.18) 100%
                    );
                z-index: 1;
            }}

            .hero-text {{
                position: absolute;
                z-index: 2;
                left: 5%;
                right: 5%;
                bottom: 52px;
            }}

            .hero-label {{
                color: #93c5fd;
                font-size: 0.95rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.8rem;
                font-family: Arial, Helvetica, sans-serif;
            }}

            .hero-title {{
                color: white;
                font-size: 3.25rem;
                font-weight: 800;
                line-height: 1.08;
                max-width: 1120px;
                margin: 0;
                font-family: Arial, Helvetica, sans-serif;
            }}

            @media (max-width: 900px) {{
                .hero-full {{
                    height: 330px;
                }}

                .hero-text {{
                    left: 24px;
                    right: 24px;
                    bottom: 28px;
                }}

                .hero-title {{
                    font-size: 2.05rem;
                    line-height: 1.15;
                }}
            }}
        </style>

        <div class="hero-full">
            <video id="heroVideo" autoplay muted loop playsinline>
                <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
            </video>

            <div class="hero-overlay"></div>

            <div class="hero-text">
                <div class="hero-label">Vision-to-Brain</div>
                <h1 class="hero-title">{title_text}</h1>
            </div>
        </div>

        <script>
            const video = document.getElementById("heroVideo");

            if (video) {{
                video.playbackRate = {playback_speed};

                video.addEventListener("loadedmetadata", function() {{
                    video.playbackRate = {playback_speed};
                }});
            }}
        </script>
        """,
        height=485,
    )


def show_workflow():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info(
            "**1. Visual Stimuli**\n\n"
            "THINGS object images used as input stimuli"
        )

    with col2:
        st.info(
            "**2. Semantic Feature Extraction**\n\n"
            "CLIP generates 512-dimensional image embeddings"
        )

    with col3:
        st.info(
            "**3. Supervised Regression Modeling**\n\n"
            "Train OLS, Ridge, Lasso, and XGBoost"
        )

    with col4:
        st.info(
            "**4. Predicted Brain Response**\n\n"
            "Predict high-dimensional fMRI response patterns"
        )

# ---------------------------------------------------------
# Results Tables
# ---------------------------------------------------------

def make_results_tables():
    """
    Stores the actual model results from our experiments.
    """

    direct_results = pd.DataFrame([
        ["Subject 1", "Initial direct prediction", -0.000898, 0.017895, 0.775366, 1.005840, 0.793247],
        ["Subject 2", "Initial direct prediction", -0.000806, 0.020496, 0.755087, 1.004219, 0.792966],
        ["Subject 3", "Initial direct prediction", -0.000908, 0.012864, 0.764670, 1.000712, 0.788913],

        ["Subject 1", "Ridge baseline", -0.000688, 0.000001, 0.987281, 1.005632, 0.793121],
        ["Subject 2", "Ridge baseline", -0.000773, 0.000001, 0.987281, 1.005632, 0.793121],
        ["Subject 3", "Ridge baseline", -0.000699, 0.000001, 0.987281, 1.005632, 0.793121],

        ["Subject 1", "Lasso baseline", -0.000688, 0.000001, 0.987281, 1.005632, 0.793121],
        ["Subject 2", "Lasso baseline", -0.000773, 0.000001, 0.987281, 1.005632, 0.793121],
        ["Subject 3", "Lasso baseline", -0.000699, 0.000001, 0.987281, 1.005632, 0.793121],
    ], columns=[
        "Subject",
        "Method",
        "R2 Mean",
        "Top 1% R2",
        "Negative Voxel Fraction",
        "MSE Mean",
        "MAE Mean"
    ])

    pca_results = pd.DataFrame([
        ["Subject 1", "OLS", 0.0921, -0.2201, 198.2313, 187.8345],
        ["Subject 1", "Ridge α=0.1", 0.0898, -0.1754, 198.7151, 183.7834],
        ["Subject 1", "Ridge α=1.0", 0.0729, -0.0825, 202.4204, 175.3736],
        ["Subject 1", "Ridge α=10.0", 0.0358, -0.0126, 210.3973, 168.8280],
        ["Subject 1", "Lasso α=0.001", 0.0705, -0.0818, 200.7209, 179.8816],
        ["Subject 1", "Lasso α=0.01", 0.0092, 0.0007, 211.7757, 169.8120],
        ["Subject 1", "Lasso α=0.1", 0.0000, -0.0006, 217.6862, 167.5978],

        ["Subject 2", "OLS", 0.0924, -0.2518, 190.8067, 173.8676],
        ["Subject 2", "Ridge α=0.1", 0.0901, -0.2006, 191.2805, 169.8724],
        ["Subject 2", "Ridge α=1.0", 0.0732, -0.0941, 194.8911, 161.4147],
        ["Subject 2", "Ridge α=10.0", 0.0361, -0.0149, 202.8494, 155.1943],
        ["Subject 2", "Lasso α=0.001", 0.0708, -0.0925, 193.3168, 165.8217],
        ["Subject 2", "Lasso α=0.01", 0.0095, 0.0005, 204.7128, 155.7088],
        ["Subject 2", "Lasso α=0.1", 0.0001, -0.0006, 210.4190, 154.3754],

        ["Subject 3", "OLS", 0.0902, -0.2240, 188.1142, 176.5418],
        ["Subject 3", "Ridge α=0.1", 0.0880, -0.1786, 188.5976, 172.6395],
        ["Subject 3", "Ridge α=1.0", 0.0711, -0.0845, 192.1585, 164.5175],
        ["Subject 3", "Ridge α=10.0", 0.0342, -0.0143, 199.9006, 158.3157],
        ["Subject 3", "Lasso α=0.001", 0.0676, -0.0810, 190.5627, 168.7176],
        ["Subject 3", "Lasso α=0.01", 0.0077, -0.0000, 201.2713, 159.0654],
        ["Subject 3", "Lasso α=0.1", 0.0000, -0.0006, 206.7127, 157.0028],
    ], columns=["Subject", "Model", "R2 Train", "R2 Test", "MSE Train", "MSE Test"])

    pca_xgboost_results = pd.DataFrame([
        ["Subject 1", "XGBoost PCA-200", 0.319836, -0.045611, 148.379349, 171.939499],
        ["Subject 2", "XGBoost PCA-200", 0.319879, -0.053604, 143.106049, 158.335510],
        ["Subject 3", "XGBoost PCA-200", 0.318597, -0.048085, 140.788635, 161.202499],
        ["Subject 3", "XGBoost PCA-200 tweaked", 0.088917, -0.013137, 188.425705, 158.125549],
    ], columns=["Subject", "Model", "R2 Train", "R2 Test", "MSE Train", "MSE Test"])

    pls20_results = pd.DataFrame([
        ["Subject 1", "OLS", 1.0000, 1.0000, 0.0000, 0.0000],
        ["Subject 1", "Ridge α=0.1", 0.9996, 0.9996, 0.0000, 0.0000],
        ["Subject 1", "Ridge α=1.0", 0.9888, 0.9872, 0.0000, 0.0000],
        ["Subject 1", "Ridge α=10.0", 0.8830, 0.8765, 0.0005, 0.0005],
        ["Subject 1", "Lasso α=0.001", 0.2419, 0.2391, 0.0033, 0.0032],
        ["Subject 1", "Lasso α=0.01", 0.0000, -0.0003, 0.0066, 0.0065],
        ["Subject 1", "Lasso α=0.1", 0.0000, -0.0003, 0.0066, 0.0065],

        ["Subject 2", "OLS", 1.0000, 1.0000, 0.0000, 0.0000],
        ["Subject 2", "Ridge α=0.1", 0.9998, 0.9997, 0.0000, 0.0000],
        ["Subject 2", "Ridge α=1.0", 0.9913, 0.9906, 0.0000, 0.0000],
        ["Subject 2", "Ridge α=10.0", 0.8883, 0.8858, 0.0005, 0.0004],
        ["Subject 2", "Lasso α=0.001", 0.2468, 0.2472, 0.0034, 0.0033],
        ["Subject 2", "Lasso α=0.01", -0.0000, -0.0003, 0.0065, 0.0063],
        ["Subject 2", "Lasso α=0.1", -0.0000, -0.0003, 0.0065, 0.0063],

        ["Subject 3", "OLS", 1.0000, 1.0000, 0.0000, 0.0000],
        ["Subject 3", "Ridge α=0.1", 0.9997, 0.9996, 0.0000, 0.0000],
        ["Subject 3", "Ridge α=1.0", 0.9890, 0.9884, 0.0000, 0.0000],
        ["Subject 3", "Ridge α=10.0", 0.8862, 0.8827, 0.0005, 0.0005],
        ["Subject 3", "Lasso α=0.001", 0.2437, 0.2429, 0.0035, 0.0035],
        ["Subject 3", "Lasso α=0.01", 0.0000, -0.0004, 0.0060, 0.0059],
        ["Subject 3", "Lasso α=0.1", 0.0000, -0.0004, 0.0060, 0.0059],
    ], columns=["Subject", "Model", "R2 Train", "R2 Test", "MSE Train", "MSE Test"])

    pls50_results = pd.DataFrame([
        ["Subject 1", "OLS", 1.0000, 1.0000, 0.0000, 0.0000],
        ["Subject 1", "Ridge α=0.1", 0.9994, 0.9992, 0.0000, 0.0000],
        ["Subject 1", "Ridge α=1.0", 0.9815, 0.9792, 0.0000, 0.0000],
        ["Subject 1", "Ridge α=10.0", 0.8148, 0.8083, 0.0004, 0.0005],
        ["Subject 1", "Lasso α=0.001", 0.0969, 0.0955, 0.0025, 0.0025],
        ["Subject 1", "Lasso α=0.01", 0.0000, -0.0004, 0.0038, 0.0038],
        ["Subject 1", "Lasso α=0.1", 0.0000, -0.0004, 0.0038, 0.0038],

        ["Subject 2", "OLS", 1.0000, 1.0000, 0.0000, 0.0000],
        ["Subject 2", "Ridge α=0.1", 0.9995, 0.9994, 0.0000, 0.0000],
        ["Subject 2", "Ridge α=1.0", 0.9830, 0.9812, 0.0000, 0.0000],
        ["Subject 2", "Ridge α=10.0", 0.8170, 0.8120, 0.0004, 0.0004],
        ["Subject 2", "Lasso α=0.001", 0.0987, 0.0985, 0.0025, 0.0025],
        ["Subject 2", "Lasso α=0.01", 0.0000, -0.0005, 0.0038, 0.0037],
        ["Subject 2", "Lasso α=0.1", 0.0000, -0.0005, 0.0038, 0.0037],

        ["Subject 3", "OLS", 1.0000, 1.0000, 0.0000, 0.0000],
        ["Subject 3", "Ridge α=0.1", 0.9994, 0.9993, 0.0000, 0.0000],
        ["Subject 3", "Ridge α=1.0", 0.9813, 0.9798, 0.0000, 0.0000],
        ["Subject 3", "Ridge α=10.0", 0.8112, 0.8062, 0.0005, 0.0005],
        ["Subject 3", "Lasso α=0.001", 0.0975, 0.0969, 0.0026, 0.0025],
        ["Subject 3", "Lasso α=0.01", -0.0000, -0.0004, 0.0035, 0.0035],
        ["Subject 3", "Lasso α=0.1", -0.0000, -0.0004, 0.0035, 0.0035],
    ], columns=["Subject", "Model", "R2 Train", "R2 Test", "MSE Train", "MSE Test"])

    reconstruction_results = pd.DataFrame([
        ["Subject 1", "PLS-20 Reconstruction", -0.005385, 0.019684, 0.865202, 1.010352],
        ["Subject 2", "PLS-20 Reconstruction", -0.003425, 0.024054, 0.835268, 1.006818],
        ["Subject 3", "PLS-20 Reconstruction", -0.005252, 0.008013, 0.875885, 1.004994],

        ["Subject 1", "PLS-50 Reconstruction", -0.008264, 0.020040, 0.897009, 1.013245],
        ["Subject 2", "PLS-50 Reconstruction", -0.006585, 0.023066, 0.880494, 1.009994],
        ["Subject 3", "PLS-50 Reconstruction", -0.008498, 0.007605, 0.912045, 1.008230],
    ], columns=[
        "Subject",
        "Method",
        "R2 Mean",
        "Top 1% R2",
        "Negative Voxel Fraction",
        "MSE Mean"
    ])

    return {
        "direct": direct_results,
        "pca": pca_results,
        "pca_xgboost": pca_xgboost_results,
        "pls20": pls20_results,
        "pls50": pls50_results,
        "reconstruction": reconstruction_results,
    }

# ---------------------------------------------------------
# Project Overview
# ---------------------------------------------------------

if page == "Project Overview":
    show_video_title_hero(VIDEO_PATH, playback_speed=0.22)

    st.write(
        """
        **Goal:**  
        This project investigates whether semantic image representations from CLIP can predict 
        high-dimensional fMRI BOLD response patterns. Using the THINGS-fMRI dataset, we train 
        supervised regression models on paired image embeddings and brain responses, then evaluate 
        whether those models can predict neural activity for novel, unseen object images.
        """
    )

    st.subheader("Simplified Goal")

    st.success(
        "If we train models on pairs of images and the brain activity they produce, "
        "can we predict what brain activity would look like for new images?"
    )

    st.subheader("Project Workflow")
    show_workflow()

# ---------------------------------------------------------
# Background
# ---------------------------------------------------------

elif page == "Background":
    st.title("Background")

    tab1, tab2, tab3 = st.tabs(["fMRI", "CLIP Embeddings", "THINGS-fMRI"])

    with tab1:
        st.header("Functional MRI")

        st.write(
            """
            Functional MRI (fMRI) measures neural activity indirectly using the blood oxygen 
            level-dependent signal, which reflects changes in blood oxygenation associated with neural activity.
            """
        )

        st.warning(
            "fMRI is not a mind-reading tool. It is indirect, noisy, high-dimensional, "
            "and requires careful statistical validation."
        )

        st.write(
            """
            fMRI divides the brain into small 2–3 mm cubic units called voxels. Each voxel represents 
            aggregated activity from many thousands of neurons.
            """
        )

    with tab2:
        st.header("CLIP Embeddings")

        st.write(
            """
            CLIP is a neural network that maps images into a high-dimensional feature space. 
            These embeddings capture semantic and visual similarity between images.
            """
        )

        st.write(
            """
            In this project, each image is represented by a 512-dimensional CLIP embedding. 
            Across all image samples, the full input matrix X has shape 8,003 × 512.
            """
        )

    with tab3:
        st.header("THINGS-fMRI Dataset")

        st.write(
            """
            The THINGS-fMRI dataset is part of the broader THINGS initiative, which investigates 
            how the human brain represents visual objects and associated semantic concepts.
            """
        )

        st.write(
            """
            In this dataset, three participants viewed thousands of natural object images spanning 
            720 object concepts, with approximately 12 images per concept.
            """
        )

# ---------------------------------------------------------
# Research Question
# ---------------------------------------------------------

elif page == "Research Question":
    st.title("Research Question")

    st.info(
        "Can modern semantic image embeddings explain or predict high-dimensional fMRI response "
        "patterns in the human brain?"
    )

    st.subheader("Context")

    st.write(
        """
        The human brain represents object information through distributed patterns of neural activity. 
        However, it remains unclear how semantic representations of objects relate to these neural 
        patterns in visual cortex and broader voxel-response spaces.
        """
    )

    st.write(
        """
        fMRI has important limitations: it is not a magical neuroimaging technique for reading minds, 
        false positives can be high without proper statistical rigor, and the data is highly dimensional 
        because of the large number of response features.
        """
    )

    st.write(
        """
        This project asks whether CLIP embeddings, which encode semantic and visual information about images, 
        can be used to predict fMRI response patterns for novel, unseen object images.
        """
    )

    st.subheader("Why Is This Important?")

    st.write(
        """
        This project helps us better understand the relationship between visual objects, computational 
        image representations, and human brain activity. It also highlights both the potential and 
        limitations of using high-dimensional fMRI data for prediction.
        """
    )

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

elif page == "Data":
    st.title("Data")

    st.write(
        """
        This project uses the THINGS-fMRI dataset, where participants viewed natural object images 
        while fMRI responses were recorded. Each image is represented using CLIP embeddings, and each 
        corresponding brain response is represented as a high-dimensional voxel-response pattern.
        """
    )

    st.subheader("Key Dataset Dimensions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Participants", "3")

    with col2:
        st.metric("Object Concepts", "720")

    with col3:
        st.metric("Image Samples", "8,003")

    with col4:
        st.metric("Voxel Features", "~200,000")

    st.subheader("Dataset Summary")

    data_summary = pd.DataFrame(
        {
            "Category": [
                "Dataset",
                "Participants",
                "Object concepts",
                "Images per concept",
                "Input X",
                "Input dimension",
                "Output Y",
                "Output dimension",
                "Prediction task"
            ],
            "Description": [
                "THINGS-fMRI",
                "3 participants",
                "720 everyday object concepts",
                "Approximately 12 natural images per concept",
                "CLIP semantic image embedding matrix",
                "8,003 × 512",
                "fMRI voxel-response matrix",
                "8,003 × ~200,000",
                "Predict fMRI response patterns for novel, unseen object images"
            ]
        }
    )

    st.dataframe(data_summary, use_container_width=True, hide_index=True)

    st.subheader("Prediction Setup")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Input X: CLIP Embedding Matrix")
        st.write(
            """
            Each image is converted into a 512-dimensional semantic feature representation using CLIP. 
            Across all image samples, these embeddings form the input matrix X.
            """
        )

        st.info("Model input X: 8,003 × 512 CLIP embedding matrix")

    with col2:
        st.markdown("### Output Y: fMRI Voxel-Response Matrix")
        st.write(
            """
            The target output is the fMRI response pattern across voxel-level brain activity features. 
            Across all image samples, these responses form the output matrix Y.
            """
        )

        st.info("Model output Y: 8,003 × ~200,000 voxel-response matrix")

    st.subheader("Dataset Source")

    st.markdown(f"[OpenNeuro THINGS-fMRI Dataset]({DATASET_URL})")

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

elif page == "Preprocessing":
    st.title("Preprocessing")

    st.write(
        """
        The preprocessing stage prepares the CLIP image embeddings and fMRI response matrix for 
        supervised learning. These steps help make the input and output data more stable, comparable, 
        and suitable for training predictive models.
        """
    )

    st.subheader("Preprocessing Workflow")

    preprocessing_steps = pd.DataFrame(
        {
            "Step": [
                "Normalize input X",
                "Normalize target Y",
                "Apply PCA-based dimensionality reduction",
                "Apply PLS-based dimensionality reduction",
                "Process subjects separately",
                "Create train/test split",
                "Prepare matrices for model"
            ],
            "Purpose": [
                "Apply L2-normalization to the 8,003 × 512 CLIP embedding matrix.",
                "Apply Z-score normalization to the fMRI response matrix.",
                "Reduce the high-dimensional response space using PCA components.",
                "Use PLS to learn latent components relating image embeddings to fMRI responses.",
                "Handle each participant independently because fMRI response patterns can vary across subjects.",
                "Use a 75/25 train/test split to evaluate generalization.",
                "Create aligned input-output matrices for supervised regression models."
            ]
        }
    )

    st.dataframe(preprocessing_steps, use_container_width=True, hide_index=True)

    st.subheader("Important Considerations")

    st.markdown(
        """
        - **High-dimensional output space:** The output matrix Y has shape **8,003 × ~200,000**, making direct prediction difficult.
        - **Subject-specific fMRI patterns:** Each participant may have a different response structure, so subject-level processing is important.
        - **Avoiding data leakage and overfitting:** Preprocessing should be fit using training data only, then applied to test data.
        - **Evaluating on unseen images:** The test set should measure whether models generalize to novel object images.
        """
    )

# ---------------------------------------------------------
# Model Performance
# ---------------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance")

    results = make_results_tables()

    st.write(
        """
        This section presents the actual results from our experiments. 
        We compare baseline models, PCA-based models, PLS-based latent-space models, 
        and PLS reconstruction results.
        """
    )

    st.info(
        "Metrics: R² measures explained variance and MSE measures prediction error. "
        "Higher R² is better; lower MSE is better."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview",
            "Baseline Models",
            "PCA",
            "PLS",
            "PLS Reconstruction"
        ]
    )

    with tab1:
        st.subheader("Experiment Summary")

        summary_table = pd.DataFrame(
            {
                "Component": [
                    "Input matrix X",
                    "Output matrix Y",
                    "Input normalization",
                    "Target normalization",
                    "Train/test split",
                    "PCA components",
                    "PLS components",
                    "Models tested",
                    "Evaluation metrics",
                ],
                "Description": [
                    "8,003 × 512 CLIP embedding matrix",
                    "8,003 × ~200,000 fMRI voxel-response matrix",
                    "L2-normalization on input X",
                    "Z-score normalization on target Y",
                    "75/25 train/test split",
                    "200 manually selected components because of memory limitations",
                    "20 and 50 manually selected components",
                    "OLS, Ridge, Lasso, and XGBoost Regression",
                    "MSE and R² score",
                ],
            }
        )

        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        st.subheader("High-Level Interpretation")

        st.markdown(
            """
            - Baseline models produced near-zero or negative R² values.
            - PCA-based models reduced the output space, but test R² remained weak or negative.
            - XGBoost improved training performance in PCA space, but test R² still remained negative.
            - PLS-20 and PLS-50 produced strong latent-space results.
            - PLS reconstruction back toward voxel-level response space remained difficult, with slightly negative mean R².
            """
        )

    with tab2:
        st.subheader("Baseline Models")

        direct_df = results["direct"]

        st.dataframe(direct_df, use_container_width=True, hide_index=True)

        st.markdown("#### Mean R² by Subject and Method")
        direct_r2 = direct_df.pivot(
            index="Subject",
            columns="Method",
            values="R2 Mean"
        )
        st.bar_chart(direct_r2)

        st.markdown("#### MSE Mean by Subject and Method")
        direct_mse = direct_df.pivot(
            index="Subject",
            columns="Method",
            values="MSE Mean"
        )
        st.bar_chart(direct_mse)

        st.caption(
            "Baseline results show that direct prediction without dimensionality reduction performed poorly, "
            "with mean R² values close to or below zero."
        )

    with tab3:
        st.subheader("PCA Results")

        pca_df = results["pca"]

        selected_subject = st.selectbox(
            "Select subject",
            ["Subject 1", "Subject 2", "Subject 3"],
            key="pca_subject"
        )

        pca_subject_df = pca_df[pca_df["Subject"] == selected_subject]

        st.markdown("#### PCA-200: Test R² by Model")
        pca_r2 = pca_subject_df.set_index("Model")[["R2 Test"]]
        st.bar_chart(pca_r2)

        st.markdown("#### PCA-200: Train vs Test MSE by Model")
        pca_mse = pca_subject_df.set_index("Model")[["MSE Train", "MSE Test"]]
        st.line_chart(pca_mse)

        st.markdown("#### PCA-200 Result Table")
        st.dataframe(pca_subject_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.subheader("PCA with XGBoost")

        pca_xgb_df = results["pca_xgboost"]

        st.dataframe(pca_xgb_df, use_container_width=True, hide_index=True)

        st.markdown("#### XGBoost Test R²")
        xgb_r2 = pca_xgb_df.pivot(
            index="Subject",
            columns="Model",
            values="R2 Test"
        )
        st.bar_chart(xgb_r2)

        st.caption(
            "XGBoost achieved stronger train R² in PCA space, but test R² was still negative."
        )

    with tab4:
        st.subheader("PLS Latent-Space Results")

        pls_choice = st.radio(
            "Select PLS setting",
            ["PLS-20", "PLS-50"],
            horizontal=True
        )

        if pls_choice == "PLS-20":
            pls_df = results["pls20"]
        else:
            pls_df = results["pls50"]

        selected_subject = st.selectbox(
            "Select subject",
            ["Subject 1", "Subject 2", "Subject 3"],
            key="pls_subject"
        )

        pls_subject_df = pls_df[pls_df["Subject"] == selected_subject]

        st.markdown(f"#### {pls_choice}: Test R² by Model")
        pls_r2 = pls_subject_df.set_index("Model")[["R2 Test"]]
        st.bar_chart(pls_r2)

        st.markdown(f"#### {pls_choice}: Train vs Test MSE by Model")
        pls_mse = pls_subject_df.set_index("Model")[["MSE Train", "MSE Test"]]
        st.line_chart(pls_mse)

        st.markdown(f"#### {pls_choice} Result Table")
        st.dataframe(pls_subject_df, use_container_width=True, hide_index=True)

        st.warning(
            "PLS gives very strong latent-space results. These should be interpreted carefully "
            "because strong performance in reduced latent space does not necessarily mean accurate "
            "voxel-level reconstruction."
        )

    with tab5:
        st.subheader("PLS Reconstruction Results")

        recon_df = results["reconstruction"]

        st.dataframe(recon_df, use_container_width=True, hide_index=True)

        st.markdown("#### Mean R² After Reconstruction")
        recon_r2 = recon_df.pivot(
            index="Subject",
            columns="Method",
            values="R2 Mean"
        )
        st.bar_chart(recon_r2)

        st.markdown("#### Negative Voxel Fraction")
        neg_voxels = recon_df.pivot(
            index="Subject",
            columns="Method",
            values="Negative Voxel Fraction"
        )
        st.bar_chart(neg_voxels)

        st.markdown("#### MSE Mean After Reconstruction")
        recon_mse = recon_df.pivot(
            index="Subject",
            columns="Method",
            values="MSE Mean"
        )
        st.bar_chart(recon_mse)

        st.caption(
            "Although PLS latent-space models performed strongly, reconstruction back toward voxel-level "
            "responses remained difficult, with mean R² values still slightly negative."
        )

# ---------------------------------------------------------
# Conclusion
# ---------------------------------------------------------

elif page == "Conclusion":
    st.title("Conclusion")

    st.write(
        """
        Our results show that predicting fMRI BOLD responses from CLIP image embeddings 
        is a challenging high-dimensional regression problem. The input matrix X has shape 
        8,003 × 512, while the output response matrix Y has shape 8,003 × ~200,000. 
        This large output space makes direct voxel-response prediction difficult.

        Baseline models without dimensionality reduction produced near-zero or negative R² values, 
        suggesting that a direct mapping from CLIP embeddings to voxel-level activity does not 
        generalize well.

        PCA-based models reduced the target response space, but test R² values generally remained weak 
        or negative. XGBoost improved training performance in PCA space, but test performance still 
        remained below zero, suggesting overfitting and limited generalization.

        PLS-based models produced much stronger results in the reduced latent space, showing that 
        supervised dimensionality reduction can capture relationships between CLIP embeddings and fMRI 
        responses more effectively than unsupervised PCA. However, reconstruction back toward voxel-level 
        response space remained difficult, with mean voxel-wise R² values still slightly negative.

        Overall, our findings suggest that CLIP embeddings contain information relevant to visual brain 
        responses, but accurately predicting full voxel-wise fMRI activity requires stronger modeling, 
        better dimensionality reduction, and improved reconstruction strategies.
        """
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - **Indirect and noisy signal:** fMRI measures neural activity indirectly through BOLD responses, so voxel-level prediction is inherently difficult.
        - **High-dimensional output space:** The output matrix has shape **8,003 × ~200,000**, making voxel-response prediction computationally and statistically challenging.
        - **Small participant sample:** The dataset includes only **three participants**, so subject-specific variability may strongly affect results.
        - **Manual component selection:** PCA and PLS component counts were manually selected rather than fully optimized through cross-validation.
        - **Computational constraints:** PCA was limited by memory/storage constraints, which restricted the number of component settings we could test.
        - **Latent-space vs voxel-space performance:** Strong PLS latent-space performance does not necessarily mean accurate voxel-level reconstruction.
        - **Limited feature representation:** The current models use CLIP embeddings only, which may not capture all low-level visual features or neural factors that influence fMRI responses.
        """
    )

    st.subheader("Future Work")

    st.markdown(
        """
        - **Tune dimensionality reduction more systematically:** Use cross-validation to select PCA and PLS component counts instead of choosing them manually.
        - **Improve voxel-space reconstruction:** Develop better methods for mapping reduced latent-space predictions back to full voxel-level activity.
        - **Explore region-of-interest analysis:** Predict responses within specific visual regions instead of modeling the full response space at once.
        - **Compare different image representations:** Test CLIP against other embeddings such as CNN features, DINO, or lower-level image features.
        - **Try additional modeling approaches:** Explore models designed for high-dimensional multi-output prediction.
        - **Analyze subject and category differences:** Evaluate whether some participants, object categories, or visual regions are more predictable than others.
        - **Scale experiments with more compute:** Use additional computational resources to test broader hyperparameter ranges and larger component settings.
        """
    )

    st.success(
        """
        Final takeaway: CLIP embeddings provide a useful semantic representation of images, but direct 
        fMRI response prediction remains difficult. PLS improved the prediction problem in latent space, 
        while reconstruction back to voxel-level brain activity remains the main challenge for future work.
        """
    )
