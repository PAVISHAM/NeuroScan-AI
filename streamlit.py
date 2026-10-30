import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_REPO_ID = "pavisha-neuroscan/brain-tumor-classifier"
MODEL_FILENAME = "brain_tumor_vgg16_model.keras"

ALL_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

IMG_SIZE = (128, 128)

# Glioma will not be displayed in the final application
EXCLUDED_CLASS = "glioma"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .hero {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #1a2980 0%,
            #26d0ce 100%
        );
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.1rem;
        margin-bottom: 0.25rem;
    }

    .hero p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 14px;
        background-color: #161b22;
        border: 1px solid #2a2f3a;
        margin-bottom: 1rem;
    }

    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.6rem;
    }

    .prob-label {
        width: 130px;
        font-weight: 600;
        color: #e6edf3;
        text-transform: capitalize;
    }

    .prob-bar-bg {
        flex: 1;
        background-color: #21262d;
        border-radius: 8px;
        height: 14px;
        overflow: hidden;
        margin-right: 10px;
    }

    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
    }

    .prob-value {
        width: 55px;
        text-align: right;
        color: #c9d1d9;
        font-size: 0.9rem;
    }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }

    .footer-note {
        text-align: center;
        color: #8b949e;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #2a2f3a;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(
            135deg,
            #1a2980 0%,
            #26d0ce 100%
        );
        border: none;
        color: white;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        opacity: 0.85;
        color: white;
        border: none;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CLASS COLORS
# ============================================================

CLASS_COLORS = {
    "meningioma": "#f2994a",
    "pituitary": "#9b51e0",
    "notumor": "#27ae60",
    "glioma": "#e74c3c",
}


# ============================================================
# CLASS INFORMATION
# ============================================================

CLASS_INFO = {

    "meningioma":
        "Meningioma detected — a tumor arising from the membranes "
        "(meninges) surrounding the brain and spinal cord. "
        "Usually slow-growing and often benign.",

    "notumor":
        "No tumor detected. The scan does not show patterns "
        "associated with the tumor types this model was trained "
        "to recognize.",

    "pituitary":
        "Pituitary tumor detected — arises in the pituitary gland "
        "at the base of the brain and can affect hormone regulation.",

    "glioma":
        "Glioma detected — a tumor arising from glial cells in "
        "the brain or spinal cord."
}


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🧠 NeuroScan AI</h1>
        <p>
            Deep learning-based classification of brain MRI scans —
            Meningioma, Pituitary, or No Tumor.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING FROM HUGGING FACE
# ============================================================

@st.cache_resource(show_spinner="Loading VGG16 model...")
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    model = tf.keras.models.load_model(model_path)

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()
    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error("❌ Failed to load VGG16 model")

    st.code(str(e))

    st.info(
        "Please check the Hugging Face model repository, "
        "model filename, and TensorFlow compatibility."
    )

    st.stop()


# ============================================================
# CHECK MODEL OUTPUT
# ============================================================

try:

    output_shape = model.output_shape
    output_units = output_shape[-1]

except Exception:

    output_units = None


if output_units is not None:

    if output_units != len(ALL_CLASSES):

        st.error(
            f"""
            ❌ Model output mismatch.

            Model output classes: {output_units}

            Expected classes: {len(ALL_CLASSES)}

            Current class order: {ALL_CLASSES}
            """
        )

        st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ About This Project")

    st.write(
        "NeuroScan AI is a deep learning system that classifies "
        "brain MRI scans using a VGG16-based convolutional neural "
        "network."
    )

    st.write(
        "Upload an MRI scan and the model predicts the tumor "
        "category along with its confidence."
    )

    st.markdown("**Tech Stack**")

    st.markdown(
        """
        - TensorFlow / Keras
        - VGG16
        - Streamlit
        - Pillow
        - NumPy
        - Hugging Face
        """
    )

    st.markdown("---")

    st.markdown("**Model Information**")

    st.write("VGG16 Keras Model")

    st.markdown("---")

    if model_loaded:

        st.success("✅ VGG16 Model Loaded")

    st.caption("Hugging Face Model Repository")

    st.code(
        f"{MODEL_REPO_ID}/{MODEL_FILENAME}",
        language="text"
    )


# ============================================================
# CLASS SETTINGS
# ============================================================

all_classes = ALL_CLASSES

if EXCLUDED_CLASS in all_classes:

    excluded_idx = all_classes.index(EXCLUDED_CLASS)

else:

    excluded_idx = None


display_classes = [
    c for c in all_classes
    if c != EXCLUDED_CLASS
]


# ============================================================
# MAIN COLUMNS
# ============================================================

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT COLUMN - IMAGE UPLOAD
# ============================================================

with left_col:

    st.subheader("📤 Upload MRI Scan")

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:

            image = Image.open(uploaded_file)

            image = image.convert("RGB")

            st.image(
                image,
                caption="Uploaded MRI Scan",
                use_container_width=True
            )

            classify_clicked = st.button(
                "🔍 Classify Scan",
                use_container_width=True,
                type="primary"
            )

        except Exception as e:

            st.error(f"Error opening image: {e}")

            classify_clicked = False

    else:

        classify_clicked = False

        st.info("Upload an MRI scan to begin.")


# ============================================================
# RIGHT COLUMN - RESULTS
# ============================================================

with right_col:

    st.subheader("📊 Results")

    if uploaded_file is not None and classify_clicked:

        with st.spinner("Analyzing MRI image..."):

            try:

                # --------------------------------------------
                # IMAGE PREPROCESSING
                # --------------------------------------------

                img_resized = image.resize(IMG_SIZE)

                img_array = np.array(img_resized)

                img_normalized = (
                    img_array.astype("float32") / 255.0
                )

                img_batch = np.expand_dims(
                    img_normalized,
                    axis=0
                )


                # --------------------------------------------
                # MODEL PREDICTION
                # --------------------------------------------

                raw_predictions = model.predict(
                    img_batch,
                    verbose=0
                )[0]


                # --------------------------------------------
                # REMOVE GLIOMA
                # --------------------------------------------

                masked = raw_predictions.copy()

                if excluded_idx is not None:

                    masked[excluded_idx] = 0.0


                # --------------------------------------------
                # NORMALIZE PROBABILITIES
                # --------------------------------------------

                total = masked.sum()

                if total > 0:

                    probs = masked / total

                else:

                    st.error("Unable to calculate prediction.")

                    st.stop()


                # --------------------------------------------
                # FIND PREDICTED CLASS
                # --------------------------------------------

                pred_idx = int(np.argmax(probs))

                pred_class = all_classes[pred_idx]

                confidence = probs[pred_idx] * 100


            except Exception as e:

                st.error(f"❌ Prediction failed: {e}")

                st.stop()


        # ====================================================
        # RESULT CARD
        # ====================================================

        color = CLASS_COLORS.get(
            pred_class,
            "#58a6ff"
        )

        description = CLASS_INFO.get(
            pred_class,
            ""
        )

        st.markdown(
            f"""
            <div class="result-card">

                <span
                    class="badge"
                    style="
                        background-color:{color}22;
                        color:{color};
                    "
                >
                    {pred_class.upper()}
                </span>

                <div
                    style="
                        font-size:1.05rem;
                        color:#e6edf3;
                        margin-bottom:0.75rem;
                    "
                >
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.metric(
            label="🎯 Confidence",
            value=f"{confidence:.2f}%"
        )


        # ====================================================
        # PROBABILITY DISTRIBUTION
        # ====================================================

        st.markdown("##### Probability Distribution")

        prob_html = ""

        for name in display_classes:

            idx = all_classes.index(name)

            prob = probs[idx] * 100

            c = CLASS_COLORS.get(
                name,
                "#58a6ff"
            )

            prob_html += f"""
            <div class="prob-row">

                <div class="prob-label">
                    {name}
                </div>

                <div class="prob-bar-bg">

                    <div
                        class="prob-bar-fill"
                        style="
                            width:{prob:.2f}%;
                            background-color:{c};
                        "
                    >
                    </div>

                </div>

                <div class="prob-value">
                    {prob:.1f}%
                </div>

            </div>
            """


        st.markdown(
            prob_html,
            unsafe_allow_html=True
        )


    elif uploaded_file is not None:

        st.write(
            "Click **🔍 Classify Scan** to run the model."
        )


    else:

        st.write(
            "Results will appear here once a scan is analyzed."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">

        NeuroScan AI is a demonstration of applied
        deep learning and is not intended for
        clinical or diagnostic use.

    </div>
    """,
    unsafe_allow_html=True
)
