import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = (299, 299)
CLASSES  = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_INFO = {
    "Glioma": {
        "color":       "#ff4d6d",
        "icon":        "🔴",
        "severity":    "High",
        "description": "Gliomas arise from glial cells in the brain or spine. They are the most common primary brain tumors and can range from slow-growing (low-grade) to aggressive (high-grade). Treatment typically involves surgery, radiation, and chemotherapy.",
    },
    "Meningioma": {
        "color":       "#ff9f1c",
        "icon":        "🟠",
        "severity":    "Moderate",
        "description": "Meningiomas form in the meninges — the protective layers surrounding the brain and spinal cord. They are usually benign and slow-growing, often managed with monitoring or surgery.",
    },
    "No Tumor": {
        "color":       "#06d6a0",
        "icon":        "🟢",
        "severity":    "None",
        "description": "No tumor was detected in this MRI scan. The brain tissue appears normal. Always consult a qualified medical professional for a confirmed clinical diagnosis.",
    },
    "Pituitary": {
        "color":       "#a78bfa",
        "icon":        "🟣",
        "severity":    "Low–Moderate",
        "description": "Pituitary tumors develop in the pituitary gland at the base of the brain. Most are benign adenomas and can affect hormone production. Treatment varies from medication to surgery.",
    },
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #f1f5f9 !important; }

/* Section labels */
.sec-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.6rem;
}

/* Result chip */
.result-chip {
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.result-chip .r-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1;
}
.result-chip .r-conf {
    font-size: 0.88rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}
.sev-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.18rem 0.55rem;
    border-radius: 99px;
    margin-top: 0.45rem;
}

/* Confidence bars */
.conf-row { margin-bottom: 0.7rem; }
.conf-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.83rem;
    color: #64748b;
    margin-bottom: 0.28rem;
}
.conf-meta .cname { color: #cbd5e1; font-weight: 500; }
.bar-track { background: #1e293b; border-radius: 99px; height: 7px; overflow: hidden; }
.bar-fill   { height: 7px; border-radius: 99px; }

/* Steps */
.step-row {
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
    margin-bottom: 1.2rem;
}
.step-num {
    flex-shrink: 0;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: #1e3a5f;
    border: 1px solid #3b82f680;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    color: #93c5fd;
}
.step-title { font-weight: 600; color: #e2e8f0; font-size: 0.88rem; margin-bottom: 0.15rem; }
.step-desc  { font-size: 0.82rem; color: #64748b; line-height: 1.5; margin: 0; }

/* Tumor class card */
.tc-card {
    background: #0f172a;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
}
.tc-name { font-weight: 700; font-size: 0.88rem; margin-bottom: 0.15rem; }
.tc-desc { font-size: 0.8rem; color: #64748b; line-height: 1.5; }

/* Stat box */
.stat-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-val { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 800; color: #93c5fd; }
.stat-lbl { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

/* Disclaimer */
.disclaimer {
    background: #7f1d1d20;
    border: 1px solid #ef444430;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.81rem;
    color: #fca5a5;
    margin-top: 1.5rem;
    line-height: 1.55;
}

/* Arch block */
.arch-block {
    background: #0f172a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: monospace;
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 2;
}

/* Streamlit overrides */
[data-testid="stFileUploader"] section {
    background: #111827 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 14px !important;
}
[data-testid="stTabs"] button { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; }
[data-testid="stVerticalBlock"] { gap: 0.6rem; }
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_classifier.h5")

model = load_model()

def preprocess(image):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(image):
    preds = model.predict(preprocess(image), verbose=0)[0]
    idx   = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx]), preds

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🧠 Brain Tumor MRI Classifier")
st.caption("Upload a brain MRI scan to classify it as Glioma, Meningioma, Pituitary, or No Tumor.")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  Classifier", "  About & How It Works"])

# ══════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        st.markdown("<br>", unsafe_allow_html=True)
        col_img, col_res = st.columns([1, 1.3], gap="large")

        with col_img:
            st.markdown('<p class="sec-label">MRI Scan</p>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col_res:
            with st.spinner("Analyzing scan..."):
                label, confidence, all_preds = predict(image)

            info  = CLASS_INFO[label]
            color = info["color"]

            st.markdown('<p class="sec-label">Result</p>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-chip" style="background:{color}15; border:1px solid {color}40;">
                <div class="r-label" style="color:{color};">{info['icon']} {label}</div>
                <div class="r-conf">Confidence: <b style="color:{color};">{confidence*100:.1f}%</b></div>
                <div class="sev-badge" style="background:{color}20; color:{color};">
                    Severity: {info['severity']}
                </div>
            </div>
            <p style="font-size:0.87rem; color:#64748b; line-height:1.65; margin-bottom:1.4rem;">
                {info['description']}
            </p>
            """, unsafe_allow_html=True)

            st.markdown('<p class="sec-label">Confidence — All Classes</p>', unsafe_allow_html=True)
            for cls, prob in sorted(zip(CLASSES, all_preds), key=lambda x: -x[1]):
                c    = CLASS_INFO[cls]["color"]
                fill = c if cls == label else "#334155"
                st.markdown(f"""
                <div class="conf-row">
                    <div class="conf-meta">
                        <span class="cname">{CLASS_INFO[cls]['icon']} {cls}</span>
                        <span>{prob*100:.1f}%</span>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{prob*100:.1f}%; background:{fill};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Medical Disclaimer:</b> This tool is for educational and research purposes only.
            It is not a substitute for professional medical diagnosis, advice, or treatment.
            Always consult a qualified healthcare provider.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:3.5rem 1rem; color:#334155;">
            <div style="font-size:2.8rem; margin-bottom:0.8rem;">🩻</div>
            <div style="font-size:1rem; color:#475569;">Upload an MRI scan above to get started</div>
            <div style="font-size:0.82rem; margin-top:0.4rem; color:#334155;">Supports JPG and PNG</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")

    # ── Left column ──
    with col_a:
        st.markdown("#### About This Project")
        st.markdown("""
        <p style="font-size:0.9rem; color:#94a3b8; line-height:1.7;">
            This is a deep learning classifier that detects and categorizes brain tumors from MRI scans.
            It uses the <b style="color:#e2e8f0;">Xception</b> architecture pretrained on ImageNet,
            fine-tuned on thousands of labeled MRI images across four classes using
            <b style="color:#e2e8f0;">transfer learning</b>.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        for col, val, lbl in [(s1, "4", "Classes"), (s2, "Xception", "Model"), (s3, "299×299", "Input")]:
            with col:
                st.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Tumor Classes")
        for cls, info in CLASS_INFO.items():
            st.markdown(f"""
            <div class="tc-card" style="border-left: 3px solid {info['color']};">
                <span style="font-size:1.1rem;">{info['icon']}</span>
                <div>
                    <div class="tc-name" style="color:{info['color']};">{cls}</div>
                    <div class="tc-desc">{info['description']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Right column ──
    with col_b:
        st.markdown("#### How It Works")

        steps = [
            ("Upload",            "You upload a brain MRI scan in JPG or PNG format."),
            ("Preprocess",        "The image is resized to 299×299 pixels and normalized to [0, 1]."),
            ("Feature Extraction","Xception's convolutional layers extract deep visual features from the scan."),
            ("Classification",    "A custom dense head outputs a probability score for each of the 4 classes."),
            ("Result",            "The highest-probability class is shown as the prediction with confidence scores."),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f"""
            <div class="step-row">
                <div class="step-num">{i}</div>
                <div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Model Architecture")
        st.markdown("""
        <div class="arch-block">
            Xception (ImageNet weights)<br>
            &nbsp;&nbsp;↓ Global Max Pooling<br>
            &nbsp;&nbsp;↓ Flatten<br>
            &nbsp;&nbsp;↓ Dropout (0.3)<br>
            &nbsp;&nbsp;↓ Dense (128, ReLU)<br>
            &nbsp;&nbsp;↓ Dropout<br>
            &nbsp;&nbsp;↓ Dense (4, Softmax)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Dataset")
        st.markdown("""
        <p style="font-size:0.88rem; color:#64748b; line-height:1.6;">
            Trained on the
            <a href="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
               style="color:#60a5fa; text-decoration:none;">
               Brain Tumor MRI Dataset
            </a>
            by Masoud Nickparvar on Kaggle — MRI scans across Glioma, Meningioma,
            Pituitary, and No Tumor categories.
        </p>
        """, unsafe_allow_html=True)