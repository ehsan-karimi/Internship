import streamlit as st
import re

# --------- CLEANING FUNCTION ---------
def clean_latex(text):
    text = re.sub(r"\\\((.*?)\\\)", r"`\1`", text)
    text = re.sub(r"\$(.*?)\$", r"`\1`", text)
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\$\$(.*?)\$\$", r"$$\1$$", text, flags=re.DOTALL)
    return text.strip()

# --------- PAPER DICTIONARY (CLEANED INLINE) ---------
papers = {
    "CLIP (Radford et al., 2021)": {
        "title": "CLIP: Contrastive Language-Image Pretraining",
        "authors": "Alec Radford et al. (OpenAI, 2021)",
        "slides": {
            "Motivation & Conceptual Overview": """
- Traditional supervised image classification requires large labeled datasets and struggles to generalize to new tasks.
- CLIP leverages natural language supervision, jointly training an image encoder and a text encoder.
- This approach maps images and texts into a shared embedding space, enabling zero-shot classification.
""",
            "Step-by-Step Explanation": """
1. Collect massive image-caption pairs (e.g., 400M pairs from the internet).
2. Use an image encoder (ResNet or ViT) to map images into embedding vectors.
3. Use a Transformer-based text encoder to map captions/prompts into embedding vectors.
4. Train both encoders jointly with a contrastive loss, pulling matching pairs together and pushing mismatched pairs apart.
5. After training, given a new image and a set of class name prompts, compute embeddings and classify the image by finding the closest text embedding.
""",
            "Image Encoder in CLIP": """
- The image encoder (ResNet or Vision Transformer) transforms images into dense embeddings.
- It learns semantic visual features that can be aligned with text embeddings.
""",
            "Text Encoder in CLIP": """
- A Transformer-based text encoder converts textual prompts into embeddings.
- Prompts typically include class names in contextual templates (e.g., “a photo of a [CLASS]”).
- Text embeddings share the same space as image embeddings for similarity comparison.
""",
            "Contrastive Learning": """
- CLIP uses a contrastive loss (InfoNCE) to align matching image-text pairs and repel mismatched pairs.

- Similarity between image embedding `f_I(i_i)` and text embedding `f_T(t_j)`:

$$
s_{ij} = \\frac{f_I(i_i) \\cdot f_T(t_j)}{\\tau}
$$

- The loss function is:

$$
L = - \\frac{1}{N} \\sum_{k=1}^N \\left(
\\log \\frac{e^{s_{kk}}}{\\sum_j e^{s_{kj}}} +
\\log \\frac{e^{s_{kk}}}{\\sum_i e^{s_{ik}}}
\\right)
$$
""",
            "Zero-Shot Learning": """
- Enables recognition of classes unseen during training by comparing image embeddings with embeddings of class prompts.
- Selects the class whose text embedding is most similar to the image embedding.
- This enables scalable, flexible classification without retraining.
""",
            "Dataset, Results & Insights": """
- Trained on ~400 million image-text pairs from the internet.
- Achieves about 76% zero-shot accuracy on ImageNet, comparable to supervised models.
- Works well on diverse tasks (OCR, geo-localization), but struggles with fine-grained spatial details.
- Training is very computationally intensive.
"""
        },
        "reference": "https://arxiv.org/abs/2103.00020",
        "image": "zero-shot.png"
    },
    "An Individual Identity-Driven Framework for Animal Re-Identification": {
        "title": "An Individual Identity-Driven Framework for Animal Re-Identification",
        "authors": "Yihao Wu, Di Zhao, Jingfeng Zhang, Yun Sing Koh",
        "slides": {
            "Disentangled Feature Representation": """
- Features are separated into identity-related `z_id` and nuisance `z_nuis` factors.
- Orthogonality constraint enforces independence between `z_id` and `z_nuis`.
""",
            "Training Losses": """
- Identity classification loss: cross-entropy on `z_id`.
- Triplet loss to improve discriminability.
- Orthogonality regularization.
""",
            "Prompt Generation & Cross-Modal Alignment": """
- Prompts are automatically generated for animal image embeddings.
- These pseudo-prompts enrich text-image alignment for improved re-ID.
- No need for manual attribute labeling.
""",
            "Dataset, Results & Insights": """
- Dataset: ElephantID and others.
- Achieved state-of-the-art on animal re-ID benchmarks.
- Benefits: avoids manual labeling, more robust identity learning.
- Limitation: requires identity-labeled data during training.
"""
        },
        "reference": "https://arxiv.org/pdf/2410.22927"
    }
    # ⏳ (You can add the other cleaned papers here in the same structure)
}

# --------- STREAMLIT APP ---------
st.set_page_config(layout="wide")
st.title("Vision & Re-Identification Papers Presentation")

paper_choice = st.sidebar.radio("Select a Paper", list(papers.keys()))
paper = papers[paper_choice]

st.header(paper["title"])
st.subheader(f"Authors: {paper['authors']}")
st.markdown("---")

# Render each slide
for slide_title, content in paper["slides"].items():
    st.subheader(slide_title)

    if paper_choice == "CLIP (Radford et al., 2021)" and slide_title == "Zero-Shot Learning":
        st.image(paper["image"], caption="CLIP Zero-Shot Learning Illustration", use_container_width=True)

    # Split and render mixed Markdown + LaTeX
    cleaned = clean_latex(content)
    parts = re.split(r"(\$\$.*?\$\$)", cleaned, flags=re.DOTALL)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("$$") and part.endswith("$$"):
            st.latex(part[2:-2])
        else:
            st.markdown(part)

st.markdown("---")
st.markdown(f"[🔗 Paper link]({paper['reference']})")
