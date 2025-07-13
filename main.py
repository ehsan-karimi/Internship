import streamlit as st

# Paper data: title, authors, slides content, reference link
papers = {
    "CLIP (Radford et al., 2021)": {
        "title": "CLIP: Contrastive Language-Image Pretraining",
        "authors": "Alec Radford et al. (OpenAI, 2021)",
        "slides": {
            "Slide 1: Motivation & Conceptual Overview": """
- Traditional supervised image classification requires massive labeled datasets and generalizes poorly to new tasks. CLIP aims to learn universal visual representations by leveraging natural language supervision.
- Jointly train an image encoder and text encoder to map images and their corresponding captions into a shared embedding space. Enables zero-shot learning on a variety of vision tasks.
            """,
            "Slide 2: Mathematical Framework & Method": """
- Encoders:
  - Image encoder fI(⋅) (ResNet/ViT) outputs embeddings v∈R^d.
  - Text encoder fT(⋅) (Transformer) outputs embeddings t∈R^d.
- Similarity function:
  s_ij = (fI(i_i) ⋅ fT(t_j)) / τ
  where τ is a learnable temperature scalar.
- Contrastive Loss (InfoNCE):
  L = - (1/N) ∑_{k=1}^N [ log( exp(s_kk) / ∑_j exp(s_kj) ) + log( exp(s_kk) / ∑_i exp(s_ik) ) ]
  where N is batch size.
            """,
            "Slide 3: Datasets, Results & Insights": """
- Datasets:
  - Trained on ~400 million image-text pairs scraped from the internet.
  - Evaluated zero-shot on ImageNet and 30+ classification benchmarks.
- Results:
  - Zero-shot ImageNet accuracy ~76%, comparable with supervised models.
  - Strong generalization on diverse datasets including OCR and geo-localization.
- Insights & Limitations:
  - Limited fine-grained spatial reasoning due to global embeddings.
  - Very resource-intensive training process.
            """
        },
        "reference": "https://arxiv.org/abs/2103.00020"
    },

    "ViT (Dosovitskiy et al., 2020)": {
        "title": "An Image is Worth 16x16 Words: Vision Transformer",
        "authors": "Alexey Dosovitskiy et al. (Google Research, 2020)",
        "slides": {
            "Slide 1: Motivation & Conceptual Overview": """
- Transformers have revolutionized NLP; this paper explores their potential for image recognition, questioning whether CNNs’ inductive bias is necessary.
- Treat images as sequences of fixed-size patches (tokens) and apply standard Transformer architecture for classification.
            """,
            "Slide 2: Mathematical Framework & Method": """
- Patch embedding:
  An input image x∈R^{H×W×C} is split into N patches x_p∈R^{P×P×C}, where N=HW/P².
- Each patch is flattened and projected via linear layer into embedding:
  z_0 = [x_{p1} E; x_{p2} E; …; x_{pN} E] + E_{pos}
  where E∈R^{P²C × D} is learnable projection, and E_{pos} is positional encoding.
- Transformer encoder:
  Standard multi-head self-attention blocks process embeddings.
            """,
            "Slide 3: Datasets, Results & Insights": """
- Datasets:
  - Trained on ImageNet-21k or JFT-300M for best results.
  - Fine-tuned and evaluated on ImageNet-1k.
- Results:
  - Outperforms CNN baselines when pretraining data is large enough.
- Insights & Limitations:
  - Requires huge datasets for competitive performance.
  - Computationally expensive for very large models.
            """
        },
        "reference": "https://arxiv.org/abs/2010.11929"
    },

    "ResNet (He et al., 2016)": {
        "title": "Deep Residual Learning for Image Recognition",
        "authors": "Kaiming He et al. (Microsoft Research, 2016)",
        "slides": {
            "Slide 1: Motivation & Conceptual Overview": """
- Training very deep CNNs is challenging due to degradation problems.
- Introduce residual connections to allow training of ultra-deep networks.
            """,
            "Slide 2: Mathematical Framework & Method": """
- Residual block:
  Instead of directly learning a mapping H(x), learn residual F(x) = H(x) − x.
- Output of block:
  y = F(x) + x
  where x is input, F is residual function.
- Enables training of networks with >100 layers.
            """,
            "Slide 3: Datasets, Results & Insights": """
- Datasets:
  - Trained and evaluated on ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012.
- Results:
  - ResNet-152 achieves top-5 error of 3.57%, winning ILSVRC 2015.
  - Significantly outperforms previous architectures.
- Insights & Limitations:
  - Residual connections solve degradation, enabling very deep nets.
  - Became backbone for many vision tasks including re-ID.
  - Larger models are computationally expensive.
            """
        },
        "reference": "https://arxiv.org/abs/1512.03385"
    },

    "CoOp (Zhou et al., 2022)": {
        "title": "Learning to Prompt for Vision-Language Models",
        "authors": "Zhou et al.",
        "slides": {
            "Slide 1: Motivation & Conceptual Overview": """
- CLIP’s zero-shot performance depends heavily on manually designed text prompts (e.g., “a photo of a [CLASS]”). This limits flexibility and requires domain expertise.
- Automatically learn continuous prompt vectors optimized for downstream tasks while keeping CLIP’s backbone frozen.
            """,
            "Slide 2: Mathematical Framework & Method": """
- Prompt representation:
  Represent prompts as continuous vectors:
  P = [v_1, v_2, ..., v_m]
  where each v_i ∈ R^d is a learnable token embedding (e.g., m=16 tokens).
- Input to text encoder:
  The prompt plus class name tokens form the input sequence:
  X = [P, embedding(class name)]
- Optimization:
  Only the prompt vectors P are updated via backpropagation to minimize classification loss, while CLIP encoders remain frozen.
- Loss:
  Standard cross-entropy loss on the classification head using similarity scores from CLIP.
            """,
            "Slide 3: Datasets, Results & Insights": """
- Datasets:
  Tested on 11 image classification datasets (e.g., Caltech101, OxfordPets, Food101).
- Results:
  - CoOp significantly improves few-shot classification accuracy over manual prompts.
  - Adapts better to domain-specific or fine-grained classes.
- Insights & Limitations:
  - Efficient prompt tuning reduces need for full model fine-tuning.
  - Performance depends on number of prompt tokens and training samples.
            """
        },
        "reference": "https://arxiv.org/abs/2201.09743"
    },

    "CLIP-Adapter (Zhou et al., 2022)": {
        "title": "Parameter-Efficient Transfer Learning for Vision-Language Models",
        "authors": "Zhou et al.",
        "slides": {
            "Slide 1: Motivation & Conceptual Overview": """
- Fine-tuning large vision-language models like CLIP is costly in computation and prone to overfitting on small datasets.
- Introduce lightweight adapter modules into CLIP encoders to fine-tune only a small subset of parameters.
            """,
            "Slide 2: Mathematical Framework & Method": """
- Adapter design:
  Each adapter is a two-layer MLP with a bottleneck:
  Adapter(x) = W_2 σ(W_1 x)
  where W_1 ∈ R^{r×d}, W_2 ∈ R^{d×r}, r ≪ d, and σ is activation (ReLU).
- Integration:
  Adapter output added to transformer block output via residual connection:
  y = TransformerBlock(x) + Adapter(x)
- Training:
  Freeze original CLIP parameters; only adapter weights W_1, W_2 trained with task loss (e.g., classification).
            """,
            "Slide 3: Datasets, Results & Insights": """
- Datasets:
  Evaluated on 11 image classification datasets (same as CoOp).
- Results:
  - Achieves comparable accuracy to full fine-tuning with less than 1% of parameters updated.
  - Improves training efficiency and reduces risk of overfitting.
- Insights & Limitations:
  - Parameter-efficient, ideal for resource-constrained environments.
  - May have limited adaptation capacity on very different domains.
            """
        },
        "reference": "https://arxiv.org/abs/2201.03546"
    },

    "TransReID (He et al., 2021)": {
        "title": "TransReID: Transformer-Based Object Re-Identification",
        "authors": "He et al.",
        "slides": {
            "Slide 1: Motivation & Conceptual Overview": """
- Traditional CNN-based re-ID models lack global context modeling which is critical for distinguishing similar identities.
- Leverage Vision Transformer (ViT) architecture to capture global and part-level features for person re-ID.
            """,
            "Slide 2: Mathematical Framework & Method": """
- Architecture:
  - Image divided into patches and fed into ViT backbone to obtain patch embeddings z.
  - Introduce part tokens p representing body parts; learn interactions between z and p via cross-attention.
- Losses:
  - ID classification loss with cross-entropy.
  - Triplet loss for embedding discrimination:
    L_triplet = max{0, d(a,p) − d(a,n) + α}
    where d(·) is distance, a anchor, p positive, n negative, and α margin.
- Training:
  End-to-end optimization of transformer and part tokens.
            """,
            "Slide 3: Datasets, Results & Insights": """
- Datasets:
  - Market1501: 32,668 images, 1,501 identities.
  - DukeMTMC-reID: 36,411 images, 1,812 identities.
- Results:
  - Outperforms CNN baselines by 2-3% mAP and Rank-1 accuracy.
  - Effectively captures global context and discriminative parts.
- Insights & Limitations:
  - Demonstrates the power of transformers for fine-grained re-ID tasks.
  - Transformer models are heavier and require more compute.
            """
        },
        "reference": "https://arxiv.org/abs/2101.02370"
    },

    "CLIP-SCGI (Han et al., 2024)": {
        "title": "CLIP-SCGI: Synthesized Caption-Guided Inversion for Person Re-Identification",
        "authors": "Han et al., 2024",
        "slides": {
            "Slide 1: Motivation & Conceptual Overview": """
- Re-ID models struggle with lack of rich text annotations, limiting CLIP’s cross-modal learning potential.
- Generate pseudo-captions (learnable tokens) from images to guide cross-modal inversion and enhance alignment between image and text embeddings.
            """,
            "Slide 2: Mathematical Framework & Method": """
- Pseudo-Caption Tokens:
  Learn a set of tokens {v_i} initialized randomly and optimized to minimize:
  L = || fI(i) − fT({v_i}) ||²
  aligning image embedding fI(i) and text embedding of tokens fT(·).
- Training:
  Backpropagate through both encoders to update pseudo-caption tokens without requiring manual captions.
- Objective:
  Enhance feature consistency and semantic alignment for re-ID.
            """,
            "Slide 3: Datasets, Results & Insights": """
- Datasets:
  - CUHK03: 14,097 images, 1,467 identities.
  - Market1501: 32,668 images, 1,501 identities.
- Results:
  - +4-6% improvement in mAP and Rank-1 accuracy over baseline CLIP models.
- Insights & Limitations:
  - Removes dependency on human caption annotations.
  - Slight increase in training complexity due to token optimization.
            """
        },
        "reference": "https://arxiv.org/abs/2401.XXXX"  # Placeholder, update with actual if known
    },

    "Multi-Prompts Cross-Modal Alignment (Zhai et al., 2024)": {
        "title": "Multi-Prompts Learning with Cross-Modal Alignment for Attribute-Based Person Re-Identification",
        "authors": "Yajing Zhai et al., 2024",
        "slides": {
            "Slide 1: Motivation & Conceptual Overview": """
- Single prompt embeddings limit expressiveness and diversity of attribute descriptions in re-ID.
- Learn multiple continuous prompt vectors for each attribute, aligned across image and text modalities, to enrich semantic representation.
            """,
            "Slide 2: Mathematical Framework & Method": """
- Multi-Prompt Learning:
  For each attribute a, learn multiple prompt vectors P_a = {p_1, p_2, ..., p_K} where each p_k ∈ R^d.
- Cross-Modal Alignment Loss:
  Enforce consistency between image embeddings fI(i) and aggregated text embeddings fT(P_a) via contrastive or triplet loss.
- Optimization:
  Jointly update prompts and encoders to maximize alignment and discriminative power.
            """,
            "Slide 3: Datasets, Results & Insights": """
- Datasets:
  - Attribute-based re-ID datasets (e.g., PETA, RAP) with multiple attribute labels.
- Results:
  - Significantly improves attribute recognition and overall re-ID performance.
- Insights & Limitations:
  - Enhances fine-grained semantic understanding.
  - More parameters and complexity due to multi-prompt learning.
            """
        },
        "reference": "https://arxiv.org/abs/2402.XXXX"  # Placeholder
    },

    "CLIP-ReID (Li et al., AAAI 2023)": {
        "title": "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels",
        "authors": "Siyuan Li, Li Sun, Qingli Li, 2023",
        "slides": {
            "Slide 1: Motivation & Conceptual Overview": """
- Supervised re-ID requires costly labeling; leveraging CLIP’s vision-language knowledge without explicit text labels is challenging.
- Use unsupervised clustering and contrastive learning on CLIP embeddings to learn discriminative re-ID features without text annotations.
            """,
            "Slide 2: Mathematical Framework & Method": """
- Unsupervised Clustering:
  Extract CLIP image embeddings fI(i) and cluster into pseudo-classes using algorithms like DBSCAN or K-means.
- Contrastive Learning:
  Learn representations using contrastive loss based on pseudo-labels:
  L = - log ( exp(sim(x, x⁺)/τ) / ∑_{x⁻} exp(sim(x, x⁻)/τ) )
  where x⁺ is positive sample in same cluster, x⁻ negatives, τ temperature.
- Training:
  Iterative clustering and feature learning improves re-ID accuracy without manual labels.
            """,
            "Slide 3: Datasets, Results & Insights": """
- Datasets:
  - MSMT17: 126,441 images, 4,101 identities.
  - DukeMTMC-reID: 36,411 images, 1,812 identities.
- Results:
  - Outperforms many supervised baselines in mAP and Rank-1 accuracy.
- Insights & Limitations:
  - Enables label-free re-ID leveraging VLM knowledge.
  - Depends on clustering quality; may struggle with noisy pseudo-labels.
            """
        },
        "reference": "https://arxiv.org/abs/2301.XXXX"  # Placeholder
    },

    "Individual Identity-Driven Framework for Animal Re-Identification": {
        "title": "An Individual Identity-Driven Framework for Animal Re-Identification",
        "authors": "Yihao Wu, Di Zhao* , Jingfeng Zhang, Yun Sing Koh",
        "slides": {
            "Slide 1: Motivation & Conceptual Overview": """
- Animal re-ID faces challenges due to lack of large annotated datasets and large intra-species variation. Human re-ID methods do not directly apply.
- Incorporate individual identity-driven learning to disentangle identity-related features from other visual variations (pose, background).
            """,
            "Slide 2: Mathematical Framework & Method": """
- Feature Disentanglement:
  Learn separate embeddings for identity z_id and nuisance factors z_nuis.
- Loss functions:
  - Identity classification loss on z_id:
    L_id = - ∑_{i=1}^C y_i log ŷ_i
  - Orthogonality constraint to minimize correlation between z_id and z_nuis.
- Training:
  Joint optimization with triplet loss and classification to improve feature discriminability.
            """,
            "Slide 3: Datasets, Results & Insights": """
- Datasets:
  - ElephantID: Large-scale dataset with labeled elephant individuals.
  - Other animal re-ID datasets with varied species and environments.
- Results:
  - State-of-the-art performance on multiple animal re-ID benchmarks.
  - Significant gains in mAP and Rank-1 accuracy compared to baseline CNN or Transformer models.
- Insights & Limitations:
  - Effective for ecological monitoring and conservation efforts.
  - Requires identity-labeled data, which can be scarce for some species.
            """
        },
        "reference": "https://arxiv.org/pdf/2410.22927"
    },
}


def main():
    st.title("Vision & Re-Identification Papers Presentation")

    # Custom paper order
    custom_order = [
        "CLIP (Radford et al., 2021)",
        "ResNet (He et al., 2016)",
        "ViT (Dosovitskiy et al., 2020)",
        "Individual Identity-Driven Framework for Animal Re-Identification",
        "CLIP-ReID (Li et al., AAAI 2023)",
        "CoOp (Zhou et al., 2022)",
        "CLIP-Adapter (Zhou et al., 2022)",
        "TransReID (He et al., 2021)",
        "CLIP-SCGI (Han et al., 2024)",
        "Multi-Prompts Cross-Modal Alignment (Zhai et al., 2024)",
    ]

    paper_choice = st.sidebar.radio(
        "Select a Paper to View",
        custom_order
    )

    paper = papers[paper_choice]

    st.header(paper["title"])
    st.subheader(f"Authors: {paper['authors']}")
    st.markdown("---")

    for slide_title, content in paper["slides"].items():
        st.subheader(slide_title)

        # Custom LaTeX rendering for known formulas
        if "Mathematical Framework" in slide_title:
            lines = content.strip().splitlines()
            for line in lines:
                if "s_ij" in line:
                    st.latex(r"s_{ij} = \frac{f_I(i_i) \cdot f_T(t_j)}{\tau}")
                elif "L = - (1/N)" in line:
                    st.latex(r"L = - \frac{1}{N} \sum_{k=1}^{N} \left[ \log \left( \frac{\exp(s_{kk})}{\sum_j \exp(s_{kj})} \right) + \log \left( \frac{\exp(s_{kk})}{\sum_i \exp(s_{ik})} \right) \right]")
                elif "z_0 =" in line:
                    st.latex(r"z_0 = [x_{p1}E; x_{p2}E; \dots; x_{pN}E] + E_{pos}")
                elif "y = F(x" in line:
                    st.latex(r"y = F(x) + x")
                elif "Adapter(x)" in line:
                    st.latex(r"\text{Adapter}(x) = W_2 \cdot \sigma(W_1 x)")
                elif "L_triplet" in line:
                    st.latex(r"L_{\text{triplet}} = \max\left(0, d(a, p) - d(a, n) + \alpha\right)")
                elif "L = ||" in line:
                    st.latex(r"L = \left\| f_I(i) - f_T(\{v_i\}) \right\|^2")
                elif "L = - log" in line:
                    st.latex(r"L = -\log \left( \frac{\exp(\text{sim}(x, x^+)/\tau)}{\sum_{x^-} \exp(\text{sim}(x, x^-)/\tau)} \right)")
                elif "L_id = -" in line:
                    st.latex(r"L_{\text{id}} = - \sum_{i=1}^{C} y_i \log \hat{y}_i")
                else:
                    st.markdown(line)
        else:
            st.markdown(content.strip())

        st.markdown("---")

    if paper["reference"]:
        st.markdown(f"[📖 Read full paper here]({paper['reference']})")



if __name__ == "__main__":
    main()
