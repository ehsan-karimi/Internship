# Streamlit app for presenting VLM & ReID papers
import base64
import os

import streamlit as st
import re

from PIL import Image

# Paper data: title, authors, slides content, reference link
papers = {
    "Internship Roadmap": {
        "title": "Internship Roadmap and Schedule",
        "authors": "Me",
        "slides": {
            "Aim Overview": """
1. Understanding CLIP and Vision-Language Models  
2. Study VLM-based re-identification approaches  
3. Build a survey on animal re-identification (2020-2025)
""",
            "Understanding CLIP and Vision-Language Models": """
- Study the original CLIP paper: Radford et al., Learning transferable visual models from natural language supervision, ICML 2021.  
- Focus on:  
  * Architecture (image and text encoders, contrastive loss)  
  * Zero-shot capabilities  
  * Pre-training dataset and method  
- Explore related papers citing CLIP:  
  https://scholar.google.com/scholar?cites=15031020161691567042&as_sdt=2005&sciodt=0,5&hl=en
""",
            "Study VLM-based Re-identification Approaches": """
- Focus on CLIP applications for image re-identification (re-ID).  
- Understand classical CLIP limitations for re-ID.  
- Key papers:  
  1. https://arxiv.org/pdf/2410.22927  
  2. Siyuan Li et al., *CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels*, AAAI 2023.  
- Review citations and follow-up research.
""",
            "Build a Survey on Animal Re-identification (2020-2025)": """
- Use systematic literature search with keywords:  
  (“animal” AND “re-identification”) OR (“animal” AND “reidentification”)  
- Search tools:  
  * Scopus: https://www.elsevier.com/products/scopus/search  
  * Semantic Scholar: https://www.semanticscholar.org/  
  * Connected Papers: https://www.connectedpapers.com/search?q=animal%20re-identification  
- Organize survey by methodologies, e.g., leveraging vision-language models, others.  
- Suggested reference survey paper on human re-ID: https://arxiv.org/pdf/2001.04193
""", "Papers Diagram": """

"""
        },
        "reference": ""
    },
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
            "Contrastive Learning": r"""
- CLIP uses a contrastive loss (InfoNCE) to align matching image-text pairs and repel mismatched pairs.

- Similarity between image embedding \( f_I(i_i) \) and text embedding \( f_T(t_j) \):

  \[
  s_{ij} = \frac{f_I(i_i) \cdot f_T(t_j)}{\tau}
  \]

  where \(\tau\) is a learnable temperature parameter.

- The loss function is:

  \[
  L = - \frac{1}{N} \sum_{k=1}^N \left(
  \log \frac{e^{s_{kk}}}{\sum_j e^{s_{kj}}} +
  \log \frac{e^{s_{kk}}}{\sum_i e^{s_{ik}}}
  \right)
  \]

- This encourages the model to pull corresponding pairs closer and push others apart.
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

    "ResNet (He et al., 2016)": {
        "title": "Deep Residual Learning for Image Recognition",
        "authors": "Kaiming He et al. (Microsoft Research, 2016)",
        "slides": {
            "Motivation & Conceptual Overview": """
- Very deep CNNs are difficult to train due to vanishing gradients and degradation.
- Simply stacking layers often led to worse performance.
- ResNet proposes residual learning to address this.
""",
            "Step-by-Step Explanation": """
Imagine you input an image of a cat:

1. The image is passed through initial convolutional layers that extract basic features like edges and textures.
2. The result enters a residual block:
   - The input to the block is saved as a "skip connection".
   - The block computes a residual function \( F(x) \) via conv → BN → ReLU → conv → BN.
3. The skip connection input \( x \) is added back: \( y = F(x) + x \).
4. This process repeats through many residual blocks, allowing the network to learn deeper features without degradation.
5. After many blocks, features go through average pooling and a fully connected layer to classify the image (e.g., "cat").
""",
            "Why Residual Learning?": """
- Instead of learning a direct mapping \(H(x)\), layers learn the residual function \(F(x) = H(x) - x\).
- This makes optimization easier and improves gradient flow.
""",
            "Residual Block Architecture": r"""
- A residual block outputs: 

  \[
  y = F(x) + x
  \]

- Skip connections add the input \(x\) directly to the output of the block.
- This helps train deeper networks by mitigating degradation.
""",
            "Training Deep Networks with Residual Blocks": """
- Enables training of ultra-deep networks (e.g., ResNet-152).
- Uses batch normalization, ReLU, and identity mapping.
- Allows performance to improve with network depth.
""",
            "Dataset, Results & Insights": """
- Trained on ImageNet 2012.
- ResNet-152 achieved 3.57% top-5 error, winning ILSVRC 2015.
- Became a foundational architecture for many tasks.
"""
        },
        "reference": "https://arxiv.org/abs/1512.03385"
    },

    "ViT (Dosovitskiy et al., 2020)": {
        "title": "An Image is Worth 16x16 Words: Vision Transformer",
        "authors": "Alexey Dosovitskiy et al. (Google Research, 2020)",
        "slides": {
            "Motivation & Conceptual Overview": """
- Explores using transformers for image classification.
- Treats image patches like tokens in NLP.
- Challenges need for CNN-specific inductive bias.
""",
            "Step-by-Step Explanation": """
Imagine you input a 224x224 image of a dog:

1. The image is split into 16x16 patches (each patch is like a “word”).
2. Each patch is flattened and projected to a fixed-length embedding vector.
3. Positional encodings are added to keep spatial information.
4. A special [class] token is prepended to the sequence.
5. The sequence is fed through transformer encoder layers (self-attention + MLP).
6. The final [class] token embedding is used to classify the image (e.g., "dog").
""",
            "Patch Embedding": """
- Split image \(x\) into \(N\) patches (e.g., 16x16).
- Flatten each patch and project using a linear layer.
- Resulting embeddings get positional encodings.
""",
            "Transformer Encoder": """
- Standard transformer with multi-head attention and MLP layers.
- A special [class] token gathers image representation.
""",
            "Dataset, Results & Insights": """
- Pretrained on large datasets (ImageNet-21k, JFT-300M).
- Outperforms CNNs when enough data is available.
- Expensive to train and requires high compute.
"""
        },
        "reference": "https://arxiv.org/abs/2010.11929"
    },

    "CLIP-ReID (Siyuan Li et al., 2023)": {
        "title": "CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels",
        "authors": "Siyuan Li et al. (2023)",
        "slides": {
            "Motivation & Conceptual Overview": """
- CLIP was designed for general vision-language tasks.
- For person re-ID, alignment needs are different.
- This paper adapts CLIP for person re-identification.
""",
            "Step-by-Step Example": """
1. Input: image of a person wearing a red coat.
2. Text prompt: “a person with red coat and black shoes”.
3. Encode image and text using CLIP encoders.
4. Identity-aware loss optimizes matching of same person images.
5. Semantic-aware loss aligns attributes (e.g., color, clothing).
6. Combined loss improves re-ID accuracy.
""",
            "Method Overview": """
- Integrates fine-grained prompts and person-centric text.
- Uses a two-branch structure with CLIP encoders.
- Proposes identity-aware and semantic-aware alignment.
""",
            "Alignment Techniques": """
- Identity-aware alignment improves discrimination.
- Semantic-aware alignment aligns visual and text semantics.
- Combines global and local alignment strategies.
""",
            "Results & Insights": """
- Outperforms previous baselines on Market-1501 and MSMT17.
- Demonstrates benefits of leveraging CLIP for fine-grained tasks.
- Shows importance of prompt design and alignment loss.
"""
        },
        "reference": "https://arxiv.org/abs/2211.13977"
    },

    "An Individual Identity-Driven Framework for Animal Re-Identification": {
        "title": "An Individual Identity-Driven Framework for Animal Re-Identification",
        "authors": "Yihao Wu, Di Zhao, Jingfeng Zhang, Yun Sing Koh",
        "slides": {
            "Motivation & Conceptual Overview": """
- Animal re-ID faces challenges like large intra-species variation and limited labeled data.
- Standard human re-ID methods don't transfer well to animals.
- This work proposes learning disentangled identity-specific features.
""",
            "Step-by-Step Explanation with Example": """
1. Input: image of an elephant.
2. Encode into latent features, split into identity-related \( z_{id} \) and nuisance \( z_{nuis} \) factors.
3. Enforce orthogonality between \( z_{id} \) and \( z_{nuis} \) to separate identity from pose/background.
4. Use \( z_{id} \) for identity classification.
5. Generate pseudo-prompt (e.g., “elephant with tusks walking”) automatically from features.
6. Compare embeddings with others using prompt-guided similarity for re-ID.
""",
            "Disentangled Feature Representation": """
- Features are separated into identity-related (\( z_{id} \)) and nuisance (\( z_{nuis} \)) factors.
- Orthogonality constraint enforces independence between \( z_{id} \) and \( z_{nuis} \).
""",
            "Training Losses": """
- Identity classification loss: cross-entropy on \( z_{id} \).
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
    },

    "CoOp (Zhou et al., 2022)": {
        "title": "CoOp: Learning to Prompt for Vision-Language Models",
        "authors": "Zhou et al., 2022",
        "slides": {
            "Motivation & Conceptual Overview": """
- CLIP performs zero-shot classification using hand-crafted prompts like: “a photo of a [CLASS]”.
- These manually written prompts are often **not optimal** — they require human intuition and don’t generalize well across domains (e.g., fine-grained classes, medical images, etc.).
- **CoOp** proposes to **learn the prompts automatically** by treating them as **continuous, learnable vectors**.
- These prompts are optimized using labeled data for a downstream task, while the CLIP model remains frozen.
""",
            "Step-by-Step Workflow (with Simple Example)": """
Consider classifying dog breeds:

1. Task: Classify images of dog breeds.
2. Baseline: Use manual prompts like “a photo of a Labrador”.
3. CoOp learns 16 continuous prompt vectors \( [v_1, v_2, \dots, v_{16}] \) appended to the class name.
4. The image encoder processes the image to an embedding.
5. The text encoder processes the learned prompt + class label into an embedding.
6. Training maximizes similarity between matching image and prompt embeddings.
7. Result: learned prompts adapt to the dataset improving classification without modifying CLIP itself.
""",
            "Prompt Representation & Optimization Strategy": r"""
- CoOp replaces manual text prompts with learnable embeddings:

  \[
  P = [v_1, v_2, \dots, v_m]
  \]

  where each \( v_i \in \mathbb{R}^d \) is a learnable vector.

- Input to the CLIP text encoder is:

  \[
  [v_1, v_2, \dots, v_m, \text{embedding(class name)}]
  \]

- Only the prompt vectors \( P \) are optimized, CLIP’s backbone remains frozen.
""",
            "Dataset, Results & Insights": """
- Evaluated on 11 datasets (OxfordPets, Caltech101, Food101, etc.).
- CoOp outperforms manual prompt CLIP, especially on fine-grained domains.
- Strong few-shot learning performance.
"""
        },
        "reference": "https://arxiv.org/abs/2109.01134"
    },

    "CLIP-Adapter (Zhang et al., 2022)": {
        "title": "CLIP-Adapter: Better Vision-Language Models by Feature Adaptation",
        "authors": "Zhang et al., 2022",
        "slides": {
            "Motivation & Conceptual Overview": """
- CLIP is powerful but fixed after pre-training.
- Fine-tuning the entire CLIP is costly and may cause overfitting.
- CLIP-Adapter adds lightweight modules to adapt CLIP features to downstream tasks.
""",
            "Step-by-Step Explanation": """
1. Start with frozen CLIP encoders.
2. Insert small adapter layers after image and text encoders.
3. During training on target task, only adapter layers are updated.
4. The adapter layers adjust features to improve task-specific performance.
5. Outputs from adapters and original CLIP features are combined via residual connections.
""",
            "Adapter Architecture": """
- Adapter is a bottleneck module with two linear layers and a non-linearity.
- Parameters are much fewer than full CLIP fine-tuning.
- Allows faster adaptation with fewer data.
""",
            "Training & Results": """
- Trained on 11 image recognition datasets.
- Outperforms zero-shot CLIP and full fine-tuning in many cases.
- Efficient and effective for domain adaptation.
""",
            "Dataset, Results & Insights": """
- Significant gains with minimal additional parameters.
- Flexible for many vision-language tasks.
"""
        },
        "reference": "https://arxiv.org/abs/2110.04544"
    },

    "TransReID (He et al., 2021)": {
        "title": "TransReID: Transformer-based Person Re-Identification",
        "authors": "He et al., 2021",
        "slides": {
            "Motivation & Conceptual Overview": """
- Classical person re-ID uses CNNs but struggles to capture long-range dependencies.
- Transformers excel in modeling global context.
- TransReID applies Vision Transformer (ViT) to person re-ID.
""",
            "Step-by-Step Explanation": """
1. Input image is split into patches.
2. Patch embeddings are created and fed into transformer layers.
3. Use identity tokens and class tokens for identity classification.
4. Incorporate camera and pose information for better features.
5. Train with classification and triplet losses.
""",
            "Transformer Encoder in ReID": """
- Self-attention models long-range relations.
- Helps handle occlusion and pose variations.
""",
            "Training Losses": """
- Cross-entropy loss for identity classification.
- Triplet loss to separate identities in feature space.
""",
            "Dataset, Results & Insights": """
- Tested on Market-1501, DukeMTMC-reID.
- Outperforms many CNN-based baselines.
- Shows transformers are promising for re-ID.
"""
        },
        "reference": "https://arxiv.org/abs/2102.04378"
    },

    "CLIP-SCGI (Zheng et al., 2023)": {
        "title": "CLIP-SCGI: Synthesized Caption-Guided Image Re-Identification",
        "authors": "Zheng et al., 2023",
        "slides": {
            "Motivation & Conceptual Overview": """
- Text-guided re-ID methods require manual attribute annotations.
- CLIP-SCGI synthesizes captions automatically to guide re-ID.
- Leverages pseudo-caption learned from images.
""",
            "Step-by-Step Explanation": """
1. An image is passed to a caption generator that produces a full descriptive sentence.

2. That caption is then encoded using the CLIP text encoder.

3. The image goes through the CLIP image encoder.

4. The model then uses cross-modal alignment to bring the image and its pseudo-caption closer together in the shared embedding space.

5. Two losses are used:

    Contrastive loss: ensures matching image-caption pairs stay close, mismatches are pushed apart.

    Classification loss: improves person identification across images.
""",
            "Cross-Modal Alignment": """
- Align image and pseudo-caption embeddings.
- Use contrastive and classification losses.
""",
            "Dataset, Results & Insights": """
- Evaluated on multiple person re-ID datasets.
- Outperforms baselines without manual text.
- Shows potential for automatic text guidance.
"""
        },
        "reference": "https://arxiv.org/abs/2410.09382"
    },

    "Multi-Prompts Learning with Cross-Modal Alignment (Zhai et al., 2022)": {
        "title": "Multi-Prompts Learning with Cross-Modal Alignment for Attribute-based Person Re-Identification",
        "authors": "Zhai et al., 2022",
        "slides": {
            "Motivation & Conceptual Overview": """
- Fine-grained attribute-based re-ID requires rich textual descriptions.
- Hand-crafted prompts limit flexibility.
- Multi-prompts learning generates diverse prompts and aligns with images.
""",
            "Step-by-Step Explanation": """
1. Define multiple learnable prompt vectors.
2. Encode multiple text prompts for each image.
3. Fuse multiple prompt embeddings with image embedding.
4. Train with cross-modal contrastive losses.
5. Improves attribute-based re-ID robustness and accuracy.
""",
            "Prompt Generation": """
- Prompts capture different attribute combinations.
- Helps model focus on multiple aspects of person appearance.
""",
            "Cross-Modal Alignment": """
- Uses contrastive loss to align image and multi-prompt embeddings.
""",
            "Results & Insights": """
- Improves attribute-based re-ID on Market-1501 and DukeMTMC.
- Demonstrates benefits of diverse prompt learning.
"""
        },
        "reference": "https://arxiv.org/abs/2312.16797"
    },

    "Loss Functions Overview": {
        "title": "Common Loss Functions in VLM and Re-Identification Models",
        "authors": "Compiled Summary",
        "slides": {
            "Contrastive Loss (InfoNCE)": r"""
    - Aligns matching image-text pairs, repels mismatched pairs.
    - Drives the learning of joint embeddings in CLIP-like models.

    \[
    L_{\text{contrastive}} = - \frac{1}{N} \sum_{k=1}^N \left(
    \log \frac{e^{s_{kk} / \tau}}{\sum_j e^{s_{kj} / \tau}} +
    \log \frac{e^{s_{kk} / \tau}}{\sum_i e^{s_{ik} / \tau}}
    \right)
    \]

    Where \(s_{ij}\) is the similarity between image \(i\) and text \(j\), and \(\tau\) is a learnable temperature.
    """,

            "Cross-Entropy Loss (CE)": r"""
    - Standard classification loss used in most models.
    - Encourages the model to assign high probability to the correct class.

    \[
    L_{\text{CE}} = -\sum_{c=1}^C y_c \log(p_c)
    \]

    Where \(y_c\) is the true label and \(p_c\) is the predicted probability for class \(c\).
    """,

            "Triplet Loss": r"""
    - Learns to bring similar pairs closer and push dissimilar ones apart.
    - Popular in Re-ID tasks to enforce embedding separation.

    \[
    L_{\text{triplet}} = \max\left(0, d(A, P) - d(A, N) + \alpha \right)
    \]

    Where:
    - \(A\): anchor, \(P\): positive, \(N\): negative sample
    - \(d\): distance function (e.g., Euclidean or cosine)
    - \(\alpha\): margin
    """,

            "Orthogonality Loss": r"""
    - Ensures disentanglement by penalizing correlation between identity and nuisance features.

    \[
    L_{\text{orth}} = \left\| z_{id}^\top z_{nuis} \right\|^2
    \]

    - \(z_{id}\): identity-specific features  
    - \(z_{nuis}\): pose/background/etc.
    """,

            "Identity-Aware Loss": """
    - Encourages images of the same person to be embedded closely.
    - Often paired with triplet or contrastive loss.
    - Targets instance-level matching in person re-ID.

    Formula not fixed — typically integrated into contrastive or CE loss with identity supervision.
    """,

            "Semantic-Aware Loss": """
    - Aligns specific visual attributes (like "red shirt") with text features.
    - Useful in models like CLIP-ReID for fine-grained alignment.

    Often implemented using contrastive or alignment loss between local visual and text embeddings.
    """,

            "Alignment Loss": """
    - General term for losses used to align two modalities (image and text).
    - Could be cosine similarity, contrastive, or MSE between image-text embeddings.
    - Used in CLIP-Adapter, CLIP-ReID, and SCGI.

    No universal formula, depends on model. Common structure:
    \[
    L_{\text{align}} = \text{Dist}(f_I(x), f_T(t))
    \]
    Where \(\text{Dist}\) is usually cosine distance or MSE.
    """
        },
        "reference": ""
    },
    "Paper Comparison Table": {
        "title": "Comparison of Vision-Language and Re-Identification Papers",
        "authors": "Compiled Overview",
        "slides": {
            "Summary Table": """
| **Paper & Authors**                         | **Main Idea**                                                                                      | **Key Techniques**                                                                   | **Loss Functions**                                      | **Dataset / Task**                        | **Key Result / Insight**                                            |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------|--------------------------------------------|---------------------------------------------------------------------|
| **CLIP** (Radford et al., 2021)            | Learn joint image-text embeddings from large-scale image-caption pairs for zero-shot classification | Contrastive learning with ResNet/ViT + Transformer text encoder                       | Contrastive loss (InfoNCE)                              | 400M image-text pairs, ImageNet zero-shot | Strong zero-shot accuracy (~76%), scalable, flexible               |
| **CoOp** (Zhou et al., 2022)               | Learn continuous prompt vectors instead of manual prompts for downstream tasks                      | Learnable prompt vectors with frozen CLIP encoders                                    | Contrastive loss (from CLIP)                            | 11 fine-grained datasets                   | Better adaptation than manual prompts                              |
| **CLIP-Adapter** (Zhang et al., 2022)      | Add lightweight adapters to CLIP for efficient task-specific adaptation                             | Adapter MLPs + residual connections after encoders                                     | Cross-Entropy + Alignment loss                          | 11 image classification datasets           | Efficient adaptation, better than full fine-tuning or zero-shot    |
| **CLIP-ReID** (Li et al., 2023)            | Adapt CLIP to person re-ID by aligning both identity and semantics                                  | Two-branch architecture, identity-aware + semantic-aware alignment                    | Identity-aware + Semantic-aware + Contrastive           | Market-1501, MSMT17                        | Better person re-ID via fine-grained alignment                     |
| **Multi-Prompts** (Zhai et al., 2022)      | Train multiple prompts per attribute and align across modalities                                    | Multi-learnable prompts, cross-modal contrastive learning                             | CE + Contrastive + Attribute alignment                  | Market-1501, DukeMTMC-reID                 | More robust attribute-level person re-ID                           |
| **CLIP-SCGI** (Zheng et al., 2023)         | Use pseudo-captions to guide re-ID without human-written text                                       | Caption generator + CLIP encoders + cross-modal alignment                             | Contrastive + Classification (CE)                       | Person re-ID datasets                      | Improves re-ID without any manual caption annotation               |
| **Animal Re-ID** (Wu et al., 2024)         | Disentangle identity from nuisance variation in animal images                                       | Feature disentanglement + orthogonality + pseudo-prompt guidance                      | CE + Triplet + Orthogonality                            | ElephantID, other species datasets         | State-of-the-art animal re-ID, no manual label needed              |
| **TransReID** (He et al., 2021)            | Apply transformer architecture (ViT) to model person identity features                              | ViT + camera/pose tokens + triplet & ID losses                                        | CE + Triplet                                            | Market-1501, DukeMTMC-reID                 | Transformers outperform CNNs on person re-ID                       |
| **ResNet** (He et al., 2016)               | Enable deep CNNs by introducing residual learning                                                   | Residual blocks with skip connections                                                  | Cross-Entropy                                           | ImageNet 2012                             | Solves vanishing gradients, won ILSVRC 2015                        |
| **ViT** (Dosovitskiy et al., 2020)         | Use transformers for image classification without CNN biases                                       | Patch embeddings + position encoding + transformer encoder                             | Cross-Entropy                                           | ImageNet-21k, JFT-300M                     | Beats CNNs with enough data, needs massive compute                 |
"""
        },
        "reference": ""
    }, "Survey Paper Tables": {
        "title": "Core and Foundational Papers for Animal Re-ID Survey",
        "authors": "Me",
        "slides": {
            "Core Animal Re-ID Papers (2020–2025)": """
| # | Authors | Title | Document Type | Source Title | Year | Link |
|---|---------|-------|--------------|-------------|------|------|
| 1 | Kuncheva L.I.; Williams F.; Hennessey S.L.; Rodriguez J.J. | A Benchmark Database for Animal Re-Identification and Tracking | Conference Paper | 5th IEEE International Image Processing, Applications and Systems Conference, IPAS 2022 | 2022 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85149777181&doi=10.1109%2fIPAS55744.2022.10052988&partnerID=40&md5=12b8a2b3513f65d5805b50c368e0b64f |
| 2 | Matlala B.; van der Haar D.; Vandapalli H. | A Novel Approach To Lion Re-Identification Using Vision Transformers | Conference Paper | Communications in Computer and Information Science | 2025 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85211787248&doi=10.1007%2f978-3-031-78255-8_16&partnerID=40&md5=b7efcefa9402596e4e0ec4005e28ee12 |
| 3 | Xu N.; Ma Z.; Xia Y.; Dong Y.; Zi J.; Xu D.; Xu F.; Su X.; Zhang H.; Chen F. | A Serial Multi-Scale Feature Fusion and Enhancement Network for Amur Tiger Re-Identification | Journal Article | Animals | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85190358968&doi=10.3390%2fani14071106&partnerID=40&md5=527bb613b3c606d9da5f634a099ffe90 |
| 4 | Li S.; Li J.; Tang H.; Qian R.; Lin W. | ATRW: A Benchmark for Amur Tiger Re-identification in the Wild | Conference Paper | MM 2020 - Proceedings of the 28th ACM International Conference on Multimedia | 2020 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85106335431&doi=10.1145%2f3394171.3413569&partnerID=40&md5=a392b09212d53b86188ab4480029c12b |
| 5 | Kuncheva L.I.; Garrido-Labrador J.L.; Ramos-Pérez I.; Hennessey S.L.; Rodríguez J.J. | An experiment on animal re-identification from video | Journal Article | Ecological Informatics | 2023 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85146608563&doi=10.1016%2fj.ecoinf.2023.101994&partnerID=40&md5=3d99f26e72ad7e807c31be75c8f4b82f |
| 6 | Wahltinez O.; Wahltinez S.J. | An open-source general purpose machine learning framework for individual animal re-identification using few-shot learning | Journal Article | Methods in Ecology and Evolution | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85181236065&doi=10.1111%2f2041-210X.14278&partnerID=40&md5=171375c5e3598157c7d0a2b968df8146 |
| 7 | He Z.; Qian J.; Yan D.; Wang C.; Xin Y. | Animal Re-Identification Algorithm for Posture Diversity | Conference Paper | ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing - Proceedings | 2023 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85177580270&doi=10.1109%2fICASSP49357.2023.10094783&partnerID=40&md5=795b6aeed3635fe9a39ff14d396d55bf |
| 8 | Williams F.J.; Hennessey S.L.; Kuncheva L.I. | Animal re-identification in video through track clustering | Journal Article | Pattern Analysis and Applications | 2025 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-105008725843&doi=10.1007%2fs10044-025-01497-8&partnerID=40&md5=24a47aae6997bb3a219b906023f9a165 |
| 9 | Kuncheva L. | Animal reidentification using restricted set classification | Journal Article | Ecological Informatics | 2021 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85103025472&doi=10.1016%2fj.ecoinf.2021.101225&partnerID=40&md5=8ec7eb5947b66c24150f77bc4b7267ac |
| 10 | Naik H.; Yang J.; Das D.; Crofoot M.C.; Rathore A.; Sridhar V.H. | BuckTales: A multi-UAV dataset for multi-object tracking and re-identification of wild antelopes | Conference Paper | Advances in Neural Information Processing Systems | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-105000468744&partnerID=40&md5=3f8aae46d852c260239d0aff79c20859 |
| 11 | Dlamini N.; van Zyl T.L. | Comparing class-aware and pairwise loss functions for deep metric learning in wildlife re-identification† | Journal Article | Sensors | 2021 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85114631429&doi=10.3390%2fs21186109&partnerID=40&md5=6dc8350f2ab08d10e20e6c5a38f8ffb5 |
| 12 | Ashok Kumar L.; Karthika Renuka D.; Saravana Kumar S. | Computer vision based knowledge distillation model for animal classification and re-identification using Siamese Neural Network | Journal Article | Journal of Intelligent and Fuzzy Systems | 2023 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85161231331&doi=10.3233%2fJIFS-222672&partnerID=40&md5=3f90aaeb36fdf377ca1b206bbf7fbbe6 |
| 13 | Ravoor P.C.; T.s.b. S. | Deep Learning Methods for Multi-Species Animal Re-identification and Tracking a Survey | Journal Article | Computer Science Review | 2020 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85097489104&doi=10.1016%2fj.cosrev.2020.100289&partnerID=40&md5=4b9bd2b59d1457dd2622c2127de5fa59 |
| 14 | Cheng X.; Zhu J.; Zhang N.; Wang Q.; Zhao Q. | Detection Features as Attention (Defat): A Keypoint-Free Approach to Amur Tiger Re-Identification | Conference Paper | Proceedings - International Conference on Image Processing, ICIP | 2020 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85098635758&doi=10.1109%2fICIP40778.2020.9190667&partnerID=40&md5=c53842f7d71df668e5ae1dae7f02231a |
| 15 | Zheng W.; Wang F.-Y. | Do the best of all together: Hierarchical spatial-frequency fusion transformers for animal re-identification | Journal Article | Information Fusion | 2025 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85200851433&doi=10.1016%2fj.inffus.2024.102612&partnerID=40&md5=797c12709adf565d813b8de0fd4cf517 |
| 16 | Perneel M.; Adriaens I.; Verwaeren J.; Aernouts B. | Dynamic Multi-Behaviour, Orientation-Invariant Re-Identification of Holstein-Friesian Cattle | Journal Article | Sensors | 2025 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-105006750920&doi=10.3390%2fs25102971&partnerID=40&md5=77a297e99d1a901fbb2c57f7df821b5e |
| 17 | Schmidt J.E.; Mäkinen O.E.; Flou Nielsen S.G.; Johansen A.S.; Nasrollahi K.; Moeslund T.B. | Exploring Loss Functions for Optimising the Accuracy of Siamese Neural Networks in Re-Identification Applications | Conference Paper | Proceedings of SPIE - The International Society for Optical Engineering | 2022 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85134331033&doi=10.1117%2f12.2622634&partnerID=40&md5=40cf454050633eca758385c1b7ad270b |
| 18 | Zhang J.; Zhao Q.; Liu T. | Feature-Aware Noise Contrastive Learning for Unsupervised Red Panda Re-Identification | Conference Paper | Proceedings of the International Joint Conference on Neural Networks | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85204999301&doi=10.1109%2fIJCNN60899.2024.10651256&partnerID=40&md5=657f44d0027e3aa96cbd64edd70f6639 |
| 19 | Fruhner M.; Tapken H. | From Persons to Animals: Transferring Person Re-Identification Methods to a Multi-Species Animal Domain | Conference Paper | ACM International Conference Proceeding Series | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85201311241&doi=10.1145%2f3665026.3665032&partnerID=40&md5=aa0af1e10990e764e3cf89752d73f376 |
| 20 | Chen X.; Yang T.; Mai K.; Liu C.; Xiong J.; Kuang Y.; Gao Y. | Holstein Cattle Face Re-Identification Unifying Global and Part Feature Deep Network with Attention Mechanism | Journal Article | Animals | 2022 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85128454276&doi=10.3390%2fani12081047&partnerID=40&md5=4659fc5b6bcc14334b6044da0756063d |
| 21 | Yu P.; Burghardt T.; Dowsey A.W.; Campbell N.W. | Holstein-Friesian re-identification using multiple cameras and self-supervision on a working farm | Journal Article | Computers and Electronics in Agriculture | 2025 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-105007902406&doi=10.1016%2fj.compag.2025.110568&partnerID=40&md5=aca0e250fcfdd681526824c28f00d9e2 |
| 22 | Chan J.; Carrión H.; Mégret R.; Agosto Rivera J.L.; Giray T. | Honeybee Re-identification in Video: New Datasets and Impact of Self-supervision | Conference Paper | Proceedings of the International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications | 2022 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85141473898&doi=10.5220%2f0010843100003124&partnerID=40&md5=693e68e6b4163b2d794c773a9c9f2caa |
| 23 | Moskvyak O.; Maire F.; Dayoub F.; Baktashmotlagh M. | Learning Landmark Guided Embeddings for Animal Re-identification | Conference Paper | Proceedings - 2020 IEEE Winter Conference on Applications of Computer Vision Workshops, WACVW 2020 | 2020 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85085951090&doi=10.1109%2fWACVW50321.2020.9096932&partnerID=40&md5=05ed719553c162aa99618cd30b9273a9 |
| 24 | Nepovinnykh E.; Eerola T.; Kalviainen H.; Chelak I. | NORPPA: NOvel Ringed Seal Re-Identification by Pelage Pattern Aggregation | Conference Paper | Proceedings - 2024 IEEE Winter Conference on Applications of Computer Vision Workshops, WACVW 2024 | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85191691004&doi=10.1109%2fWACVW60836.2024.00008&partnerID=40&md5=5b93a8fc2d2880a4365f270074f0e2b8 |
| 25 | Zuerl M.; Dirauf R.; Koeferl F.; Steinlein N.; Sueskind J.; Zanca D.; Brehm I.; Fersen L.V.; Eskofier B. | PolarBearVidID: A Video-Based Re-Identification Benchmark Dataset for Polar Bears | Journal Article | Animals | 2023 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85149733738&doi=10.3390%2fani13050801&partnerID=40&md5=e87b7db22d2932943069dec51f3f8a83 |
| 26 | Odo A.; McLaughlin N.; Kyriazakis I. | Re-identification for long-term tracking and management of health and welfare challenges in pigs | Journal Article | Biosystems Engineering | 2025 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85217087262&doi=10.1016%2fj.biosystemseng.2025.02.001&partnerID=40&md5=efb03ebe0d3ce02193dcb6db3c149c09 |
| 27 | Nepovinnykh E.; Immonen V.; Eerola T.; Stewart C.V.; Kälviäinen H. | Re-identification of patterned animals by multi-image feature aggregation and geometric similarity | Journal Article | IET Computer Vision | 2025 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85214363986&doi=10.1049%2fcvi2.12337&partnerID=40&md5=827b51dfbfd903b45bf6bf22d5d9e24c |
| 28 | Li Z.; Yan Z.; Tian W.; Zeng D.; Liu Y.; Li W. | ReDeformTR: Wildlife Re-Identification Based on Light-Weight Deformable Transformer with Multi-Image Feature Fusion | Journal Article | IEEE Access | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85200223151&doi=10.1109%2fACCESS.2024.3436813&partnerID=40&md5=3dede48049100cb733181f0280f03205 |
| 29 | Moskvyak O.; Maire F.; Dayoub F.; Armstrong A.O.; Baktashmotlagh M. | Robust Re-identification of Manta Rays from Natural Markings by Learning Pose Invariant Embeddings | Conference Paper | DICTA 2021 - 2021 International Conference on Digital Image Computing: Techniques and Applications | 2021 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85124308312&doi=10.1109%2fDICTA52665.2021.9647359&partnerID=40&md5=34809e17d41bbbee479c8483cea7a9fe |
| 30 | Adam L.; Čermák V.; Papafitsoros K.; Picek L. | SeaTurtleID2022: A long-span dataset for reliable sea turtle re-identification | Conference Paper | Proceedings - 2024 IEEE Winter Conference on Applications of Computer Vision, WACV 2024 | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85191982358&doi=10.1109%2fWACV57701.2024.00699&partnerID=40&md5=f070e5bd4409a5fd7889189389cdd408 |
| 31 | Nepovinnykh E.; Eerola T.; Biard V.; Mutka P.; Niemi M.; Kunnasranta M.; Kälviäinen H. | SealID: Saimaa Ringed Seal Re-Identification Dataset | Journal Article | Sensors | 2022 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85139811494&doi=10.3390%2fs22197602&partnerID=40&md5=e02fac59ad39eeb94818997c2abae50c |
| 32 | Moosa M.; Cheikh F.A.; Beghdadi A.; Ullah M. | Self-Supervised Animal Detection, Tracking & Re-Identification | Conference Paper | Proceedings - 13th International Conference on Image Processing Theory, Tools and Applications, IPTA 2024 | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85212590725&doi=10.1109%2fIPTA62886.2024.10755717&partnerID=40&md5=ec623a2041c53ba8fd5c76d40b44b8f6 |
| 33 | Wang Y.; Xu X.; Wang Z.; Li R.; Hua Z.; Song H. | ShuffleNet-Triplet: A lightweight RE-identification network for dairy cows in natural scenes | Journal Article | Computers and Electronics in Agriculture | 2023 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85146056740&doi=10.1016%2fj.compag.2023.107632&partnerID=40&md5=aa4261f7966ba83777859bbbd9f4cdcc |
| 34 | Nepovinnykh E.; Eerola T.; Kalviainen H. | Siamese Network Based Pelage Pattern Matching for Ringed Seal Re-identification | Conference Paper | Proceedings - 2020 IEEE Winter Conference on Applications of Computer Vision Workshops, WACVW 2020 | 2020 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85085920863&doi=10.1109%2fWACVW50321.2020.9096935&partnerID=40&md5=d57f050b3069d4b42e2b8dfd311ead04 |
| 35 | Schneider S.; Taylor G.W.; Kremer S.C. | Similarity Learning Networks for Animal Individual Re-Identification-Beyond the Capabilities of a Human Observer | Conference Paper | Proceedings - 2020 IEEE Winter Conference on Applications of Computer Vision Workshops, WACVW 2020 | 2020 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85081283800&doi=10.1109%2fWACVW50321.2020.9096925&partnerID=40&md5=9de375fa288ac0ad582855ec07836390 |
| 36 | Schneider S.; Taylor G.W.; Kremer S.C. | Similarity learning networks for animal individual re-identification: an ecological perspective | Journal Article | Mammalian Biology | 2022 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85128806984&doi=10.1007%2fs42991-021-00215-1&partnerID=40&md5=f4f6f41321b4bcf05d571fd8cf18cf46 |
| 37 | Nepovinnykh E.; Chelak I.; Eerola T.; Immonen V.; Kälviäinen H.; Kholiavchenko M.; Stewart C.V. | Species-Agnostic Patterned Animal Re-identification by Aggregating Deep Local Features | Journal Article | International Journal of Computer Vision | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85191703797&doi=10.1007%2fs11263-024-02071-1&partnerID=40&md5=0d61e52c529d7c2e5d8d5509747992dd |
| 38 | Jiao B.; Liu L.; Gao L.; Wu R.; Lin G.; Wang P.; Zhang Y. | Toward Re-Identifying Any Animal | Conference Paper | Advances in Neural Information Processing Systems | 2023 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85183295165 |
| 39 | Wang M.; Larsen M.L.V.; Liu D.; Winters J.F.M.; Rault J.-L.; Norton T. | Towards re-identification for long-term tracking of group housed pigs | Journal Article | Biosystems Engineering | 2022 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85136110888&doi=10.1016%2fj.biosystemseng.2022.07.017 |
| 40 | Bai X.; Islam T.; Bin Azhar M.A.H. | Transformer-based Models for Enhanced Amur Tiger Re-Identification | Conference Paper | 2024 IEEE 22nd World Symposium on Applied Machine Intelligence and Informatics, SAMI 2024 - Proceedings | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85186762273&doi=10.1109%2fSAMI60510.2024.10432893 |
| 41 | Lamping C.; Kootstra G.; Derks M. | Transformer-based similarity learning for re-identification of chickens | Journal Article | Smart Agricultural Technology | 2025 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-105002670424&doi=10.1016%2fj.atech.2025.100945 |
| 42 | Guo Q.; Sun Y.; Min L.; Putten A.; Knol E.F.; Visser B.; Rodenburg T.B.; Bolhuis J.E.; Bijma P.; de With P.H.N. | Video-based Detection and Tracking with Improved Re-Identification Association for Pigs and Laying Hens in Farms | Conference Paper | Proceedings of the International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications | 2022 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85137896283&doi=10.5220%2f0010788100003124 |
| 43 | Zheng Z.; Zhao Y.; Li A.; Yu Q. | Wild Terrestrial Animal Re-Identification Based on an Improved Locally Aware Transformer with a Cross-Attention Mechanism | Journal Article | Animals | 2022 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85144674178&doi=10.3390%2fani12243503 |
| 44 | Čermák V.; Picek L.; Adam L.; Papafitsoros K. | WildlifeDatasets: An open-source toolkit for animal re-identification | Conference Paper | Proceedings - 2024 IEEE Winter Conference on Applications of Computer Vision, WACV 2024 | 2024 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85192022557&doi=10.1109%2fWACV57701.2024.00585 |
| 45 | Zhang T.; Zhao Q.; Da C.; Zhou L.; Li L.; Jiancuo S. | YakReID-103: A Benchmark for Yak Re-Identification | Conference Paper | 2021 IEEE International Joint Conference on Biometrics, IJCB 2021 | 2021 | https://www.scopus.com/inward/record.uri?eid=2-s2.0-85113293503&doi=10.1109%2fIJCB52358.2021.9484341 |
| 46 | Borlinghaus P.; Tausch F.; Rettenberger L. | A Purely Visual Re-ID Approach for Bumblebees and its Application to Ecological Monitoring | Journal Article | Smart Agricultural Technology | 2023 | https://www.scopus.com/pages/publications/85141543745 |

            """

        },
        "reference": ""
    },
    "Survey Structure Overview": {
        "title": "A Survey on Animal Re-Identification: Structure Diagram",
        "authors": "Me",
        "slides": {
            "Structure Diagram": """
"""
        },
        "reference": ""
    },
    "Survey Beta": {
        "title": "Just a Beta Version",
        "authors": "Me",
        "slides": {
            "Here we are": """
"""
        },
        "reference": ""
    }

}

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Vision & Re-Identification Papers Presentation")

paper_choice = st.sidebar.radio("Select a Paper", list(papers.keys()))
paper = papers[paper_choice]

st.header(paper["title"])
st.subheader(f"Authors: {paper['authors']}")
st.markdown("---")

# Render slides
from textwrap import dedent


def render_slide(slide_title, content):
    st.subheader(slide_title)

    if paper_choice == "CLIP (Radford et al., 2021)" and slide_title == "Zero-Shot Learning":
        st.image("zero-shot.png", caption="CLIP Zero-Shot Learning Illustration", use_container_width=True)

    if paper_choice == "Internship Roadmap" and slide_title == "Papers Diagram":
        st.image("Internship-Diagram.jpg", caption="Papers Roadmap Diagram", use_container_width=True)

    if paper_choice == "Survey Structure Overview" and slide_title == "Structure Diagram":
        st.image("Structure-Diagram.png", caption="Survey Structure Diagram", use_container_width=True)

    if paper_choice == "Survey Beta" and slide_title == "Here we are":
        # Define the folder and load image filenames
        image_folder = 'Images/'  # or use 'pages/' if in subfolder

        # Load images in order
        image_files = sorted([
            file for file in os.listdir(image_folder)
            if file.endswith((".png", ".jpg", ".jpeg")) and file.startswith("A_Survey")
        ])

        # Show page slider if images exist
        if image_files:
            page_number = st.slider("Select Page", 1, len(image_files), 1)
            image_path = os.path.join(image_folder, image_files[page_number - 1])
            image = Image.open(image_path)
            st.image(image, use_container_width=True)
        else:
            st.warning("No page images found. Make sure your images are named like page1.jpg, page2.jpg, etc.")

    # Extract block LaTeX expressions \[ ... \]
    latex_blocks = re.findall(r"\\\[.*?\\\]", content, re.DOTALL)
    parts = re.split(r"\\\[.*?\\\]", content, flags=re.DOTALL)

    for i, part in enumerate(parts):
        if part.strip():
            # For inline LaTeX, replace \( ... \) with $...$ for inline rendering inside markdown
            part_with_inline_latex = re.sub(r"\\\((.+?)\\\)", r'$\1$', part)
            st.markdown(dedent(part_with_inline_latex).strip())

        # Render block LaTeX centered with st.latex()
        if i < len(latex_blocks):
            raw_latex = latex_blocks[i][2:-2]  # remove \[ and \]
            st.latex(raw_latex)


# Render slides
for slide_title, content in paper["slides"].items():
    render_slide(slide_title, content)

st.markdown("---")
if paper_choice not in ["Internship Roadmap", "Loss Functions Overview", "Paper Comparison Table"] and paper.get(
        "reference"):
    st.markdown(f"[Paper link]({paper['reference']})")
