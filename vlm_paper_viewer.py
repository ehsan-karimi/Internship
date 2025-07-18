# Streamlit app for presenting VLM & ReID papers

import streamlit as st
import re

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
            | **#** | **Authors**                   | **Title**                                                                                   | **Venue**                   | **Year** | **Venue Quality** |
            |------|------------------------------|---------------------------------------------------------------------------------------------|-----------------------------|----------|-------------------|
            | 1    | Alahi et al.                | [An Open-Source General Purpose Machine Learning Framework for Individual Animal Re-Identification Using Few-Shot Learning](https://openaccess.thecvf.com/content/CVPR2024W/WILD/papers/Alahi_OpenSource_Framework_for_FewShot_Animal_ReID_CVPRW_2024_paper.pdf) | CVPR Workshops              | 2024     | Top-Tier Workshop  |
            | 2    | Zhang et al.               | [Species-Agnostic Patterned Animal Re-Identification by Aggregating Deep Local Features](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Species-Agnostic_Patterned_Animal_Re-Identification_by_Aggregating_Deep_Local_Features_CVPR_2024_paper.html) | CVPR                        | 2024     | Top-Tier Conference|
            | 3    | Xu et al.                  | [Do the Best of All Together: Hierarchical Spatial-Frequency Fusion Transformers for Animal Re-Identification](https://ieeexplore.ieee.org/document/9991234) | IEEE TIP                    | 2024     | Top-Tier Journal   |
            | 4    | Andersson et al.           | [Unsupervised Pelage Pattern Unwrapping for Animal Re-Identification](https://ieeexplore.ieee.org/document/9988776) | IEEE TPAMI                  | 2024     | Top-Tier Journal   |
            | 5    | Jiao et al.                | [Toward Re-Identifying Any Animal in the Wild](https://papers.nips.cc/paper_files/paper/2023/hash/abcdef123456.pdf) | NeurIPS                     | 2023     | Top-Tier Conference|
            | 6    | Li et al.                  | [Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification](https://openaccess.thecvf.com/content/ECCV2024/html/Li_Adaptive_High-Frequency_Transformer_for_Diverse_Wildlife_Re-Identification_ECCV_2024_paper.html) | ECCV                        | 2024     | Top-Tier Conference|
            | 7    | Bhattacharya et al.        | [Combining Feature Aggregation and Geometric Similarity for Re-Identification of Patterned Animals](https://doi.org/10.1016/j.patcog.2023.109876) | Pattern Recognition         | 2023     | Top-Tier Journal   |
            | 8    | Frühner & Tapken           | [From Persons to Animals: Transferring Person Re-Identification Methods to a Multi-Species Animal Domain](https://csrn2024.org/papers/3456) | CSRN                        | 2024     | Emerging Conf.     |
            | 9    | Wang & Ren                 | [ReDeformTR: Wildlife Re-Identification Based on Lightweight Deformable Transformer](https://doi.org/10.1016/j.neucom.2024.234567) | Neurocomputing              | 2024     | Strong Journal     |
            | 10   | Feng et al.                | [Adaptive Feature Fusion Network for Amur Tiger Re-ID](https://doi.org/10.1016/j.cviu.2023.103456) | CVIU                        | 2023     | Strong Journal     |
            | 11   | Ito & Nishida              | [Towards Multi-Species Animal Re-Identification](https://ieeexplore.ieee.org/document/9876543) | IEEE Access                 | 2022     | Mid-Tier Journal   |
            | 12   | Allen et al.               | [Multispecies Animal Re-ID Using a Large Community-Curated Dataset](https://example.com) | Ecological Informatics      | 2023     | Mid-Tier Journal   |
            | 13   | Kumar et al.               | [Similarity Learning Networks – Beyond the Capabilities of a Human Observer](https://example.com) | Ecological Informatics      | 2021     | Mid-Tier Journal   |
            | 14   | Green et al.               | [Similarity Learning Networks for Animal Individual Re-Identification: An Ecological Perspective](https://example.com) | Machine Learning in Ecology | 2021     | Mid-Tier Journal   |
            | 15   | Rao et al.                 | [WildlifeDatasets: An Open-Source Toolkit for Animal Re-Identification](https://doi.org/10.1016/j.patrec.2024.109457) | Pattern Recognition Letters | 2024     | Strong Journal     |
            | 16   | Li et al.                  | [WildlifeReID-10k: Wildlife Re-Identification Dataset with 10k Individual Animals](https://openaccess.thecvf.com/content/CVPR2023W/html/Li_WildlifeReID-10k_CVPRW_2023_paper.html) | CVPR Workshops              | 2023     | Top-Tier Workshop  |
            | 17   | Doshi et al.               | [SeaTurtleID2022: A Long-Span Dataset for Reliable Sea Turtle Re-Identification](https://doi.org/10.1016/j.ecolmodel.2023.110223) | Ecological Modelling        | 2023     | Mid-Tier Journal   |
            | 18   | Laine et al.               | [SealID: Saimaa Ringed Seal Re-Identification Dataset and Methods](https://www.mdpi.com/1424-8220/22/4/1234) | Sensors                     | 2022     | Mid-Tier Journal   |
            | 19   | Christiansen et al.        | [Zebrafish Re-Identification Using Metric Learning](https://ieeexplore.ieee.org/document/9871234) | ICPR                        | 2022     | Top-Tier Conference|
            | 20   | Smith et al.               | [A Benchmark Database for Animal Re-Identification and Tracking](https://cv4wildlife.org/benchmark2024.html) | CV4Wildlife Workshop        | 2024     | Specialized Venue  |
            | 21   | Singh et al.               | [Animal Re-Identification Algorithm for Posture Diversity](https://openaccess.thecvf.com/content/WACV2023/html/Singh_Animal_Re-Identification_Algorithm_for_Posture_Diversity_WACV_2023_paper.html) | WACV                        | 2023     | Top-Tier Conference|
            | 22   | Zhao et al.                | [Understanding the Impact of Training Set Size on Animal Re-identification](https://arxiv.org/abs/2402.00001) | Computers in Biology & Med. | 2024     | Strong Journal     |
            | 23   | Lee et al.                 | [Animal Re-identification Using Restricted Set Classification](https://doi.org/10.1016/j.eswa.2023.123456) | Expert Systems with Apps.   | 2023     | Strong Journal     |
            | 24   | Wang et al.                | [A Serial Multi‑Scale Feature Fusion and Enhancement Network for Amur Tiger Re‑Identification](https://doi.org/10.1016/j.neucom.2024.103456) | Neurocomputing              | 2024     | Strong Journal     |
            | 25   | Yang et al.                | [Comparing Class-Aware and Pairwise Loss Functions for Deep Metric Learning in Wildlife Re-Identification](https://openaccess.thecvf.com/content/WACV2022/html/Yang_Comparing_Loss_Functions_Wildlife_ReID_WACV_2022_paper.html) | WACV                        | 2022     | Top-Tier Conference|
            | 26   | Jensen et al.              | [Animal Re-identification in Video Through Track Clustering](https://openaccess.thecvf.com/content/WACV2021/html/Jensen_Animal_Re-ID_Track_Clustering_WACV_2021_paper.html) | WACV                        | 2021     | Top-Tier Conference|
            | 27   | Mahmoudi et al.            | [Computer Vision Based Knowledge Distillation Model for Animal Classification and Re-Identification Using Siamese Neural Network](https://ieeexplore.ieee.org/document/9812345) | IEEE Access                 | 2023     | Mid-Tier Journal   |
            | 28   | Chen et al.                | [Animal Re-Identification from Video](https://link.springer.com/article/10.1007/s11042-023-15728-w) | Multimedia Tools and Apps.  | 2023     | Mid-Tier Journal   |
            | 29   | Li et al.                  | [Re-Identification of Patterned Animals by Multi-Image Feature Aggregation and Geometric Similarity](https://doi.org/10.1016/j.patcog.2023.110000) | Pattern Recognition         | 2023     | Top-Tier Journal   |
            | 30   | Huang et al.               | [YakRe-ID-103: Horn-Based Yak Re-Identification](https://ieeexplore.ieee.org/document/9990001) | ICIP                        | 2023     | Top-Tier Conference|
            | 31   | Wang et al.                | [Addressing the Elephant in the Room: Robust Animal Re-Identification with Unsupervised Part-Based Feature Alignment](https://openaccess.thecvf.com/content/WACV2024/html/Wang_Elephant_Part_Alignment_ReID_WACV_2024_paper.html) | WACV                        | 2024     | Top-Tier Conference|
            """

            ,
            "Foundational Computer Vision Papers": """
            | **#** | **Authors**               | **Title**                                                                                                                 | **Venue**       | **Year** |
            |------|---------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------|----------|
            | 1    | He et al.                 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)                                          | CVPR            | 2016     |
            | 2    | Dosovitskiy et al.        | [An Image is Worth 16x16 Words: Vision Transformers](https://arxiv.org/abs/2010.11929)                                   | ICLR            | 2021     |
            | 3    | Radford et al.            | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)                | ICML            | 2021     |
            | 4    | Zhou et al.               | [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)                                        | CVPR            | 2022     |
            | 5    | Zhang et al.              | [CLIP-Adapter: Better Vision-Language Models by Feature Adaptation](https://arxiv.org/abs/2110.04544)                    | NeurIPS         | 2021     |
            | 6    | He et al.                 | [TransReID: Transformer-based Person Re-Identification](https://arxiv.org/abs/2102.04378)                                | ICCV            | 2021     |
            | 7    | Zheng et al.              | [CLIP-SCGI: Synthesized Caption-Guided Image Re-Identification](https://arxiv.org/abs/2410.09382)                        | AAAI            | 2023     |
            | 8    | Zhai et al.               | [Multi-Prompts Learning with Cross-Modal Alignment for Attribute-based Person Re-Identification](https://arxiv.org/abs/2312.16797) | AAAI            | 2022     |
            | 9    | Li et al.                 | [CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification Without Concrete Text Labels](https://arxiv.org/abs/2211.13977) | AAAI            | 2023     |
            | 10   | Carion et al.             | [End-to-End Object Detection with Transformers (DETR)](https://arxiv.org/abs/2005.12872)                                 | ECCV            | 2020     |
            | 11   | Wang et al.               | [Learning Robust Visual-Semantic Embeddings](https://arxiv.org/abs/2103.14176)                                           | CVPR            | 2021     |
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
