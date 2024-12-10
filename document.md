
<h1> AI for healthcare: Summary</h1>


Artificial Intelligence (AI) is no longer a futuristic aspiration in the field of healthcare—it is a transformative force actively shaping the landscape of medicine, diagnostics, and patient care. This document,  "Application of AI for Healthcare " offers an advanced exploration of the cutting-edge intersections between AI and medicine, catering to professionals and researchers deeply involved in AI development and application.  

The book is structured around ten pivotal topics, each reflecting a critical domain where AI is driving innovation and solving complex healthcare challenges. By bridging theoretical frameworks with practical implementations, this book not only explores the technical intricacies of AI methodologies but also emphasizes their clinical relevance and ethical considerations.  

Each chapter includes a summary of a landmark scientific paper, carefully selected to demonstrate the state-of-the-art in that specific domain. These case studies provide a rigorous foundation for understanding how AI algorithms transition from theoretical constructs to practical tools, impacting patient outcomes and medical processes.  

This document aims to be a valuable resource for AI specialists, healthcare researchers, and technology developers, fostering deeper insights into how AI solutions are conceived, evaluated, and integrated into one of humanity’s most critical domains—healthcare.                                                                                                                      
<h2> Deep Learning in Medicine </h2>

<h2>Natural Language Processing (NLP) in Healthcare </h2>

The technology for decoding thoughts into text based on brainwaves represents a crucial research area that can transform human linguistic expressions into digital forms. This technology is particularly beneficial for individuals with neurological disorders and shows potential for expanding human-machine interaction (HMI). Generally, EEG-to-Text translation decodes sentences read, spoken, or thought by humans using brainwave inputs.
EEG-to-Text [3] is one of the pioneering studies that performed EEG-to-Text translation using a language model. This paper utilizes a pre-trained natural language model (BART) to convert EEG signals into text, enabling the use of an open vocabulary. DeWave [4] addresses the issue of individual differences in brainwave characteristics using discrete codex. Studies have shown that brainwaves exhibit variability even when reading the same sentences across individuals. To enhance decoding accuracy and reduce individual differences, DeWave transforms continuous EEG signals into discrete codex. It employs VQ-VAE for codex conversion and combines it with BART to learn natural language generation through alignment between EEG and text.
NeuSpeech [5] adopts a speech-based pre-trained model. Since brainwaves are wave-like data, it uses wave2vec, a language model pre-trained on audio data, instead of a natural language-based model. To address the differences in channels between audio and EEG, NeuSpeech incorporates CNN layers.
These advancements aim to overcome the existing limitations in EEG-to-Text technology. However, current challenges include the scarcity of data and the fact that most datasets are limited to reading stimuli. Future research is expected to expand this technology by incorporating more diverse data sources and practical application scenarios.

<h2> AI in Diagnostics and Imaging</h2> 

* Glaucoma is the disease that affects the human eye and results in a permanent blindness if not detect at early stage. The Situation is very complex, so the proper
detection is must. Detection at early stages may be improved else it may lead toloss
of vision. The main reason for the vision loss is affected optic nerve of the eye. The doctor needs at least 5 check-up reports to confirm the glaucoma a ffection, soit
becomes essential to design a system to detect this disease accurately in one attempt. This paper presents architecture for the proper glaucoma detection based ontheconvolutional neural network (CNN). It will make differentiation between the patterns
formed for glaucoma and non-glaucoma. CNN used for achieving the adequateperformance in the glaucoma detection. Overall Architecture is having six layers for
proper detection of the disease. A dropout mechanism is also applied to improve theperformance of the given approach. SCES(has 46 images for glaucoma and 1676images for the normal fundus.) and ORIGA(having 168 glaucoma and 482 images of
the normal fundus) datasets used for the experiments and obtained values are .822and .882 for the ORIGA and SCES respectively. The major objective is to findthemost similar patterns in between the normal human eye and the infected glaucomaeye. The presented approach works in six layers. The four layers are the convolutional
layers, and the last two layers are fully connected. Presented CNN to get the input, theROI of the image is used instead of full image for the Faster processing comparedtofull disc or cup images and Improved detection speed for glaucoma. Also used ARGA LI Approach (Adaptive Region Growing and Level Set Integration)
for the removal of bright fringe to improve image quality also helps determine thecenter and radius for trimming as well as fixed to 256 x 256 pixels for consistency. Dropout Implemented in two fully connected layers. To prevent overfitting, dataaugmentation generates translations and horizontal reflections of images. The areaunder the curve () AUC is utilised of the receiver operating characteristics (ROC)
curve for the evaluation of glaucoma detecting performance of the model. For validating the proposed CNN-based approach for glaucoma detection, it was
compared with state-of-the-art reconstruction-based methods. The model was trainedon 90 images from a 600-image dataset, and the remaining images were usedfor
testing. The proposed method achieved an AUC of 0.822 for the ORIGAdataset and0.882 for the SCES dataset, The state-of-the-art method, achieved an AUCs of 0.809and 0.859, respectively. The detection capability of the proposed systemseems higher
than other methods and effective for glaucoma detection [1]

* Recent advancements in medical imaging have made anomaly detection a critical area of research due to its ability to identify diseases by detecting deviations from normal anatomical structures. One of the latest innovations in this field is the use of Diffusion Models, particularly Denoising Diffusion Implicit Models (DDIMs), for medical anomaly detection. DDIMs stand out for their ability to function with only image-level annotations, avoiding the need for pixel-level annotations while maintaining high precision and simplifying the training process.
The core methodology of this approach involves translating diseased images into healthy ones. This process unfolds in two main steps. First, images undergo iterative noising to encode anatomical information, followed by a denoising process guided by a classifier to reconstruct the image in a healthy state. Second, the anomaly map is generated by calculating the pixel-wise difference between the original diseased image and the reconstructed healthy image. This ensures precise localization of anomalies while preserving non-pathological regions of the image, leading to high anomaly detection accuracy.
The DDIM-based approach has been tested on datasets such as BRATS2020 (for brain tumor segmentation) and CheXpert (for detecting pleural effusions in chest X-rays). In these applications, DDIM outperformed traditional methods like Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs). It demonstrated superior capabilities in detecting anomalies with high precision while maintaining the anatomical integrity of healthy regions.
One of the standout features of this method is its use of weak supervision, requiring only image-level labels, which significantly reduces the complexity and cost associated with data annotation. Furthermore, the use of classifier-guided gradient-based adjustments during the denoising process ensures that only anomalous regions are modified while reconstructing healthy counterparts, enabling more precise anomaly mapping.
Despite its strengths, DDIM has two primary limitations. First, its performance is sensitive to hyperparameters such as noise levels and gradient scaling, which require careful tuning. Second, the computational cost is relatively high due to the iterative nature of the process, though optimization strategies like reducing the number of steps have been proposed to improve efficiency. Future research directions include integrating more complex datasets, enhancing real-time diagnostic capabilities, and combining diffusion models with other AI modalities, such as transformers, to further improve anomaly detection performance.
In conclusion, diffusion models represent a promising leap forward in medical imaging diagnostics, offering precise and interpretable results while reducing data requirements. DDIM bridges the gap between advanced AI methodologies and practical clinical applications, providing a robust tool for detecting and mapping medical anomalies with exceptional accuracy and reliability. [6]

* Neural Attenuation Fields for Sparse-View CBCT Reconstruction" introduces a groundbreaking self-supervised method to address the challenges of sparse-view CBCT, which is critical for reducing radiation dose during medical imaging. Sparse-view CBCT reconstruction is inherently ill-posed, leading to artifacts and inaccuracies when using traditional methods. To overcome these challenges, the proposed Neural Attenuation Fields (NAF) framework parameterizes the 3D attenuation field using a neural network, leveraging a novel hash encoding technique that efficiently captures high-frequency details while maintaining computational efficiency. Unlike existing methods that rely on extensive external datasets or computationally intensive iterative processes, NAF synthesizes projections through an implicit neural representation of the attenuation process, optimizing the network by minimizing the difference between real and predicted projections. This approach achieves state-of-the-art reconstruction accuracy and significantly reduces computational time, demonstrating its potential for fast, accurate, and low-dose CBCT applications in clinical settings.  

* In many medical image analysis applications, only a limited amount of training data is available due to the costs of image acquisition and the large manual annotation effort required from experts. Training recent state-of-the-art machine learning methods like convolutional neural networks (CNNs) from small datasets is a challenging task. In this work on anatomical landmark localization, they propose a CNN architecture that learns to split the localization task into two simpler sub-problems, reducing the overall need for large training datasets. Fully convolutional SpatialConfiguration-Net (SCN) learns this simplification due to multiplying the heatmap predictions of its two components and by training the network in an end-to-end manner. Thus, the SCN dedicates one component to locally accurate but ambiguous candidate predictions, while the other component improves robustness to ambiguities by incorporating the spatial configuration of landmarks. In extensive experimental evaluation, it show that the proposed SCN outperforms related methods in terms of landmark localization error on a variety of size-limited 2D and 3D landmark localization datasets, i.e., hand radiographs, lateral cephalograms, hand MRIs, and spine CTs.
* AI is widely used in healthcare for tasks like disease detection and anomaly identification. In this context, Shvetsova et al. (2021) proposed a novel approach for anomaly detection in medical imaging using Deep Perceptual Autoencoders (DPA) Shvetsova et al., 2021. Unlike traditional autoencoders that rely on pixel-level differences, DPA employs perceptual loss to capture content-level differences, making it more effective at detecting subtle abnormalities in complex medical images. The authors also introduce a progressive training strategy where the model gradually increases image resolution and complexity, enhancing its ability to handle high-dimensional medical data. To address the challenge of hyperparameter tuning, a small set of labeled abnormal samples is used, reflecting real-world clinical scenarios. Experimental results on the Camelyon16 (pathology) and NIH (chest X-ray) datasets demonstrate significant improvements over state-of-the-art methods, achieving ROC AUCs of 93.4% and 92.6%, respectively. This study establishes a new baseline for anomaly detection in medical imaging, offering a practical, efficient, and reproducible framework for clinical applications.

<h2>AI in Drug Discovery and Development </h2> 

<h2> Robotics and AI in Surgery </h2>
Medical image segmentation is a critical foundation for the advancement of healthcare systems, playing an essential role in accurately diagnosing diseases and planning treatment strategies. Among segmentation models, the U-Net architecture, a U-shaped convolutional neural network (CNN), has become a widely accepted standard due to its success in various medical image segmentation tasks. U-Net’s encoderdecoder structure allows it to capture both global and local features, but it often struggles with modeling longrange dependencies due to the intrinsic locality of its convolution operations, which primarily focus on neighboring pixels. This limitation can restrict U-Net’s ability to recognize broader contextual information, which is often crucial for precise segmentation in
complex medical images. Transformers, originally designed for tasks involving sequence-to-sequence
prediction, have gained popularity as potential alternatives in segmentation tasks, as their selfattention mechanism captures global dependencies
across the input. However, the Transformer architecture tends to lose some fine-grained
localization, as its strength in capturing long-range dependencies comes at the expense of some low-level
detail accuracy. In this work, introduce TransUNet, a hybrid model that merges the strengths of both Transformers and U-Net, creating a powerful solution
for medical image segmentation. TransUNet leverages Transformers to process tokenized image patches
derived from CNN feature maps, effectively capturing global context and long-range dependencies.
Simultaneously, it incorporates U-Net’s decoder to upsample the encoded features and combine them
with high-resolution CNN feature maps, thereby restoring precise localization and low-level details. This
combination allows TransUNet to capitalize on the Transformer’s global feature extraction while
preserving U-Net’s ability to enhance spatial detail and segmentation accuracy. TransUNet consistently
outperforms other methods across various medical segmentation applications, including multi-organ
segmentation and cardiac segmentation, highlighting its effectiveness in improving segmentation outcomes through its innovative, dual-component design [2]

<h2>AI in Patient Monitoring and Management </h2>

**AI-Enabled Remote Patient Monitoring (RPM)**

AI in remote patient monitoring (RPM) is revolutionizing healthcare by enhancing patient care, increasing efficiency, and enabling early intervention.
Talati et al., 2023 [7] highlight how AI is transforming healthcare through remote patient monitoring. The research emphasizes on early intervention, the effectiveness of wearable devices, and AI algorithms to analyze large volumes of patient data. 
AI-driven RPM enhances healthcare by enabling early intervention, reducing hospitalizations, and offering personalized care.
Wearable devices and non-invasive monitoring techniques play a crucial role in AI-powered RPM systems.
AI algorithms analyze large volumes of patient data to identify patterns, abnormalities, and potential health issues.

Furthermore, the research published in [8] outlines several applications of AI in RPM. 
AI-powered remote patient monitoring systems are revolutionizing healthcare by enabling real-time tracking of vital signs and health indicators. These systems use wearable devices and sensors to continuously collect data on parameters such as heart rate, blood pressure, and oxygen saturation. AI algorithms analyze this data in real-time, allowing for immediate detection of concerning changes in a patient's condition.

Predictive analytics leverages AI to analyze historical patient data, identify patterns, and forecast potential health issues before they occur. This enables healthcare providers to implement proactive interventions, potentially preventing adverse outcomes. AI-driven decision support systems assist in recommending appropriate diagnostic tests and treatment adjustments based on individual patient data and the latest medical evidence. Automated patient triage uses AI to assess patient data and prioritize care for high-risk individuals, ensuring efficient resource allocation. Finally, AI enables the creation of personalized care plans tailored to each patient's unique health profile, preferences, and treatment responses, leading to more effective and patient-centered care


<h2>Ethical Considerations and Bias in AI </h2>

<h2>AI and Healthcare Policy </h2>

<h2>Case Studies and Real-World Applications </h2>

<h2> Future Directions and Course Wrap-Up </h2>

<h2> References</h2>
[1] Arkaja Saxena, Abhilasha Vyas, Lokesh Parashar, Upendra Singh, A Glaucoma Detection using Convolution Neural Network 

[2] TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation

[3] Wang, Z., & Ji, H. (2022). Open vocabulary electroencephalography-to-text decoding and zero-shot sentiment classification. Proceedings of the AAAI Conference on Artificial Intelligence, 36(5), 5350–5358.

[4] Duan, Y., Zhou, C., Wang, Z., Wang, Y.-K., & Lin, C.-T. (2023). DeWave: Discrete encoding of EEG waves for EEG to text translation. Thirty-seventh Conference on Neural Information Processing Systems. Retrieved from https://openreview.net/forum?id=WaLI8slhLw

[5] Yang, Y., Duan, Y., Zhang, Q., Jo, H., Zhou, J., Lee, W. H., Xu, R., & Xiong, H. (2024). NeuSpeech: Decode neural signal as speech. arXiv. https://arxiv.org/abs/2403.01748

[6] Wolleb, Julia, et al. "Diffusion models for medical anomaly detection." International Conference on Medical image computing and computer-assisted intervention. Cham: Springer Nature Switzerland, 2022.

[7] Talati, D. (2023). Telemedicine and AI in Remote Patient Monitoring. Journal of Knowledge Learning and Science Technology ISSN: 2959-6386 (online), 2(3), 254-255. https://doi.org/10.60087/jklst.vol2.n3.p255

[8] Tsvetanov, F. Integrating AI Technologies into Remote Monitoring Patient Systems. Eng. Proc. 2024, 70, 54. https://doi.org/10.3390/engproc2024070054


