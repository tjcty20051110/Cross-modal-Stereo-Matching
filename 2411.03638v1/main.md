

## Adaptive Stereo Depth Estimation with Multi-Spectral Images Across All Lighting Conditions

Zihan Qin, Jialei Xu, Wenbo Zhao, Junjun Jiang and Xianming Liu$^{*}$

Abstract—Depth estimation under adverse conditions remains a significant challenge. Recently, multi-spectral depth estimation, which integrates both visible light and thermal images, has shown promise in addressing this issue. However, existing algorithms struggle with precise pixel-level feature matching, limiting their ability to fully exploit geometric constraints across different spectra. To address this, we propose a novel framework incorporating stereo depth estimation to enforce accurate geometric constraints. In particular, we treat the visible light and thermal images as a stereo pair and utilize a Cross-modal Feature Matching (CFM) Module to construct a cost volume for pixel-level matching. To mitigate the effects of poor lighting on stereo matching, we introduce Degradation Masking, which leverages robust monocular thermal depth estimation in degraded regions. Our method achieves state-of-the-art (SOTA) performance on the Multi-Spectral Stereo (MS2) dataset, with qualitative evaluations demonstrating high-quality depth maps under varying lighting conditions.

## I. INTRODUCTION

Depth estimation has received considerable attention in recent years due to its widespread applications in areas such as autonomous driving $[1]$, $[2]$, robotics $[3]$, and 3D reconstruction $[4]$. Significant advancements have been made in both monocular and stereo depth estimation through deep learning-based approaches. However, these depth estimation algorithms predominantly rely on the visible domain. Consequently, their performance often suffers significant degradation due to the decline in image quality, particularly under poor illumination conditions such as nighttime or rainy weather $[5]$, which prevented these algorithms from being widely applied in real-world scenarios.

To address this challenge, recent research has increasingly investigated alternative vision modalities such as near-infrared images $[6]$, $[7]$, and long-wave infrared (also known as thermal) images $[5]$, $[8]$, $[9]$ to achieve reliable and robust depth estimation in adverse conditions. Among these alternative modalities, thermal images have gained more popularity due to their low acquisition cost, robustness in adverse conditions, and consistent performance regardless of lighting variations. Some researchers have attempted to achieve depth estimation with only thermal images $[5]$, $[8]$. However, thermal images typically exhibit lower texture information, resolution, and higher noise levels compared to visible light images, resulting in less accurate depth estimation in well-lit areas.

![](./images/image_001.jpg)

Fig. 1. Depth from images of different modalities. (a) and (b) show the visible light and thermal images, respectively; (c) is the LiDAR ground truth corresponding to the thermal image; (d) and (e) present the depth maps obtained from monocular methods using the visible light and thermal modalities, respectively; (f) illustrates the depth map estimated by our multispectral method.

Given that thermal and visible light depth estimation each have distinct strengths and weaknesses, researchers have focused on integrating thermal images with visible light images to leverage complementary information from both modalities. However, the significant differences between thermal and visible images present challenges in exploiting correlations across modalities. The substantial appearance differences, pacrtiularly the lack of texture in thermal images, complicate keypoint matching. Additionally, low illumination can obscure objects in visible light images, leading to mismatches with thermal images. Consequently, previous multi-spectral methods [9], [10] generally avoid direct pixel-level matching between images, limiting their ability to leverage geometric constraints. As a result, their performance is highly dependent on the training dataset and exhibits poor generalization.

To this end, we propose a novel framework that integrates thermal and visible light images for robust and accurate depth estimation under varying lighting conditions, as shown in Fig. 1. Specifically, we first treat the input visible light and thermal images as a stereo pair and train the Cross-modal Feature Matching (CFM) Module to generate aligned feature vectors for each pixel. This alignment allows us to project visible light features onto thermal features across candidate depths, constructing a cost volume that provides accurate pixel-level matching in well-lit regions. Then we estimate depth probability distributions for both modalities, and introduce a degradation mechanism based on those distributions, which reverts to monocular thermal image depth estimation for regions with adverse conditions. Finally, we employ a Depth Module to generate the final depth map. Our experimental evaluations demonstrate that our

method surpasses existing state-of-the-art depth estimation methods, marking a significant advancement in the field. Our contributions are summarized as follows:

- We propose a novel multi-spectral depth estimation method that leverages pixel-level matching through stereo depth estimation. By incorporating geometric constraints between cameras, our method achieves more accurate depth estimation. To the best of our knowledge, this is the first application of stereo depth estimation architecture in this domain.

- For low-light regions where the visible light image becomes unreliable, we introduce a novel degradation mechanism that effectively degrades to monocular thermal depth estimation.

• We demonstrate that the proposed method achieves substantial improvements over current state-of-the-art techniques on the MS2 benchmark dataset $[5]$.

## II. RELATED WORKS

## A. Depth Estimation in the Visible Light Domain

Monocular Depth Estimation aims to infer scene depth from a single image. It can be approached as either a regression problem or a classification problem. Regression-based methods $[11]$, $[12]$, $[13]$, $[14]$ involve predicting per-pixel depth values using convolutional neural networks. In contrast, classification-based approaches $[15]$, $[16]$, $[17]$ divide depth into discrete intervals and predict probabilities for each pixel, transforming the depth estimation task into a classification problem.

However, monocular depth estimation is an ill-posed problem because a single 2D image can correspond to an infinite number of different 3D scenes. As a result, estimating absolute depth often leads to overfitting on specific datasets, capturing patterns that may not generalize to new environments $[18]$. Meanwhile, estimating only relative depth provides limited practical value in real-world applications.

Stereo Depth Estimation involves estimating a pixel-wise disparity map from a stereo image pair, which can be used to determine the depth of each pixel in the scene. Learning-based stereo depth estimation methods can roughly be divided into two main categories: the encoder-decoder network with 2D convolution [19], [20], [21], [22] and the cost volume matching with 3D convolution [23], [24], [25], [26]. The former directly outputs a disparity map, while the latter requires feature matching at multiple disparity levels to construct a cost volume, leading to higher computational costs but typically achieving greater accuracy.

Compared to monocular depth estimation, stereo depth estimation benefits from the geometric constraints between two viewpoints, resulting in significantly improved accuracy. However, stereo depth estimation faces challenges in handling occlusions, textureless areas, and reflective surfaces.

Multi-view Stereo (MVS) estimates the dense depth map from overlapping images. Yao et al. [27] extract deep visual features from the input images, followed by the construction of a 3D cost volume via differentiable homography warping.

Subsequent works have largely adopted these steps. Yao et al. [28] replaces the 3D CNN used in cost volume regularization with GRUs to reduce memory consumption, and similarly, Xu et al. [29] utilize a non-local RNN to achieve this objective. Yang et al. [30] have explored the use of feature pyramid networks to extract multi-scale features, enabling a coarse-to-fine construction of the cost volume. Bae et al. [31] integrate monocular depth estimation with Multi-view Stereo techniques, using a depth consistency constraint to ensure alignment between the cost volume and monocular depth, effectively addressing occlusion and textureless surface challenges.

Unlike stereo matching methods, MVS methods do not require stereo rectification of input images, making its approach to constructing the cost volume more generalizable.

## B. Multi-spectral Depth Estimation

Due to the decreased accuracy of depth estimation methods based on visible light under poor lighting conditions, researchers have increasingly turned to multi-spectral images for depth estimation. Treible et al. [32] attempted pixel-level matching between thermal and visible light images but were unable to obtain effective disparity and depth maps because of significant distribution differences between the modalities. Shin et al. [10] employed an Adversarial Multi-spectral Adaptation method, using visible light images as auxiliary supervision to estimate the depth of thermal images. Kim et al. [33] and Lu et al. [8] utilized specialized hardware to create accurately aligned visible-thermal image pairs for training, yet they still faced limitations in achieving accurate depth estimation due to a lack of geometric constraints. Guo et al. [9] employed cross-spectrum spatial consistency between visible light and thermal images for self-supervised learning. Lastly, Shin et al. [5] proposed a conditional random field block to estimate depth from thermal image pairs. However, due to the inherent limitations of thermal images, their method performs worse than visible light methods under favorable lighting conditions.

Compared to existing multi-spectral depth estimation methods, our approach effectively capitalizes on the complementary strengths of thermal and visible light images. In areas with favorable illumination, we employ geometric constraints derived from stereo vision to achieve precise depth estimation. In contrast, for regions with insufficient illumination, our method transitions to a monocular depth estimation approach based on thermal images. This adaptive mechanism ensures robust and accurate depth estimation across varying lighting conditions, addressing the limitations inherent in prior methodologies.

## III. METHOD

## A. Problem Formulation

Given a pair of calibrated visible light and thermal images, $I _ {vis}$ and $I _ {thr}$, with their respective intrinsics, rotations and translations $\{K _ {vis}, R _ {vis}, t _ {vis}\}$ and $\{K _ {thr}, R _ {thr}, t _ {thr}\}$, our goal is to find an accurate depth estimation from the two images by leveraging their complementary information. The problem

![](./images/image_002.jpg)

Fig. 2. The architecture of our proposed network. The method involves the following steps: First, based on the input visible light images $I _ {vis}$ and thermal images $I _ {thr}$, we utilize a cross-attention-based feature extractor to generate aligned feature vectors for each pixel by projecting between the two modalities and obtaining the cost volume for pixel-level matching. Second, we perform monocular depth estimation independently for each modality, producing pixel-wise depth probability distributions. Third, we apply Degradation Masking, derived from the visible image's depth probability distribution, to the cost volume to remove inaccurate matches. Finally, we utilize the final layer of features from the thermal MDP Module to degrade the masked cost volume into monocular thermal depth estimation, producing the final depth map through the Depth Module.

can be formulated as finding a function $f(\cdot |\theta)$ that maps the input images and camera parameters to the depth map:

$$
\mathbf{D}_{\mathrm{map}}=f(\mathbf{I}_{\mathrm{vis}},\mathbf{I}_{ \mathrm{thr}},\mathbf{K}_{\mathrm{vis}},\mathbf{K}_{\mathrm{thr}},[\mathbf{R}_ {\mathrm{vis}},\mathbf{t}_{\mathrm{vis}}],[\mathbf{R}_{\mathrm{thr}},\mathbf{t }_{\mathrm{thr}}]|\boldsymbol{\theta}),\ (1)
$$

which is supervised by ground truth depth maps obtained through LiDAR measurements. As illustrated in Fig. 2, our proposed approach encompasses three principal phases:

• Cross-modal Feature Matching. We first treat the input $I _ {vis}$ and $I _ {thr}$ as a stereo pair and train two Cross-modal Feature Matching (CFM) Modules to generate aligned feature vectors for each pixel. The visible light features are projected onto the thermal features across candidate depths, constructing a cost volume that facilitates pixel-level matching.

• Degradation Masking. To address the challenges posed by low-light regions that are difficult to match, we employ a Modality-specific Depth Probability (MDP) Module to estimate the depth probability of each pixel in $I _ {vis}$ as Gaussian distribution, and generate a degradation mask based on the probability. Then the mask is applied on the cost volume, which allows us to remove inaccurate matches.

- Depth Map Generation. Finally, to compensate for the removed portions, we employ another MDP Module to extract features from the thermal image and concatenate them with the masked cost volume. This is then fed into the Depth Module to produce the final depth estimation, allowing regions where matching fails to degrade to monocular thermal depth estimation, providing robust results under varying lighting conditions.

## B. Cross-modal Feature Matching

We leverage Multi-view Stereo (MVS) methods to construct a cost volume for feature matching across different views. Initially, we generate feature vectors for each pixel in both views using the CFM Module based on the PSMNet backbone [23]. Considering the different modalities of input images, their features do not reside in the same feature space, which hinders subsequent feature matching. To address this, we introduce a Cross-Attention Module to align the feature spaces. This module consists of two cross-attention layers. In the first layer, visible light features are used as queries and thermal features as keys, while in the second layer, the roles are reversed, with thermal features as queries and visible light features as keys:

$$
\mathbf{f}_{aligned}=\text{softmax}\left(\frac{\mathbf{QK}^{\top}}{ \sqrt{d}}\right)\mathbf{f}_{origin},\qquad\qquad(2)
$$

where Q and K are the query vector and key vector, respectively, d is the dimensionality of the key vector.

These aligned feature vectors are then projected across views by utilizing the intrinsic $K _ {thr}$, $K _ {vis}$ and extrinsic parameters $[R _ {vis}, t _ {vis}]$, $[R _ {thr}, t _ {thr}]$ to compute the matching pixels. The similarity between features of corresponding pixels is subsequently calculated to construct the cost volume. Specifically, for each pixel $(u, v)$ in $I _ {thr}$, we find its corresponding pixel $(u', v')$ in $I _ {vis}$ and select a set of uniformly sampled depth candidates $\{d _ {k}\} _ {k=1}^{N}$. For each depth candidate $d _ {k}$, the matching score for depth candidate $d _ {k}$ is calculated by taking the dot product of the features from both views:

$$
C(u,v,d_{k})=\mathbf{f}_{thr}(u,v)\cdot\mathbf{f}_{vis}(u^{'},v^{ '},d_{k}),\qquad\quad(3)
$$

where $f _ {thr}$ and $f _ {vis}$ are the feature vectors extracted from the thermal and visible light views, respectively. These computed similarities are organized into a cost volume, which is then processed with a softmax operation to generate a depth probability volume:

$$
D(u,v)=\{P(u,v,d_{1}),P(u,v,d_{2}),\ldots,P(u,v,d_{N})\},\ \ (4)
$$

where $P(u,v,d _ {k})$ represents the probability of depth $d _ {k}$ at pixel $(u,v)$, and N is the number of depth candidates.

## C. Degradation Masking

The Multi-view Stereo matching can provide accurate matching in well-lit regions. However, extracting visible light information in regions with adverse conditions is challenging, which can lead to unreliable matches. To address this problem, we propose a novel strategy, namely degradation masking, to remove inaccurate matches from the cost volume, and degrade them to monocular thermal depth estimation.

Specifically, we firstly identify the regions of low reliability within the visible light modality. To achieve this goal, we compute the depth probability distribution of $I _ {vis}$, as the probability can be a good representation of reliability. Here we employ the D-Net in MaGNet [31] as the MDP model to predict the depth value $d _ {uv}$ for each pixel $(u,v)$ in $I _ {vis}$, and model their probability $(P)$ as a Gaussian distribution to capture the depth uncertainty:

$$
P(d_{uv})=\frac{1}{\sqrt{2\pi\sigma_{uv}^{2}}}\exp\left(-\frac{(d_{uv}- \mu_{uv})^{2}}{2\sigma_{uv}^{2}}\right),\qquad(5)
$$

where $\mu _ {uv}$ represents the mean depth, and $\sigma _ {uv}^{2}$ denotes the variance at pixel $(u,v)$.

Since the low probability indicates that the corresponding pixel is more likely to be mismatch. We can simply remove the depth candidate corresponding to low probability to achieve degradation masking. This is achieved by exclude the depth candidate $d _ {k}$ for that pixel if $P _ {vis}(d _ {k}|(u,v))$ is below a certain threshold $\theta _ {(u,v)}$, which is computed by:

$$
\theta_{(u,v)}=\mu_{uv}+k*\sigma_{uv},\hskip 2cm(6)
$$

where k is a hyperparameter. We have found that setting k = 1 yields satisfactory results.

## D. Depth Map Generation

To degrade the mismatch of poorly lit regions to monocular thermal depth estimation, we utilize an additional MDP model to predict the depth of $I _ {thr}$. Features extracted from the final layer of this MDP module are concatenated with the masked cost volume. As discussed in [31], [23], maintaining the size of depth estimation and cost volume at $(H/4, W/4)$ ensures both computational efficiency and accuracy. To generate the depth map and recover the final depth map at full resolution, we apply our proposed Depth Module, which incorporates the learnable upsampling method introduced by Bae et al. [31]:

$$
[\mu_{d},\sigma_{d}^{2}]=\text{DepthModule}(\text{Concat}(C,F_{\text{ thermal}})),\qquad(7)
$$

where C denotes the cost volume constructed by the CFM Module, and $F _ {thermal}$ represents the feature map from the last layer of the thermal MDP Module. Depth Module outputs the depth mean $\mu _ {d}$ as the final estimated depth, while the variance $\sigma _ {d}^{2}$ is used only during training for loss computation.

## E. Training Details

We divide the training process of the whole network into three stages: CFM Module training, MDP Module training, and Depth Module training. The order of training the CFM and MDP Modules can be interchanged.

In the training of the CFM Module, we multiply the cost volume output by the depth candidates to obtain the expected depth values. The L1 loss is then computed between these expected depth values and the ground truth:

$$
L_{1}^{MS}=\sum_{u}^{W}\sum_{v}^{H}\left|\sum_{k=1}^{N}d_{k}\cdot p(d)- d_{uv}^{gt}\right|,\qquad\qquad(8)
$$

where $d _ {uv}$ represents the depth value at pixel $(u, v)$, H and W represent the height and width of the image, respectively.
In the training of the MDP Module, we use the encoder pre-trained on the KITTI dataset [34] from AdaBins [15], and employ the Negative Log-Likelihood (NLL) loss to optimize the mean and variance for each modality separately.

$$
L_{NLL}=\sum_{u}^{W}\sum_{v}^{H}\left[\frac{(d_{uv}^{gt}-\mu_{uv}( \mathbf{I}_{t}))^{2}}{2\sigma_{uv}^{2}(\mathbf{I}_{t})}+\frac{1}{2}\log\sigma _{uv}^{2}(\mathbf{I}_{t})\right],\\(9)
$$

where $\mu _ {uv}$ denotes the predicted mean depth, and $\sigma _ {uv}^{2}$ indicates the variance. Since there are two MDP Modules, we compute two loss $L _ {NLL}^{VIS}$, $L _ {NLL}^{THR}$ for $I _ {vis}$ and $I _ {thr}$, respectively.
In the training of Depth Module, the weights of the MDP Module and CFM Module are frozen. The training is conducted using the same NLL loss function in Eq. 9 as $L _ {NLL}^{MS}$. The final depth map $D _ {map}$ is obtained as the mean $\mu$ derived from this process.

## IV. EXPERIMENTS

## A. Datasets

We utilize Multi-Spectral Stereo (MS2) benchmark dataset $[5]$ to evaluate our proposed method. MS2 dataset $[5]$ consists of approximately 195,000 pairs of multi-modal data, including stereo visible light images, stereo near-infrared (NIR) images, stereo long-wave infrared (thermal) images, stereo LiDAR point clouds, and GNSS/IMU information. We follow the official data split and conduct evaluations on test sets corresponding to different weather and lighting conditions, including clear daytime, nighttime, and rainy weather.

## B. Implementation Details

We implement our network with PyTorch [37] and conduct training on two NVIDIA RTX 4090 GPUs. We use AdamW optimizer [38] and schedule the learning rate using 1cycle policy [39] with $lr _ {max} = 3.57 \times  10^{-5}$ across all three modules. The batch size is 16/8/4 for MDP Module, CFM Module and Depth Module respectively. The number of

TABLE I

QUANTITATIVE DEPTH COMPARISON ON THE OFFICIAL SPLIT OF MS2 DATASET



<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="4"></td><td colspan="3"></td></tr><tr><td>Abs Rel</td><td>$\quad \mathrm{Sq}\ \mathrm{Rel}$</td><td>RMSE</td><td>RMSE log</td><td>$\delta<1.25$</td><td>$\delta<1.25^2$</td><td></td></tr><tr><td rowspan="4"></td><td>day</td><td>0.144</td><td>1.288</td><td>5.483</td><td>0.230</td><td>0.856</td><td>0.941</td><td></td></tr><tr><td>night</td><td>0.136</td><td>1.136</td><td>5.290</td><td>0.212</td><td>0.863</td><td>0.950</td><td></td></tr><tr><td>rain</td><td>0.180</td><td>1.934</td><td>6.735</td><td>0.276</td><td>0.781</td><td>0.910</td><td></td></tr><tr><td>avg</td><td>0.151</td><td>1.419</td><td>5.776</td><td>0.237</td><td>0.837</td><td>0.935</td><td></td></tr><tr><td rowspan="4"></td><td>day</td><td>0.122</td><td>0.905</td><td>4.923</td><td>0.198</td><td>0.857</td><td>0.951</td><td></td></tr><tr><td>night</td><td>0.114</td><td>0.798</td><td>4.701</td><td>0.184</td><td>0.870</td><td>0.959</td><td></td></tr><tr><td>rain</td><td>0.157</td><td>1.395</td><td>6.053</td><td>0.243</td><td>0.791</td><td>0.926</td><td></td></tr><tr><td>avg</td><td>0.129</td><td>1.008</td><td>5.169</td><td>0.206</td><td>0.843</td><td>0.947</td><td></td></tr><tr><td rowspan="4"></td><td>day</td><td>0.129</td><td>0.976</td><td>5.108</td><td>0.205</td><td>0.847</td><td>0.947</td><td></td></tr><tr><td>night</td><td>0.119</td><td>0.822</td><td>4.749</td><td>0.187</td><td>0.864</td><td>0.958</td><td></td></tr><tr><td>rain</td><td>0.168</td><td>1.545</td><td>6.336</td><td>0.254</td><td>0.771</td><td>0.918</td><td></td></tr><tr><td>avg</td><td>0.137</td><td>1.084</td><td>5.330</td><td>0.212</td><td>0.831</td><td>0.943</td><td></td></tr><tr><td rowspan="4"></td><td>day</td><td>0.120</td><td>0.864</td><td>4.852</td><td>0.195</td><td>0.858</td><td>0.952</td><td></td></tr><tr><td>night</td><td>0.112</td><td>0.755</td><td>4.594</td><td>0.179</td><td>0.875</td><td>0.961</td><td></td></tr><tr><td>rain</td><td>0.115</td><td>1.352</td><td>5.956</td><td>0.240</td><td>0.795</td><td>0.929</td><td></td></tr><tr><td>avg</td><td>0.127</td><td>0.965</td><td>5.077</td><td>0.202</td><td>0.846</td><td>0.949</td><td></td></tr><tr><td rowspan="4"></td><td>day</td><td>0.115</td><td>0.983</td><td>4.895</td><td>0.201</td><td>0.882</td><td>0.952</td><td></td></tr><tr><td>night</td><td>0.107</td><td>0.850</td><td>4.658</td><td>0.185</td><td>0.894</td><td>0.961</td><td></td></tr><tr><td>rain</td><td>0.152</td><td>1.567</td><td>6.020</td><td>0.247</td><td>0.822</td><td>0.928</td><td></td></tr><tr><td>avg</td><td>0.123</td><td>1.103</td><td>5.134</td><td>0.208</td><td>0.869</td><td>0.948</td><td></td></tr><tr><td rowspan="4"></td><td>day</td><td>0.113</td><td>0.948</td><td>4.852</td><td>0.200</td><td>0.884</td><td>0.953</td><td></td></tr><tr><td>night</td><td>0.105</td><td>0.811</td><td>4.584</td><td>0.183</td><td>0.896</td><td>0.961</td><td></td></tr><tr><td>rain</td><td>0.149</td><td>1.499</td><td>5.940</td><td>0.245</td><td>0.826</td><td>0.929</td><td></td></tr><tr><td>avg</td><td>0.120</td><td>1.057</td><td>5.068</td><td>0.207</td><td>0.872</td><td>0.949</td><td></td></tr><tr><td rowspan="4"></td><td>day</td><td>0.098</td><td>0.549</td><td>3.593</td><td>0.139</td><td>0.893</td><td>0.980</td><td></td></tr><tr><td>night</td><td>0.103</td><td>0.519</td><td>3.398</td><td>0.142</td><td>0.888</td><td>0.980</td><td></td></tr><tr><td>rain</td><td>0.130</td><td>0.802</td><td>4.461</td><td>0.173</td><td>0.830</td><td></td><td></td></tr><tr><td>avg</td><td>0.110</td><td>0.623</td><td>3.817</td><td>0.151</td><td>0.870</td><td></td><td></td></tr></table>

![](./images/image_003.jpg)

(a) Visible Image

(b) Thermal Image

(c) Adabins (vis)

(d) Adabins (thr)

(e) Ours

(f) Ours (Variance)

Fig. 3. Quantitative depth comparison on the MS2 dataset. From left to right: visible images, thermal images, depth maps generated by Adabins $[15]$ using either visible or thermal images, and depth and variance maps produced by our approach. The first two rows show results from the day test set, the middle two from the night test set, and the last two from the rainy test set. The results demonstrate that our method effectively leverages information from different modalities, producing robust and stable results under varying lighting conditions.

epochs is 5 for all three modules. The input raw thermal image is transformed according to (10):

$$
T_{\text{Celsius}}=\frac{B}{\log\left(\frac{R}{\text{Raw}-O}+F \right)}-273.15,\qquad\quad(10)
$$

where R = 380747, B = 1428, F = 1, and O = -88.539.

## C. Results

Since there are very few methods for multi-spectrum stereo depth estimation, we compare our proposed method with the state-of-the-art monocular and stereo depth networks DETI(mono) and DETI(stereo) [5] from thermal images.

The objective comparison results are shown in Table I. We leverage the standard evaluation protocol from $[15]$, $[16]$, $[5]$ to validate the efficacy of the proposed method in experiments, i.e., relative absolute error (Abs Rel), relative squared error (Sq Rel), root mean squared error (RMSE), root mean squared logarithmic error (RMSE log) and threshold accuracy ($\delta < 1.25$, $\delta < 1.25^{2}$, $\delta < 1.25^{3}$). Due to improvements in multi-modal information fusion and dual-view geometric constraints, our method achieves state-of-the-art performance on the MS2 dataset for most metrics, demonstrating a significant advantage in the Abs Rel metric. This is attributed to the successful integration of information across different modalities, which substantially reduces depth estimation errors. However, thermal images are significantly impacted by temperature variations, leading to degraded imaging quality in rainy conditions and resulting in some performance loss for our method.

The results of the subjective comparison are shown in Fig. 3. Since DETI [5] is not open-sourced, we present a comparison of subjective results using Adabins [15], trained on visible light or thermal images. It can be seen that the methods based on the visible light modality can provide accurate depth estimation when sufficient light is guaranteed. However, when lighting conditions deteriorate (e.g., at night or in rainy weather), their performance degrades rapidly. In contrast, methods based on thermal images, while maintaining relatively high visibility under adverse light conditions, face limitations due to their inherently lower resolution and contrast. This results in the generation of less detailed depth maps. Additionally, the dependence of thermal images on temperature means that their imaging quality can be further compromised in scenarios such as rainy weather, where temperature variations may affect the thermal signature. Our method integrates detailed information from visible light images with the enhanced visibility of thermal images in low-light scenarios, enabling it to generate accurate and robust depth maps across a wide range of lighting conditions.

## D. Ablation

Ablation study on Cross-Modal Feature Matching: To investigate the importance of the CFM Module, we train two independent MDP Modules for visible light and thermal images, separately, and compare their performance with the full pipeline model. As shown in Table II, it can be

TABLE II ABLATION STUDY ON CROSS-MODAL FEATURE MATCHING



<table><tr><td rowspan="2">Method</td><td rowspan="2">Condition</td><td colspan="2">Error↓</td><td>Accuracy↑</td></tr><tr><td>Abs Rel</td><td>RMSE</td><td>$\delta<1.25$</td></tr><tr><td rowspan="4">Mono-VIS</td><td>day</td><td>0.132</td><td>4.558</td><td>0.843</td></tr><tr><td>night</td><td>0.190</td><td>5.671</td><td>0.729</td></tr><tr><td>rain</td><td>0.164</td><td>5.623</td><td>0.764</td></tr><tr><td>avg</td><td>0.162</td><td>5.284</td><td>0.779</td></tr><tr><td rowspan="4">Mono-THR</td><td>day</td><td>0.101</td><td>3.777</td><td>0.883</td></tr><tr><td>night</td><td>0.108</td><td>3.539</td><td>0.877</td></tr><tr><td>rain</td><td>0.138</td><td>4.821</td><td>0.808</td></tr><tr><td>avg</td><td>0.116</td><td>4.046</td><td>0.856</td></tr><tr><td rowspan="4">Stereo-MS</td><td>day</td><td>0.098</td><td>3.593</td><td>0.893</td></tr><tr><td>night</td><td>0.103</td><td>3.398</td><td>0.888</td></tr><tr><td>rain</td><td>0.130</td><td>4.461</td><td>0.830</td></tr><tr><td>avg</td><td>0.110</td><td>3.817</td><td>0.870</td></tr></table>

TABLE III ABLATION STUDY ON DEGRADATION MASK



<table><tr><td rowspan="2">Method</td><td rowspan="2">Condition</td><td colspan="2">Error↓</td><td>Accuracy↑</td></tr><tr><td>Abs Rel</td><td>RMSE</td><td>$\delta<1.25$</td></tr><tr><td rowspan="4">Without Degration</td><td>day</td><td>0.151</td><td>3.838</td><td>0.876</td></tr><tr><td>night</td><td>0.147</td><td>3.702</td><td>0.871</td></tr><tr><td>rain</td><td>0.178</td><td>4.540</td><td>0.823</td></tr><tr><td>avg</td><td>0.159</td><td>4.027</td><td>0.857</td></tr><tr><td rowspan="4">Full</td><td>day</td><td>0.098</td><td>3.593</td><td>0.893</td></tr><tr><td>night</td><td>0.103</td><td>3.398</td><td>0.888</td></tr><tr><td>rain</td><td>0.130</td><td>4.461</td><td>0.830</td></tr><tr><td>avg</td><td>0.110</td><td>3.817</td><td>0.870</td></tr></table>

observed that our approach effectively leverages the advantages of both modalities, utilizing the geometric properties between the two views to further improve the accuracy of depth estimation. Additionally, as shown in Fig. 1, the ablation results demonstrate that without the CFM Module, the model's ability to capture fine-grained spatial details is notably diminished, highlighting the importance of cross-modal interaction for achieving superior depth perception.

Ablation study on Degradation Masking: We conduct an ablation study to evaluate the effect of Degradation Masking. Specifically, we retrain a network using only the CFM Module and Depth Module and compare its performance with our full model. The results can be found in Table III. Our model significantly outperforms the retrained network across all metrics, demonstrating the effectiveness of the proposed Degradation Masking.

## V. CONCLUSION

In this paper, we propose a novel framework that integrates thermal and visible light images to produce accurate and robust depth maps across various lighting conditions. Specifically, we introduce Cross-Modal Feature Matching to bridge the gap between thermal and visible light images in depth estimation. Additionally, we present Degradation Masking to handle regions where matching fails due to insufficient lighting or texture loss in monocular thermal depth estimation. Our method achieves state-of-the-art performance on the MS2 [5] dataset. Ablation studies demonstrate that both the cross-modal matching mechanism and the degradation masking significantly enhance the precision and robustness of the algorithm.

## REFERENCES



[1] Y. Wang, W.-L. Chao, D. Garg, B. Hariharan, M. Campbell, and K. Q. Weinberger, “Pseudo-lidar from visual depth estimation: Bridging the gap in 3d object detection for autonomous driving,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 8445–8453.

[2] Y. You, Y. Wang, W.-L. Chao, D. Garg, G. Pleiss, B. Hariharan, M. Campbell, and K. Q. Weinberger, “Pseudo-lidar++: Accurate depth for 3d object detection in autonomous driving,” in International Conference on Learning Representations (ICLR), 2020.

[3] D. Wofk, F. Ma, T.-J. Yang, S. Karaman, and V. Sze, “Fastdepth: Fast monocular depth estimation on embedded systems,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 6101–6108.

[4] A. Geiger, J. Ziegler, and C. Stiller, “Stereoscan: Dense 3d reconstruction in real-time,” in 2011 IEEE intelligent vehicles symposium (IV). Ieee, 2011, pp. 963–968.

[5] U. Shin, J. Park, and I. S. Kweon, “Deep depth estimation from thermal image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 1043–1053.

[6] J. Park, Y. Jeong, K. Joo, D. Cho, and I. S. Kweon, “Adaptive cost volume fusion network for multi-modal depth estimation in changing environments,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 5095–5102, 2022.

[7] S. Brucker, S. Walz, M. Bijelic, and F. Heide, “Cross-spectral gated-rgb stereo depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 21654–21665.

[8] Y. Lu and G. Lu, “An alternative of lidar in nighttime: Unsupervised depth estimation based on single thermal image,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2021, pp. 3833–3843.

[9] Y. Guo, H. Kong, and S. Gu, “Unsupervised multi-spectrum stereo depth estimation for all-day vision,” IEEE Transactions on Intelligent Vehicles, 2023.

[10] U. Shin, K. Park, B.-U. Lee, K. Lee, and I. S. Kweon, “Self-supervised monocular depth estimation from thermal images via adversarial multi-spectral adaptation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 5798–5807.

[11] D. Eigen, C. Puhrsch, and R. Fergus, “Depth map prediction from a single image using a multi-scale deep network,” Advances in neural information processing systems, vol. 27, 2014.

[12] I. Laina, C. Rupprecht, V. Belagiannis, F. Tombari, and N. Navab, "Deeper depth prediction with fully convolutional residual networks," in 2016 Fourth international conference on 3D vision (3DV). IEEE, 2016, pp. 239–248.

[13] H. Fu, M. Gong, C. Wang, K. Batmanghelich, and D. Tao, “Deep ordinal regression network for monocular depth estimation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2002–2011.

[14] J. Xu, X. Liu, Y. Bai, J. Jiang, K. Wang, X. Chen, and X. Ji, "Multi-camera collaborative depth prediction via consistent structure estimation," in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 2730–2738.

[15] S. F. Bhat, I. Alhashim, and P. Wonka, “Adabins: Depth estimation using adaptive bins,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 4009–4018.

[16] ——, "Localbins: Improving depth estimation by learning local distributions," in European Conference on Computer Vision. Springer, 2022, pp. 480–496.

[17] S. Shao, Z. Pei, X. Wu, Z. Liu, W. Chen, and Z. Li, “Iebins: Iterative elastic bins for monocular depth estimation,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[18] S. F. Bhat, R. Birkl, D. Wofk, P. Wonka, and M. Müller, "Zoedepth: Zero-shot transfer by combining relative and metric depth," arXiv preprint arXiv:2302.12288, 2023.

[19] N. Mayer, E. Ilg, P. Hausser, P. Fischer, D. Cremers, A. Dosovitskiy, and T. Brox, "A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4040–4048.

[20] E. Ilg, T. Saikia, M. Keuper, and T. Brox, "Occlusions, motion and depth boundaries with a generic network for disparity, optical flow or scene flow estimation," in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 614–630.



[21] P. Weinzaepfel, T. Lucas, V. Leroy, Y. Cabon, V. Arora, R. Brégier, G. Csurka, L. Antsfeld, B. Chidlovskii, and J. Revaud, “Croco v2: Improved cross-view completion pre-training for stereo matching and optical flow,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 17969–17980.

[22] J. Xu, X. Liu, J. Jiang, and X. Ji, "Sdge: Stereo guided depth estimation for 360 $\{\backslash$deg} camera sets," arXiv preprint arXiv:2402.11791, 2024.

[23] J.-R. Chang and Y.-S. Chen, “Pyramid stereo matching network,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 5410–5418.

[24] F. Zhang, V. Prisacariu, R. Yang, and P. H. Torr, “Ga-net: Guided aggregation net for end-to-end stereo matching,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 185–194.

[25] F. Zhang, X. Qi, R. Yang, V. Prisacariu, B. Wah, and P. Torr, "Domain-invariant stereo matching networks," in Computer Vision—ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16. Springer, 2020, pp. 420–439.

[26] G. Xu, X. Wang, X. Ding, and X. Yang, “Iterative geometry encoding volume for stereo matching,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21919–21928.

[27] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “Mvsnet: Depth inference for unstructured multi-view stereo,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 767–783.

[28] Y. Yao, Z. Luo, S. Li, T. Shen, T. Fang, and L. Quan, “Recurrent mvsnet for high-resolution multi-view stereo depth inference,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 5525–5534.

[29] Q. Xu, M. R. Oswald, W. Tao, M. Pollefeys, and Z. Cui, “Non-local recurrent regularization networks for multi-view stereo,” arXiv preprint arXiv:2110.06436, 2021.

[30] J. Yang, W. Mao, J. M. Alvarez, and M. Liu, "Cost volume pyramid based depth inference for multi-view stereo," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 4877–4886.

[31] G. Bae, I. Budvytis, and R. Cipolla, “Multi-view depth estimation by fusing single-view depth probability with multi-view geometry,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 2842–2851.

[32] W. Treible, P. Saponaro, S. Sorensen, A. Kolagunda, M. O'Neal, B. Phelan, K. Sherbondy, and C. Kambhamettu, "Cats: A color and thermal stereo benchmark," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 2961–2969.

[33] N. Kim, Y. Choi, S. Hwang, and I. S. Kweon, “Multispectral transfer network: Unsupervised depth estimation for all-day vision,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.

[34] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, “Vision meets robotics: The kitti dataset,” International Journal of Robotics Research (IJRR), 2013.

[35] J. H. Lee, M.-K. Han, D. W. Ko, and I. H. Suh, “From big to small: Multi-scale local planar guidance for monocular depth estimation,” arXiv preprint arXiv:1907.10326, 2019.

[36] W. Yuan, X. Gu, Z. Dai, S. Zhu, and P. Tan, “Neural window fully-connected crfs for monocular depth estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 3916–3925.

[37] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[38] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” arXiv preprint arXiv:1711.05101, 2017.

[39] L. N. Smith and N. Topin, “Super-convergence: Very fast training of neural networks using large learning rates,” in Artificial intelligence and machine learning for multi-domain operations applications, vol. 11006. SPIE, 2019, pp. 369–386.