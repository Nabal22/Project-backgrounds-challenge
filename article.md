### Noise or Signal: The Role of Image Backgrounds in
Object Recognition
Kai Xiao Logan Engstrom Andrew Ilyas
Aleksander M ˛adry
MIT
```
{kaix, engstrom, ailyas, madry}@mit.edu
```
Abstract
We assess the tendency of state-of-the-art object recognition models to depend on
signals from image backgrounds. We create a toolkit for disentangling foreground
```
and background signal on ImageNet images, and find that (a) models can achieve
```
```
non-trivial accuracy by relying on the background alone, (b) models often misclas-
```
sify images even in the presence of correctly classified foregrounds—up to 87.5%
```
of the time with adversarially chosen backgrounds, and (c) more accurate models
```
tend to depend on backgrounds less. Our analysis of backgrounds brings us closer
to understanding which correlations machine learning models use, and how they
determine models’ out of distribution performance.
1 Introduction
Object recognition models are typically trained to minimize loss on a given dataset, and evaluated
by the accuracy they attain on the corresponding test set. In this paradigm, model performance can
be improved by incorporating any generalizing correlation between images and their labels into
decision-making. However, the actual model reliability and robustness depend on the specific set
of correlations that is used, and on how those correlations are combined. Indeed, outside of the
training distribution, model predictions can deviate wildly from human expectations either due to
```
relying on correlations that humans do not perceive [JLT18; Ily+19; Jac+19], or due to overusing
```
```
correlations, such as texture [Gei+19; Bak+18] and color [YS02], that humans do use (but to a lesser
```
```
degree). Characterizing the correlations that models depend on thus has important implications for
```
understanding model behavior, in general.
Image backgrounds are a natural source of correlation between images and their labels in object
```
recognition. Indeed, prior work has shown that models may use backgrounds in classification [Zha+07;
```
```
RSG16; ZXY17; RZT18; Bar+19; SSF19; Sag+20], and suggests that even human vision makes
```
use of image context for scene and object recognition [Tor03]. In this work, we aim to obtain a
deeper understanding of how current state-of-the-art image classifiers utilize image backgrounds.
Specifically, we investigate the extent to which models rely on them, the implications of this reliance,
and how models’ use of backgrounds has evolved over time. Concretely:
• We create a variety of datasets that help disentangle the impacts of foreground and back-
ground signals on classification. The test datasets and a public challenge related to them are
available at https://github.com/MadryLab/backgrounds_challenge.
• Using the aforementioned toolkit, we characterize models’ reliance on image backgrounds.
We find that image backgrounds alone suffice for fairly successful classification and that
changing background signals decreases average-case performance. In fact, we further show
that by choosing backgrounds in an adversarial manner, we can make standard models
misclassify 87.5% of images as the background class.
Preprint. Under review.
```
arXiv:2006.09994v1 [cs.CV] 17 Jun 2020
```
insect
Original
bird
Only-BG-B
insect
Only-BG-T
bird
No-FG
instrument
Only-FG
insect
Mixed-Same
insect
Mixed-Rand
instrument
Mixed-Next
Figure 1: Variations of the synthetic dataset ImageNet-9, as described in Table 1. We label each
```
image with its pre-trained ResNet-50 classification—green, if corresponding with the original label;
```
red, if not. The model correctly classifies the image as “insect” when given: the original image, only
the background, and two cases where the original foreground is present but the background changes.
Note that, in particular, the model fails in two cases when the original foreground is present but the
```
background changes (as in MIXED-NEXT or ONLY-FG).
```
• We demonstrate that standard models not only use but require backgrounds for correctly
```
classifying large portions of test sets (35% on our benchmark).
```
• We study the impact of backgrounds on classification for a variety of classifiers, and find
that more accurate models tend to simultaneously exploit background correlations more and
have greater robustness to changes in image background.
2 Methodology
To properly gauge image backgrounds’ role in image classification, we construct a synthetic dataset
for disentangling background from foreground signal: ImageNet-9.
Base dataset: ImageNet-9. We organize a subset of ImageNet into a new dataset with nine coarse-
```
grained classes and call it ImageNet-9 (IN-9) 1. To create it, we group together ImageNet classes
```
sharing an ancestor in the WordNet [Mil95] hierarchy. We separate out foregrounds and backgrounds
via the annotated bounding boxes provided in ImageNet, and remove all candidate images whose
bounding boxes are unavailable. We use coarse-grained classes because there are not enough images
with bounding boxes to use the standard labels, and the resulting IN-9 dataset has 5045 training and
450 testing images per class. We describe the dataset creation process in detail in Appendix A.
Variations of ImageNet-9 From this base set of images, which we call the ORIGINAL version of IN-
9, we create seven other synthetic variations designed to understand the impact of backgrounds. We
visualize these variations in Figure 1, and provide a detailed reference in Table 1. These subdatasets
of IN-9 differ only in how they process the foregrounds and backgrounds of each constituent image.
Larger dataset: IN-9L We finally create a dataset called IN-9L that consists of all the images in
```
ImageNet corresponding to the classes in ORIGINAL (rather than just the images that have associated
```
```
bounding boxes). We leverage this larger dataset to train better generalizing models.
```
1These classes are dog, bird, vehicle, reptile, carnivore, insect, instrument, primate, and fish.
2
Table 1: The 8 modified subdatasets created from ImageNet-9. The foreground detection method
refers to how the pixels corresponding to the foreground are found. ImageNet annotation refers to the
annotated bounding boxes found in ImageNet. GrabCut refers to the GrabCut algorithm [RKB04] as
implemented in OpenCV2. Random backgrounds in the last three datasets are taken from ONLY-BG-
T. For more details see Appendix A.
Name Foreground Background Foreground Detection Method
ORIGINAL Unmodified Unmodified —
ONLY-BG-B Black Unmodified ImageNet Annotation
ONLY-BG-T Tiled background Unmodified ImageNet Annotation
NO-FG Black Unmodified GrabCut
ONLY-FG Unmodified Black GrabCut
MIXED-SAME Unmodified Random background of the same class GrabCut
MIXED-RAND Unmodified Random background of a random class GrabCut
MIXED-NEXT Unmodified Random background of the next class GrabCut
ONLY-BG-B ONLY-BG-T NO-FG
0
20
40
60
80
100
Training dataset
Test Accuracy
Testing dataset
Same as train
ORIGINAL
Figure 2: We train models on each of the “background-only” datasets, then evaluate each on its
corresponding test set as well as the ORIGINAL test set. Even though the model only learns from
```
background signal, it achieves (much) better than random performance on both the corresponding
```
```
test set and ORIGINAL. Here, random guessing would give 11.11% (the dotted line).
```
3 Quantifying Reliance on Background Signals
With ImageNet-9 in hand, we now assess the role of image backgrounds in classification.
Backgrounds suffice for classification. Prior work has found that models are able to make accurate
```
predictions based on backgrounds alone; we begin by directly quantifying this ability. Looking at the
```
```
ONLY-BG-T, ONLY-BG-B, and NO-FG datasets, we find (cf. Figure 2) that models trained on these
```
background-only training sets generalize reasonably well to both their corresponding test sets and
```
the ORIGINAL test set (around 40-50% for every model, far above the baseline of 11% representing
```
```
random classification). Our results confirm that image backgrounds contain signal that models can
```
accurately classify with.
Models exploit background signal for classification. We discover that models can misclassify due
to background signal, especially when the background class does not match that of the foreground. As
a demonstration, we study model accuracies on the MIXED-RAND dataset, where image backgrounds
are randomized and thus provide no information about the correct label. By comparing test accuracies
on MIXED-RAND and MIXED-SAME 2, where images have class-consistent backgrounds, we can
measure classifiers’ dependence on the correct background. We denote the resulting accuracy gap
```
between MIXED-SAME and MIXED-RAND as the BG-GAP; this difference represents the drop in
```
2 MIXED-SAME controls for artifacts from image processing present in MIXED-RAND. For further discussion,
see Appendix D.
3
Table 2: Performance of state-of-the-art computer vision models on select test sets of ImageNet-
9. We include both pre-trained ImageNet models and models of different architectures that we
train on IN-9L. The BG-GAP is defined as the difference in test accuracy between MIXED-SAME
and MIXED-RAND and helps assess the tendency of such models to rely on background signal.
Architectures are sorted by their test accuracies on ImageNet and ORIGINAL for pre-trained and
IN-9L-trained models, respectively. Shaded in grey are the two architectures that can be directly
```
compared across datasets (ResNet-50 and Wide-ResNet-50x2).
```
Pre-trained on ImageNet Trained on IN-9L
Test datasetMobileNet-v3EfficientNet-b0ResNet-50WRN-50x2DPN-92AlexNetShuffleNetResNet-50WRN-50x2VGG16-BN
ImageNet 67.9% 77.2% 77.6% 78.5% 80.0% ——
ORIGINAL 91.0% 95.6% 96.2% 95.8% 96.8% 86.7% 95.7% 96.3% 97.2% 97.6%
IN-9L 90.0% 94.3% 95.0% 95.5% 96.0% 83.1% 93.2% 94.6% 95.2% 96.0%
ONLY-BG-T 15.7% 11.9% 17.8% 20.7% 20.6% 41.5% 43.6% 43.6% 45.1% 45.7%
MIXED-SAME 69.7% 79.7% 82.3% 81.7% 85.4% 76.2% 86.7% 89.9% 90.6% 91.0%
MIXED-RAND 56.1% 67.8% 76.3% 73.0% 77.6% 54.2% 69.4% 75.6% 78.0% 78.0%
BG-gap 13.6% 11.9% 6.0% 8.7% 7.8% 22.0% 17.3% 14.3% 12.6% 13.0%
model accuracy due to changing the class signal from the background. In Table 2, we observe a
BG-GAP of 13-22% and 6-14% for models trained on IN-9L and ImageNet, respectively, suggesting
that backgrounds often mislead state-of-the-art models even when the correct foreground is present.
Our results indicate that ImageNet-trained models are less dependent on backgrounds than their
```
IN-9L-trained counterparts—they have a smaller (but still significant) BG-GAP, and perform worse
```
```
when predicting solely based on backgrounds (i.e., on the ONLY-BG-T dataset). An initial hypothesis
```
```
could be that ImageNet-trained models’ lesser dependence results from having either (a) a more
```
```
fine-grained class structure, or (b) more datapoints than IN-9L. However, a preliminary investigation
```
```
(Appendix B) ablating both fine-grainedness and dataset size does not find evidence supporting either
```
explanation. Therefore, understanding why pre-trained ImageNet models rely less on backgrounds
than IN-9L models remains an open question.
Models are vulnerable to adversarial backgrounds. To understand how worst-case backgrounds
impact models’ performance, we evaluate model robustness to adversarially chosen backgrounds. We
```
find that 87.5% of foregrounds are susceptible to such backgrounds; that is, for these foregrounds,
```
there is a background that causes the classifier to classify the resulting foreground-background
combination as the background class. For a finer grained look, we also evaluate image backgrounds
```
based on their attack success rate (ASR), i.e., how frequently they cause models to predict the
```
```
(background) class in the presence of a conflicting foreground class. As an example, Figure 3 shows
```
```
the five backgrounds with the highest ASR for the insect class—these backgrounds (extracted from
```
```
insect images in ORIGINAL) fool a IN-9L-trained ResNet-50 model into predicting insect on up
```
to 52% of non-insect foregrounds. We plot a histogram of ASR over all insect backgrounds in
```
Figure 4—it has a long tail. Similar results are observed for other classes as well (cf. Appendix G).
```
Training on MIXED-RAND reduces background dependence. Next, we explore how to reduce
models’ dependence on background. To this end, we train models on MIXED-RAND, a synthetic
dataset where background signals are decorrelated from class labels. As we would expect, MIXED-
RAND-trained models extract less signal from backgrounds: evaluation results show that MIXED-
```
RAND models perform poorly (15% accuracy—barely higher than random) on datasets with only
```
```
backgrounds and no foregrounds (ONLY-BG-T or ONLY-BG-B).
```
Indeed, such models are also more accurate on datasets where backgrounds do not match foregrounds.
In Figure 5, we observe that a MIXED-RAND-trained model has 17.3% higher accuracy than its
ORIGINAL-trained counterpart on MIXED-RAND, and 22.3% higher accuracy on MIXED-NEXT, a
```
dataset where background signals class-consistently mismatch foregrounds. (Recall that MIXED-
```
4
insect, 66.55%
42% 44% 45% 45% 52%
Figure 3: The adversarial backgrounds that most frequently fool IN-9L-trained models into classify-
ing a given foreground as insect, ordered by the percentage of foregrounds fooled. The total portion
```
of images that can be fooled (by any background from this class) is 66.55%.
```
0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45
0
20
40
60
Adversarial success rate
Count
Target class: insect
```
Figure 4: Histogram of insect backgrounds grouped by how often they cause (non-insect) foregrounds
```
to be classified as insect by a IN-9L-trained model. We visualize the five backgrounds that fool the
classifier on the largest percentage of images in Figure 3.
NEXT images have foregrounds from class y mixed with backgrounds from class y + 1, labeled as
```
class y.) The MIXED-RAND-trained model also has little variation (at most 3.8%) in accuracy across
```
all five test sets that contain the correct foreground.
Qualitatively, the MIXED-RAND-trained model also appears to place more relative importance on
```
foreground pixels than the ORIGINAL-trained model; the saliency maps of the two models in Figure 6
```
show that the MIXED-RAND-trained model’s saliency maps highlight more foreground pixels than
those of ORIGINAL-trained models.
A fine grained look at dependence on backgrounds. We now analyze models’ reliance on back-
grounds at an image-by-image level and ask: for which images does introducing backgrounds
help or hurt classifiers’ performance? To this end, for each image in ORIGINAL, we decompose
how models use foreground and background signals by examining classifiers’ predictions on the
corresponding image in MIXED-RAND and ONLY-BG-T. Here, we use the MIXED-RAND and
```
ONLY-BG-T predictions as a proxy for which class the foreground and background signals (alone)
```
point towards, respectively. We categorize each image based on how its background and foreground
```
signals impact classification; we list the categories in Table 3 and show the counts for each category
```
as a histogram per classifier in Figure 7. Our results show that while few backgrounds induce
```
misclassification (see Appendix H for examples), a large fraction of images require backgrounds
```
for correct classification—approximately 35% on the ORIGINAL trained classifiers, as calculated by
combining the “BG Required” and “BG+FG Required” categories.
Further insights derived from IN-9 are discussed in the Appendix D. We focus on key findings
in this section, but also include more comprehensive results and examples of other questions that can
be explored by using the toolkit of IN-9 in the Appendix.
5
MIXED-RAND ORIGINAL
0
20
40
60
80
100
Training dataset
Test Accuracy
MIXED-NEXT
MIXED-RAND
MIXED-SAME
ONLY-FG
ORIGINAL
Figure 5: We compare the test performance of a model trained on the synthetic MIXED-RAND dataset
with a model trained on ORIGINAL. We evaluate these models on variants of IN-9 that contain
identical foregrounds. For the ORIGINAL-trained model, test performance decreases significantly
when the background signal is modified during testing. However, the MIXED-RAND-trained model is
robust to background changes, albeit at the cost of lower accuracy on images from ORIGINAL.
Image Original Saliency Mixed-Rand Saliency Image Original Saliency Mixed-Rand Saliency
Figure 6: Saliency maps for the the ORIGINAL and MIXED-RAND models on two images. As
expected, the MIXED-RAND model appears to place more importance on foreground pixels.
Table 3: Prediction categories we study for a given image-model pair. For a given image, a model
can make differing predictions based on the presence or absence of its foreground/background. We
label each possible case based on how the background classification relates to the original image
classification and the foreground classification. To proxy classifying full images, foregrounds, and
```
backgrounds separately, we classify on ORIGINAL, MIXED-RAND, and ONLY-BG-T (respectively).
```
“BG Irrelevant” demarcates images where the foreground classification result is identical to that of
```
the full image (in terms of correctness).
```
Label Correct Prediction Correct Prediction Correct Prediction
on Full Image on Foreground on Background
BG Required 3 7 3
BG Fools 7 3 7
BG+FG Required 3 7 7
BG+FG Fools 7 3 3
BG Irrelevant 3/7 3/7 —
4 Benchmark Progress and Background Dependence
In the previous sections, we demonstrated that standard image classification models exploit signals
from backgrounds. Considering that these models result from progress on standard computer
vision benchmarks, a natural question is: to what extent have improvements on image classification
benchmarks resulted from exploiting background correlations? And relatedly, how has model
robustness to misleading background signals evolved over time?
6
MIXED-RAND ORIGINAL ONLY-BG-T
0
1,000
2,000
3,000
4,000
Training dataset
Number of examples
BG Irrelevant
BG Required
BG Fools
BG+FG Required
BG+FG Fools
Figure 7: We categorize each test set image based on how a model classifies the full image, the
```
background alone, and the foreground alone (cf. Table 3). The model trained on ORIGINAL needs the
```
```
background for correct classification on 35% of images (measured by adding “BG Required” and
```
```
“BG+FG Required), while a model trained on MIXED-RAND is much less reliant on background. The
```
```
model trained on ONLY-BG-T requires the background most, as expected; however, the model often
```
misclassifies both the full image and the background, so the “BG Irrelevant” subset is still sizable.
0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9
0.2
0.4
0.6
0.8
ImageNet Accuracy
Accuracy on synthetic datasets
ONLY-BG-T
MIXED-NEXT
MIXED-RAND
MIXED-SAME
NO-FG
ResNet-50
MobileNet-v3s
Figure 8: Measuring progress on each of the synthetic ImageNet-9 datasets with respect to progress
on the standard ImageNet test set. Higher accuracy on ImageNet generally corresponds to higher
accuracy on each of the constructed datasets, but the rate at which accuracy grows varies based on
the types of features present in each dataset. Each pre-trained model corresponds to a vertical line on
the plot—we mark ResNet-50 and MobileNet-v3s models for reference.
As a first step towards answering these questions, we study the progress made by ImageNet models
on our synthetic IN-9 dataset variations. In Figure 8 we plot accuracy on our synthetic datasets
against ImageNet accuracy for each of the architectures considered. As evidenced by the lines of
best fit in Figure 8, accuracy increases on the original ImageNet benchmark generally correspond to
accuracy increases on all of the synthetic datasets. This includes the ONLY-BG datasets—indicating
that models do improve at extracting correlations from image backgrounds.
```
Indeed, the ONLY-BG trend observed in Figure 8 suggests that either (a) image classification models
```
```
can only attain their reported accuracies in the presence of background signals; or (b) these models
```
carry an implicit bias towards features in the background, as a result of optimization technique, model
```
class, etc.—in this case, we may need explicit regularization (e.g., through distributionally robust
```
```
optimization [Sag+20] or related techniques) to obtain models invariant to these background features.
```
Still, models’ relative improvement in accuracy across dataset variants is promising—models improve
```
on classifying ONLY-BG-T at a slower (absolute) rate than MIXED-RAND, MIXED-SAME and
```
```
MIXED-NEXT. Furthermore, the performance gap between the MIXED datasets and the others (most
```
```
notably, between MIXED-RAND and MIXED-SAME; between MIXED-NEXT and MIXED-RAND;
```
7
```
and consequently between MIXED-NEXT and MIXED-SAME) trends towards closing, indicating that
```
models not only are becoming better at using foreground features, but also are becoming more robust
```
to misleading background features (MIXED-RAND and MIXED-NEXT).
```
Overall, the accuracy trends observed from testing ImageNet models on our synthetic datasets reveal
```
that better models (a) are capable of exploiting background correlations, but (b) are increasingly robust
```
to changes in background, suggesting that invariance to background features may not necessarily
come at the cost of benchmark accuracy.
5 Related Work
We focus our discussion here on the works most closely related to ours, specifically those investigating
```
contextual bias or background dependence in computer vision (for a more extensive survey of and
```
```
explicit comparison to prior work, see Appendix E). Prior work has explored the more general
```
```
phenomenon of contextual bias [TE11; Kho+12; MTW12; SSF19], including studying its prevalence
```
and developing methods to mitigate it. For image backgrounds specifically, prior works show
that background correlations can be predictive [Tor03], and that backgrounds can influence model
```
decisions to varying degrees [Zha+07; RSG16; ZXY17; BHP18; RZT18; Bar+19; Sag+20]. The
```
work most similar to ours is that of Zhu, Xie, and Yuille [ZXY17], who also analyze ImageNet
```
classification (focusing on the older, AlexNet model). They find that the AlexNet model achieve
```
small but non-trivial test accuracy on a dataset similar to our ONLY-BG-B dataset. While sufficient
for establishing that backgrounds can be used for classification, this dataset also introduces biases
```
by adding large black rectangular patches to all of the images. In comparison to [ZXY17] (and the
```
```
prior works mentioned earlier): (a) we create a more extensive toolkit that allows us to measure
```
not just model performance without foregrounds but also the relative influence of foregrounds and
```
backgrounds on model predictions; (b) we control for the effect of image artifacts via the MIXED-
```
```
SAME dataset; (c) we study model robustness to adversarial backgrounds; (d) we study a larger and
```
```
more recent set of image classifiers [He+16; ZK16; TL19], and how improvements they give on
```
ImageNet relate to background correlations.
6 Discussion and Conclusion
In this work, we study the extent to which classifiers rely on image backgrounds. To this end, we
create a toolkit for measuring the precise role of background and foreground signal that involves
constructing new test datasets that contain different amounts of each. Through these datasets we
establish both the usefulness of background signal and the tendency of our models to depend on
backgrounds, even when relevant foreground features are present. Our results show that our models
are not robust to changes in the background, either in the adversarial case, or in the average case.
As most ImageNet images have human-recognizable foreground objects, our models appear to rely
on background more than humans on that dataset. The fact that models can be fooled by adversarial
background changes on 87.5% of all images highlights how poorly computer vision models may
perform in an out-of-distribution setting. However, contextual information like the background can
still be useful in certain settings. After all, humans do use backgrounds as context in visual processing,
and the background may be necessary if the foreground is blurry or distorted [Tor03]. Therefore,
reliance on background is a nuanced question that merits further study.
On one hand, our findings provide evidence that models succeed by using background correlations,
which may be undesirable in some applications. On the other hand, we find that advances in classifiers
have given rise to models that use foregrounds more effectively and are more robust to changes in the
background. To obtain even more robust models, we may want to draw inspiration from successes
```
in training on the MIXED-RAND dataset (a dataset designed to neutralize background signal—cf.
```
```
Table 1), related data-augmentation techniques [SSF19], and training algorithms like distributionally
```
robust optimization [Sag+20] and model-based robust learning [RHP20]. Overall, the toolkit and
findings in this work help us to better understand models and to monitor our progress toward the goal
of reliable machine learning.
8
Acknowledgements
Thanks to Kuo-An “Andy” Wei, John Cherian, and anonymous conference reviewers for helpful
comments on earlier versions of this work. The authors would also like to thank Guillaume LeClerc,
Sam Park, and Kuo-An Wei for help with data labeling. KX was supported by the NDSEG Fellowship.
LE was supported by the NSF Graduate Research Fellowship. AI was supported by the Open Phil
AI Fellowship. Work supported in part by the NSF grants CCF-1553428, CNS-1815221, and the
Microsoft Corporation. This material is based upon work supported by the Defense Advanced
```
Research Projects Agency (DARPA) under Contract No. HR001120C0015.
```
References
[Bak+18] Nicholas Baker et al. “Deep convolutional networks do not classify based on global
object shape.” In: PLOS Computational Biology. 2018.
[Bar+19] Andrei Barbu et al. “ObjectNet: A large-scale bias-controlled dataset for pushing the
limits of object recognition models”. In: Neural Information Processing Systems. 2019.
[BHP18] Sara Beery, Grant van Horn, and Pietro Perona. “Recognition in Terra Incognita”. In:
```
European Conference on Computer Vision (ECCV). 2018.
```
[CPT04] Antonio Criminisi, Patrick Pérez, and Kentaro Toyama. “Region Filling and Object
Removal by Exemplar-Based Image Inpainting”. In: IEEE Transactions on Image Pro-
cessing. 2004.
```
[Gei+19] Robert Geirhos et al. “ImageNet-trained CNNs are biased towards texture; increasing
```
shape bias improves accuracy and robustness.” In: International Conference on Learning
Representations. 2019.
[He+16] Kaiming He et al. “Deep Residual Learning for Image Recognition”. In: Conference on
```
Computer Vision and Pattern Recognition (CVPR). 2016.
```
[HGW01] Michael Harville, Gaile Gordon, and John Woodfill. “Foreground segmentation using
adaptive mixture models in color and depth”. In: IEEE Workshop on Detection and
Recognition of Events in Video. 2001.
[Ily+19] Andrew Ilyas et al. “Adversarial Examples Are Not Bugs, They Are Features”. In:
```
Neural Information Processing Systems (NeurIPS). 2019.
```
[Jac+19] Jorn-Henrik Jacobsen et al. “Excessive Invariance Causes Adversarial Vulnerability”. In:
```
International Contemporary on Learning Representations (ICLR). 2019.
```
[JLT18] Saumya Jetley, Nicholas Lord, and Philip Torr. “With friends like these, who needs
```
adversaries?” In: Advances in Neural Information Processing Systems (NeurIPS). 2018.
```
[Kho+12] Aditya Khosla et al. “Undoing the damage of dataset bias”. In: European Conference on
```
Computer Vision (ECCV). 2012.
```
[Mil95] George Miller. “WordNet: a lexical database for English”. In: Communications of the
ACM. 1995.
[MTW12] Choi Myung Jin, Antonio Torralba, and Alan S. Willsky. “Context models and out-of-
context objects”. In: Pattern Recognition Letters. 2012.
[RHP20] Alexander Robey, Hamed Hassani, and George J. Pappas. “Model-Based Robust Deep
Learning”. In: arXiv preprint arXiv:2005.10247. 2020.
[RKB04] Carsten Rother, Vladimir Kolmogorov, and Andrew Blake. “GrabCut: Interactive Fore-
ground Extraction Using Iterated Graph Cuts”. In: ACM Transactions on Graphics.
2004.
[RSG16] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. ““Why Should I Trust You?”:
Explaining the Predictions of Any Classifier”. In: International Conference on Knowl-
edge Discovery and Data Mining. 2016.
[RZT18] Amir Rosenfeld, Richard S. Zemel, and John K. Tsotsos. “The Elephant in the Room”.
```
In: arXiv preprint arXiv:1808.03305. 2018.
```
[Sag+20] Shiori Sagawa et al. “Distributionally Robust Neural Networks for Group Shifts: On
the Importance of Regularization for Worst-Case Generalization”. In: International
Conference on Learning Representations. 2020.
9
[SFS18] Rakshith Shetty, Mario Fritz, and Bernt Schiele. “Adversarial Scene Editing: Automatic
Object Removal from Weak Supervision”. In: Neural Information Processing Systems
```
(NeurIPS). 2018.
```
[SSF19] Rakshith Shetty, Bernt Schiele, and Mario Fritz. “Not Using the Car to See the Sidewalk–
Quantifying and Controlling the Effects of Context in Classification and Segmentation”.
```
In: Conference on Computer Vision and Pattern Recognition (CVPR). 2019.
```
[TE11] Antonio Torralba and Alexei Efros. “Unbiased look at dataset bias”. In: CVPR 2011.
2011, pp. 1521–1528.
[TL19] Mingxing Tan and Quoc V. Le. “EfficientNet: Rethinking Model Scaling for Convolu-
```
tional Neural Networks”. In: International Conference on Machine Learning (ICML).
```
2019.
[Tor03] Antonio Torralba. “Contextual Priming for Object Detection”. In: International Journal
```
of Computer Vision (IJCV). 2003.
```
[YS02] Andrew W. Yip and Pawan Sinha. “Contribution of Color to Face Recognition”. In:
Perception. 2002.
[Yu+18] Jiahui Yu et al. “Generative Image Inpainting with Contextual Attention”. In: Conference
```
on Computer Vision and Pattern Recognition (CVPR). 2018.
```
[Zha+07] Jianguo Zhang et al. “Local features and kernels for classification of texture and object
```
categories: A comprehensive study”. In: International Journal of Computer Vision
```
```
(IJCV). 2007.
```
[ZK16] Sergey Zagoruyko and Nikos Komodakis. “Wide Residual Networks”. In: British Ma-
```
chine Vision Conference (BMVC). 2016.
```
[ZXY17] Zhuotun Zhu, Lingxi Xie, and Alan Yuille. “Object Recognition without and without
Objects”. In: International Joint Conference on Artificial Intelligence. 2017.
10
A Datasets Details
We choose the following 9 high-level classes.
Class WordNet ID Number of
sub-classes
Dog n02084071 116
Bird n01503061 52
Vehicle n04576211 42
Reptile n01661091 36
Carnivore n02075296 35
Insect n02159955 27
Instrument n03800933 26
Primate n02469914 20
Fish n02512053 16
Table 4: The 9 classes of ImageNet-9.
All datasets used in the paper are balanced by randomly removing images from classes that are
over-represented. We only keep as many images as the smallest post-modification synthetic dataset,
```
so all synthetic datasets (except IN-9L) have the same number of images. We also use a custom GUI
```
to manually process the test set to improve data quality. For IN-9L, the only difference from using
the corresponding classes in the original ImageNet dataset is that we balance the dataset.
For all images: we apply the following filters before adding each image to our datasets.
• The image must have bounding box annotations.
• For simplicity, each image must have exactly one bounding box. A large majority of images
that have bounding box annotations satisfy this.
For images needing a properly segmented foreground: This includes the 3 MIXED datasets,
ONLY-FG, and NO-FG. We filter out images based on the following criteria.
• Because images are cropped before they are fed into models, we require that less than 50%
of the bounding box is removed by the crop, to ensure that the foreground still exists. Almost
all images pass this filter.
```
• The OpenCV function cv2.grabCut (used to extract the foreground shape) must work on
```
the image. We remove images where it fails.
• For the test set only, we manually remove images with foreground segmentations that retain
a significant portion of the background signal.
```
• For the test set only, we manually remove foreground segmentations that are very bad (e.g.
```
the segmentation selects part of the image, and that part doesn’t contain the foreground
```
object).
```
For images needing only background signal: This includes ONLY-BG-B and ONLY-BG-T. In this
case, we apply the following criteria:
```
• The bounding box must not be too big (more than 90% of the image). The intent here is to
```
avoid ONLY-BG-B images being just a large black rectangle.
• For the test set only, we manually remove ONLY-BG images that still have an instance of
the class even after removing the bounding box. This occurs when the bounding boxes are
```
imperfect or incomplete (e.g. only one of two dogs in an image is labeled with a bounding
```
```
box).
```
Creating the ONLY-BG-T dataset: We first make a “tiled” version of the background by finding
```
the largest rectangular strip (horizontal or vertical) outside the bounding box, and tiling the entire
```
image with that strip. We then replace the removed foreground with the tiled background. A visual
11
Figure 9: Visualization of how ONLY-BG-T is created.
example is provided in Figure 9. We purposefully choose not to use deep-learning-based inpainting
techniques such as [SFS18] to replace the removed foreground, as such methods could lead to biases
that the inpainting model has learned from the data. For example, an inpainting model may learn that
the best way to inpaint a missing chunk of a flower is to place an insect there, which is something we
want to avoid.
B Explaining the Decreased BG-GAP of pre-trained ImageNet models
We investigate two possible explanations for why pre-trained ImageNet models have a smaller
BG-GAP than models trained on ImageNet-9. Understanding this phenomenon can help inform
how models should be trained to be more background-robust. We find slight improvements to
background-robustness from training on more fine-grained classes, and even smaller improvements
from training on larger datasets. These two factors thus do not completely explain the increased
background-robustness of pre-trained ImageNet models—understanding this further is an interesting
empirical mystery to study.
B.1 The Effect of Fine-grainedness on the BG-GAP
One possible explanation is that training models to distinguish between finer-grained classes forces
them to focus more on the foreground, which contains relevant features for making those fine-grained
distinctions, than the background, which may be fairly similar across sub-classes of a high-level class.
This suggests that asking models to solve more fine-grained tasks could improve model robustness to
background changes.
To test the effect of fine-grainedness on ImageNet-9, we make a related dataset called IN-9LB that
uses the same 9 high-level classes and can be cleanly modified into more fine-grained versions.
Specifically, for IN-9LB we choose exactly 16 sub-classes for each high-level class, for a total of
144 ImageNet classes. To create successively more fine-grained versions of the IN-9LB dataset, we
```
group every n sub-classes together into a higher-level class, for n ∈ {1, 2, 4, 8, 16}. Here, n = 1
```
corresponds to keeping all 144 ImageNet classes as they are, while n = 16 corresponds to only having
9 high-level classes, like ImageNet-9. Because we keep all images from those original ImageNet
classes, this dataset is the same size as IN-9L.
We train models on IN-9LB at different levels of fine-grainedness and evaluate the BG-GAP of those
models in Figure 10. We find that fine-grained models have a smaller BG-GAP as well as better
performance on MIXED-NEXT, but the improvement is very slight and also comes at the cost of
decreased accuracy on ORIGINAL. The BG-GAP of the most fine-grained classifier is 2.3% smaller
than the BG-GAP of the most coarse-grained classifier, showing that fine-grainedness does improve
background-robustness. However, the improvement is still small compared to the size of the BG-GAP
```
(which is 13.3% for the fine-grained classifier).
```
B.2 The Effect of Larger Dataset Size on the BG-GAP
A second possible explanation for why pre-trained ImageNet models have a smaller BG-GAP is that
training on larger datasets is important for background-robustness. To evaluate this possibility, we
train models on different-sized subsets of IN-9LB. The largest dataset we train on is the full IN-9LB
dataset, which is 4 times as large as IN-9, and the smallest is 1/4 as large as IN-9. Figure 11 shows
12
that increasing the dataset size does increase overall performance but does not noticeably decrease
the BG-GAP.
9 18 36 72 144
```
Number of Training Classes (log scale)
```
50
60
70
80
90
100
Test Accuracy
Effects of Training on Fine-grained Data
Test Data
Mixed-Next
Mixed-Rand
Mixed-Same
Original
```
Figure 10: We train models on IN-9LB at different levels of fine-grainedness (more training classes
```
```
is more fine-grained). The BG-GAP, or the difference between the test accuracies on MIXED-SAME
```
and MIXED-RAND, decreases as we make the classification task more fine-grained, but the decrease
is small compared to the size of the BG-GAP.
0.25 0.50 1.00 2.00 4.00
Dataset size relative to ImageNet-9
30
40
50
60
70
80
90
Test Accuracy
Effects of Training on More Data
Test Data
Mixed-Next
Mixed-Rand
Mixed-Same
Original
Figure 11: We train models on different-sized subsets of IN-9LB. The largest training set we use
is the full IN-9LB dataset, which is 4 times larger than ImageNet-9. While performance on all test
datasets improves as the amount of training data increases, the BG-GAP has almost the same size
regardless of the amount of training data used.
```
It is possible that even more data for these classes (more than is available from ImageNet) is needed
```
```
to improve background-robustness, or that more training data (from other classes) is the cause of the
```
increased background-robustness of pre-trained ImageNet models. Understanding this further would
be an interesting direction.
B.3 Summary of methods investigated to reduce the BG-GAP
In Figure 12, we compare the BG-GAP of ResNet-50 models trained on different datasets and with
different methods to a ResNet-50 pre-trained on ImageNet. We explore `p-robust training, increasing
dataset size, and making the classification task more fine-grained, and find that none of these methods
13
Pre-trained ININ-9LB CoarseIN-9LB Fine
IN-9LIN-9
```
IN-9 (Mixed-Rand)IN-9 (L2-robust)IN-9 (Linf-robust)
```
```
Training Dataset (and Method)
```
0
20
40
60
80
100
Test Accuracy
Mixed-Same
Mixed-Rand
IN
IN-9LB
IN-9L
IN-9
Mixed-Same vs. Mixed-Rand Accuracy for Different Models
Figure 12: We compare various different methods of training models and measure their BG-GAP,
```
or the difference between MIXED-SAME and MIXED-RAND test accuracy. We find that (1) Pre-
```
```
trained IN models have surprisingly small BG-GAP. (2) Increasing fine-grainedness (IN-9LB Coarse
```
```
vs. IN-9LB Fine) and dataset size (IN-9 vs. IN-9L) decreases the BG-GAP only slightly. (3)
```
```
`p-robust training does not help. (4) Training on MIXED-RAND (cf. Section 3 appears to be the most
```
effective strategy for reducing the BG-GAP. For such a model, the MIXED-SAME and MIXED-RAND
accuracies are nearly identical.
reduces the BG-GAP as much as pre-training on ImageNet. The only method that reduces the
BG-GAP significantly more is training on MIXED-RAND. Furthermore, the same trends hold true if
we measure the difference between MIXED-SAME and MIXED-NEXT as opposed to the BG-GAP
```
(the difference between MIXED-SAME and MIXED-RAND).
```
C Training details
For all models, we use fairly standard training settings for ImageNet-style models. We train for 200
```
epochs using SGD with a batch size of 256, a learning rate of 0.1 (with learning rate drops every 50
```
```
epochs), a momentum parameter of 0.9, a weight decay of 1e−4, and data augmentation (random
```
```
resized crop, random horizontal flip, and color jitter). Unless specified, we always use a standard
```
ResNet-50 architecture [He+16]. For the experiment depicted in Figure 11, we found that using a
smaller learning rate of 0.01 was necessary for training to converge on the smaller training sets. Thus,
we used that same learning rate for all models in Figure 11.
D Additional Evaluation Results
We include full results of training models on every synthetic IN-9 variation and then testing them on
every synthetic IN-9 variation in Table 5. In addition to being more comprehensive, this table can
help answer a variety of questions, of which we provide two examples here.
How much information is leaked from the size of the foreground bounding box?
The scale of an object already gives signal correlated with the object class [Tor03]. Even though
they are designed to avoid having foreground signal, the background-only datasets ONLY-BG-B and
14
ONLY-BG-T may inadvertently leak information about object scale due to the bounding box sizes
being recognizable.
To gauge the extent of this leakage, we can measure how models trained on datasets where only the
```
foreground signal has useful correlation (MIXED-RAND or ONLY-FG) perform on the background-
```
only test sets. We find that there is small signal leakage from bounding box size alone—a model
trained on ONLY-FG achieves about 23% background-only test accuracy, suggesting that it is able
to exploit the signal leakage to some degree. A model trained on MIXED-RAND achieves about
15% background-only test accuracy, just slightly better than random, perhaps because it is harder for
```
models to measure (and thus, make use of) object scale when training on MIXED-RAND.
```
The existence of a small amount of information leakage in this case shows the importance of
```
comparing MIXED-SAME (as opposed to just ORIGINAL) with MIXED-RAND and MIXED-NEXT
```
```
when assessing model dependence on backgrounds. Indeed, the MIXED datasets may contain (1)
```
```
image processing artifacts, such as rough edges from the foreground processing, and (2) small traces
```
of the original background. This makes it important to control for both factors when measuring how
models react to varying background signal.
How does more training data affect model performance with and without object shape?
We already show closely related results on the effect of more training data on the BG-GAP in
Figure 11. Here, we compare model test performance on the NO-FG and ONLY-BG-B test sets. Both
replace the foreground with black, but only NO-FG retains the foreground shape.
```
By comparing the models trained on ORIGINAL and IN-9L (4x more training data), we find that
```
1. The ORIGINAL-trained model performs similarly on NO-FG and ONLY-BG-B, indicating
that it does not use object shape effectively.
2. The IN-9L-trained model performs about 13% better on NO-FG than ONLY-BG-B, showing
that it uses object shape more effectively.
Thus, this suggests that more training data may allow models to learn to use object shape more
effectively. Understanding this phenomena further could help inform model training and dataset
collection if the goal is to train models that are able to leverage shape effectively.
Trained on Test DatasetMIXED-NEXT MIXED-RAND MIXED-SAME NO-FG ONLY-BG-B ONLY-BG-T ONLY-FG ORIGINAL IN-9L
MIXED-NEXT 78.07 53.28 48.49 16.20 11.19 8.22 59.60 52.32 46.44MIXED-RAND 71.09 71.53 71.33 26.72 15.33 14.62 74.89 73.23 67.53
MIXED-SAME 45.41 51.36 74.40 39.85 35.19 41.58 61.65 75.01 69.21NO-FG 13.70 18.74 42.79 70.91 36.79 42.52 31.48 48.94 47.62
ONLY-BG-B 10.35 15.41 38.37 37.85 54.30 42.54 21.38 42.10 41.01ONLY-BG-T 11.48 17.09 45.80 40.84 38.49 50.25 19.19 49.06 47.94
ONLY-FG 33.04 35.88 47.63 27.90 23.58 22.59 84.20 54.62 51.50ORIGINAL 48.77 53.58 73.80 42.22 32.94 40.54 63.23 85.95 80.38
IN-9L 71.21 75.60 89.90 55.78 34.02 43.60 84.12 96.32 94.61
Table 5: The test accuracies, in percentages, of models trained on all variants of ImageNet-9.
What about other ways of modifying the background signal?
One can modify the background in various other ways—for example, instead of replacing the
background with black as in ONLY-FG, the background can be blurred as in the BG-BLURRED
image of Figure 13. As expected, blurred backgrounds are still slightly correlated with the correct
class. Thus, test accuracies for standard models on this dataset are higher than on ONLY-FG, but
```
lower than on MIXED-SAME (which has signal from random class-aligned backgrounds that are
```
```
not blurred). While we do not investigate all possible methods of modifying background signal,
```
we believe that the variations we do examine in ImageNet-9 already improve our understanding of
how background signals matter. Investigating other variations could provide an even more nuanced
understanding of what parts of the background are most important.
E Additional Related Works and Explicit Comparisons
There has been prior work on mitigating contextual bias in image classification, the influence of
background signals on various datasets, and techniques like foreground segmentation that we leverage.
15
Original Only-FG BG-Blurred
```
Figure 13: Backgrounds can also be modified in other ways; for example, it can be blurred. Our
```
evaluations on this dataset show similar results.
Mitigating Contextual Bias: [Kho+12] focuses on mitigating dataset-specific contextual bias and
proposes learning SVMs with both general weights and dataset-specific weights, while [MTW12]
creates an out-of-context detection task with 209 out-of-context images and suggests using graphical
models to solve it. [SSF19] focuses on the role of co-occurring objects as context in the MS-COCO
```
dataset, and uses object removal to show that (a) models can still predict a removed object when only
```
```
co-occurring objects are shown, and (b) special data-augmentation can mitigate this.
```
Understanding the influence of backgrounds: For contextual bias from image backgrounds specif-
ically, prior works have observed that the background of an image can influence model decisions to
```
varying degrees. In particular, [Zha+07] find that (a) a bag-of-features object detection algorithm
```
```
depends on image backgrounds in the PASCAL dataset and (b) using this algorithm on a training set
```
```
with varying backgrounds leads to better generalization. [BHP18; Bar+19] collect new test datasets
```
of animals and objects, respectively. [Bar+19] focus on object classes that also exist in ImageNet,
and their new test set contains objects photographed in front of unconventional backgrounds and
in unfamiliar orientations. Both works show that computer vision models experience significant
accuracy drops when trained on data with one set of backgrounds and tested on data with another.
[Sag+20] create a small synthetic dataset of Waterbirds, where waterbirds and landbirds from one
dataset are combined with water and land backgrounds from another. They show that a model’s
reliance on spurious correlations with the background can be harmful for small subgroups of data
```
where those spurious correlations no longer hold (e.g. landbirds on water backgrounds). Furthermore,
```
they propose using distributionally robust optimization to reduce reliance on spurious correlations
with the background, but their method assumes that the spurious correlation can be precisely specified
```
in advance. [RZT18] analyzes background dependence for object detection (as opposed to classifica-
```
```
tion) models on the MS-COCO dataset. They transplant an object from one image to another image,
```
and find that object detection models may detect the transplanted object differently depending on its
location, and that the transplanted object may also cause mispredictions on other objects in the image.
Explicit Comparison to Prior Works: In comparison to prior works, our work contributes the
following.
• We develop a toolkit for analyzing the background dependence of ImageNet classifiers, the
most common benchmark for computer vision progress. Only [ZXY17], which we compare
to in Section 5, also focuses on ImageNet.
• The test datasets we create separate and mix foreground and background signals in various
```
ways (cf. Table 1), allowing us to study the sensitivity of models to these signals in a more
```
fine-grained manner.
• Our toolkit for separating foreground and background can be applied without human-
annotated foreground segmentation, which prior works on MS-COCO and Waterbirds rely
on. This is important because foreground segmentation annotations are hard to collect and
do not exist for ImageNet.
• We study the extent of background dependence in the extreme case of adversarial back-
grounds.
• We focus on better vision models, including ResNet [He+16], Wide ResNet[ZK16], and
EfficientNet [TL19].
16
• We evaluate how improvements on the ImageNet benchmark have affected background
```
dependence (cf. Section 4).
```
Foreground Segmentation and Image Inpainting: In order to create IN-9 and its variants, we
rely on OpenCV’s implementation of the foreground segmentation algorithm GrabCut [RKB04].
Foreground segmentation is a branch of computer vision that seeks to automatically extract the
foreground from an image [HGW01]. After finding the foreground, we remove it and simply replace
the foreground with copies of parts of the background. Other works solve this problem, called image
```
inpainting, either using exemplar-based methods [CPT04] or using deep learning [Yu+18; SFS18].
```
[SFS18] both detects the foreground for removal and inpaints the removed region. However, more
advanced inpaintings techniques can be slow and inaccurate when the region that must be inpainted
is relatively large [SFS18], which is the case for many ImageNet images. Exploring better ways
of segementing the foreground and inpainting the removed foreground could improve our analysis
toolkit further.
17
F Additional examples of synthetic datasets
We randomly sample an image from each class, and display all synthetic variations of that image, as
```
well as the predictions of a pre-trained ResNet-50 (trained on IN-9L) on each variant.
```
dog
Original
vehicle
Only-BG-B
vehicle
Only-BG-T
dog
No-FG
dog
Only-FG
dog
Mixed-Same
dog
Mixed-Rand
dog
Mixed-Next
Figure 14: ImageNet-9 variations—Dog.
bird
Original
fish
Only-BG-B
bird
Only-BG-T
bird
No-FG
bird
Only-FG
bird
Mixed-Same
bird
Mixed-Rand
bird
Mixed-Next
Figure 15: ImageNet-9 variations—Bird.
18
vehicle
Original
vehicle
Only-BG-B
vehicle
Only-BG-T
vehicle
No-FG
instrument
Only-FG
vehicle
Mixed-Same
vehicle
Mixed-Rand
instrument
Mixed-Next
Figure 16: ImageNet-9 variations—Vehicle.
reptile
Original
bird
Only-BG-B
dog
Only-BG-T
bird
No-FG
reptile
Only-FG
reptile
Mixed-Same
reptile
Mixed-Rand
reptile
Mixed-Next
Figure 17: ImageNet-9 variations—Reptile.
19
carnivore
Original
primate
Only-BG-B
bird
Only-BG-T
bird
No-FG
carnivore
Only-FG
carnivore
Mixed-Same
carnivore
Mixed-Rand
carnivore
Mixed-Next
Figure 18: ImageNet-9 variations—Carnivore.
instrument
Original
bird
Only-BG-B
bird
Only-BG-T
insect
No-FG
instrument
Only-FG
instrument
Mixed-Same
bird
Mixed-Rand
bird
Mixed-Next
Figure 19: ImageNet-9 variations—Instrument.
20
primate
Original
instrument
Only-BG-B
carnivore
Only-BG-T
dog
No-FG
dog
Only-FG
primate
Mixed-Same
primate
Mixed-Rand
dog
Mixed-Next
Figure 20: ImageNet-9 variations—Primate.
fish
Original
insect
Only-BG-B
fish
Only-BG-T
dog
No-FG
fish
Only-FG
fish
Mixed-Same
fish
Mixed-Rand
fish
Mixed-Next
Figure 21: ImageNet-9 variations—Fish.
21
G Adversarial Backgrounds
We include the 5 most fooling backgrounds for all classes, the fool rate for each of those 5 back-
```
grounds, and the total fool rate across all backgrounds from that class (on the left of each row) here.
```
dog, 61.77%
17% 18% 19% 20% 21%
Figure 22: Most adversarial backgrounds—Dog.
bird, 64.22%
32% 34% 35% 36% 38%
Figure 23: Most adversarial backgrounds—Bird.
vehicle, 67.47%
41% 41% 43% 44% 47%
Figure 24: Most adversarial backgrounds—Vehicle.
22
reptile, 54.94%
22% 22% 23% 25% 28%
Figure 25: Most adversarial backgrounds—Reptile.
carnivore, 53.11%
17% 18% 20% 21% 28%
Figure 26: Most adversarial backgrounds—Carnivore.
instrument, 73.19%
45% 46% 46% 46% 47%
Figure 27: Most adversarial backgrounds—Instrument.
primate, 63.83%
25% 27% 28% 33% 39%
Figure 28: Most adversarial backgrounds—Primate.
fish, 69.69%
43% 44% 44% 44% 46%
Figure 29: Most adversarial backgrounds—Fish.
23
H Examples of Fooling Backgrounds in Unmodified Images
We visualize examples of images where the background of the full original image actually fools
models in Figure 30. For these images, models classify the foreground alone correctly, but they
predict the same wrong class on the full image and the background. We denote these images as “BG
```
Fools” in Table 3 and Figure 7. While this category is relatively rare (accounting for just 3% of the
```
```
ORIGINAL-trained model’s predictions), they reveal a subset of original images where background
```
signal hurts classifier performance. Qualitatively, we observe that these images all have confusing or
misleading backgrounds.
insect
bird
Original Classification:
With Random Background:
instrument
vehicle
fish
reptile
fish
insect
vehicle
instrument
```
Figure 30: Images that are incorrectly classified (as the class on the top row, which is the same class
```
```
that their background alone from ONLY-BG-T is classified as), but are correctly classified (as the
```
```
class on the bottom row) when the background is randomized. Note that these images have confusing
```
backgrounds that could be associated with another class.
24