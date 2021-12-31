# Hand-written-digit-recognition-analysis_MNIST-
Hand written digit recoginition using Neural nets, Support Vector machine, and K-Nearest Neighbor Algorithms 


A study on factors that contribute to the performance and efficiency of selected classifiers on handwritten digit dataset (MNIST).

Students:

Alfred Ntiamoah (002317287)

Aida ArjmandMoghadam (002292217)

Anjeeth Pai (002308550)

Abstract

Handwritten character recognition has been one of the challenges in the field of pattern recognition which still needs improvement. Several research works have already been done on area with different types of classifiers such as K-Nearest Neighbor (KNN), Artificial Neural Network (ANN), Support Vector Machines (SVM), Random forest etc. Most of these works concentrated on improving the performance of the classifiers and about 97% accuracy has been achieved yet still not adequate for real world applications. The focus of this research work is to help identify the most significant factors that contributed to the performance of these classifiers. The work will be concentrated on K-NN due to its high accuracy in digits’ prediction and training efficiency [1], ANN due to its high accuracy with its variant convolutional neural network (CNN) scoring about 98.8% [2] and SVM which has very high recognition rate but varying performance [3]. We hope to discover other subtle factors that affect these classifiers negatively in terms of performance (speed and accuracy) that has been overlooked.

Goal: The objective of this research work is to identify the key contributors to the performance and efficiency of the selected classifiers thus to answer the question what influenced this classifier to achieve high performance, accuracy and (or) efficiency than other classifiers.

Algorithms: K-Nearest Neighbor, Convolutional Neural Network, and Support Vector Machine will be considered for this research project. Existing methods for evaluating performance on classifiers such as bootstrap, random sampling and cross-validation [4] will be employed. F1 scores and ROC curves will be the main tool for validation. This is to ensure that models achieve good scores for both training and test set – optimum performance and efficiency. Parameter tuning techniques will also be used for parameterized models to improve performance if required.Methodology: • Tools and dataset Yann LeCun's version of MNIST dataset, which contains 7000 examples of handwritten digits will be use for the studies[5]. A lot of preprocessing such as normalization, anti-aliasing, image cropping and centering, and translation techniques has already been applied to the data. Implementation will be carried out with the Scikit-learn open source machine learning library. This library supports both supervised and unsupervised learning and also provides tools for model fitting, data pre-processing, model selection and evaluation and many other toolsets.[6] The library is written in python language hence; we will use python language to perform this research.

• Classifiers and model analysis For each classifier, two different models will be built which can be achieved by modifying either the dataset or specifying different parameters. Confidence interval will be estimated for each model’s accuracy and the best will be selected to represent the classifier. Further analysis will be performed on the selected model to identify the various contributing factors that are affecting the performance and accuracy of the algorithm. These factors can be a feature(s) of the dataset such as size, color (grayscale), orientation of digit image or parameter(s) specified during model training. The MNIST data contains 60,000 training samples and 10,000 test samples by default [5] which means 60,000 samples of the training set will be used to train the models while 10,000 samples will be used for testing. A new dataset that do not conform to the MNIST dataset specifications will be used to perform final validation on the classifiers. Images will be resized and then tilted to create a new test set that will be used to validate the classifiers. Also Gaussian noise [] will also be introduced to create a new test set which will also be used for validation. Results analysis and remarks: F1 scores will be calculated for the classifiers after the final test. From the results, we will conclude on the effects of image size, orientation and the introduction of background noise have on the various classifiers. Also, analysis on parameter values will also be concluded based on the F1 score. ROC curves will be constructed for visual analysis.

References:

Shengfeng C. et al, offline handwritten digits’ recognition using machine learning, IEOM Society International, Sept. 2018
Ankita M., Singh D. S., Handwritten digit recognition using neural network approaches and feature extraction techniques. A survey, Depart. Of CSE, MMUT Gorakhpur, 2016
Wan Z., Classification of MNIST handwritten digit database using neural network, Research School of Computer Science - Australian National University, 2018
Tan, Pang-ning, Introduction to data mining, ISBN: 9783642197208, Intelligent Systems Reference Library, 2011
Yann L., Corinna C. and Christopher j., The MNIST database of handwritten digits, url: http://yann.lecun.com/exdb/mnist/, last accessed: October 28, 2021
F. Pedregosa et. al, Scikit-learn: Machine Learning in Python, Journals of Machine Learning Research, 2011
[1]https://wiki.pathmind.com/convolutional-network [2]https://github.com/GoogleCloudPlatform/trainingdata-analyst [3]https://stackoverflow.com/questions/40727793/howto-convert-a-grayscale-image-into-a-list-of-pixelvalues [4]https://github.com/suvoooo/MNIST_digit_classify/tr ee/description [5]https://keras.io/api/datasets/mnist/ [6]https://scikitlearn.org/stable/modules/generated/sklearn.svm.SVC.h tml [7]Yevhen C. etal., Handwritten Digits Recognition Using SVM, KNN, RF and Deep Learning Neural Networks, CEUR-WS, Vol-2864 [8]https://github.com/pramodini18/Digit-recognitionusingSVM/blob/master/Handwritten%20digit%20recognitio n%20using%20SVM.ipynb [9]Scholkopf B. etal., Comparing Support Vector Machines with Gaussian Kernels to Radial Basis Function Classifiers, IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 45, NO. 11, NOVEMBER 1997 [10] Yann L., Corinna C. and Christopher j., The MNIST database of handwritten digits, url: http://yann.lecun.com/exdb/mnist/ [11] F. Pedregosa et. al, Scikit-learn: Machine Learning in Python, Journals of Machine Learning Research, 2011 [12] Zhengyi Ma. 2018. Methods Of Classification Implemented In MNIST Recognition. In Woodstock ’18: ACM Symposium on Neural Gaze Detection, June 03–05, 2018, Woodstock, NY . ACM, New York, NY, USA, 3 pages. https://doi.org/10.1145/1122445.1122456 | https://github.com/zhengyima/mnist-classification