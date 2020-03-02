# NUT-Carcinoma-Pathology-Classification
Machine learning image classification to attempt to detect NUT Midline Carcinoma in cancer pathology images.

## Gathering Data
I had to gather data in order to attempt this classification task. As far as I know, there is no dataset for NMC (NUT Midline Carcinoma) images. I gathered as many unique images from Google Images as I could find to have as my target class. NMC is an extremely rare cancer -- many doctors go through their careers without seeing this diagnosis. Because of this, there are not many images and not a lot of data available yet, so I had to work with what I could find.

For binary classification, I needed to gather cancer pathology images that were NOT NMC pathology as well. This was easier as there is a lot more pathology images for other types of cancer.

Initially, I had at least 35 images of NMC pathology. This is incredibly small amount of data for a machine learning task. To attempt to combat the fact that there is little data, and to hopefully improve results, I ran a series of image augmentations over all images gathered for this task, producing 450+ images for both classes (1 for NMC, 0 for non-NMC images).

This has substantially improved results.

## Results
Multiple machine learning algorithms were used for this project to compare the results and performance of each. Different hyperparameters were chosen via grid search with cross validation using 5 folds.

Accuracy and confusion matrices were used to examine performance of each classifier. Confusion matrices are very important to understand what is happening with classification of images, especially when it comes to cancer. It is important to understand the false positives and false negatives, as you might have a desire to minimize one more over the other. This is especially the case for cancer diagnosis. For example, would you rather have more false positives, or more false negatives? This use of machine learning has very critical repercussions if the results are not evaluated and used properly.

### Support Vector Machine
<b>Test Accuracy:</b> 0.94048
Confusion Matrix:
[[116 8]
 [7 121]]

### XGBoost
<b>Test Accuracy:</b> 0.92857
Confusion Matrix:
[[118 6]
 [12 116]]

### Decision Tree
<b>Test Accuracy:</b> 0.87302
Confusion Matrix:
[[109 15]
 [17 111]]

### Naive Bayes
<b>Test Accuracy:</b> 0.54365
Confusion Matrix:
[[119 5]
 [110 18]]

### Random Forest
<b>Test Accuracy:</b> 0.90873
Confusion Matrix:
[[115 9]
 [14 114]]

### Gradient Boost
<b>Test Accuracy:</b> 0.94444
Confusion Matrix:
[[117 7]
 [7 121]]

### Multi-layer Perceptron (Neural Net)
<b>Test Accuracy:</b> 0.81746
Confusion Matrix:
[[116 8]
 [38 90]]


## Motivation
The motivation for this little project was for my best friend Brandon Gordon. I grew up with Brandon since we were in 5th grade, and I am now 24 as of creating this repository -- so we've known each other almost our entire lives. Brandon was sent to the hospital with what seemed to be Pneumonia at first. After some time in the hospital and more tests, tumors were found in his body, and instead of his problem being Pneumonia, he was diagnosed with NUT Midline Carcinoma.

Brandon's cancer progressed incredibly fast. On top of being one of the rarest forms of cancer, NMC is also one of the most aggressive. The average life expectancy after diagnosis, at the time, was two months. A Harvard professor whom specializes in NMC was consulted to assist with the doctors that were working with Brandon to better understand what this cancer is and what the options were.

Over a short period of time, the cancer progressed incredibly fast, despite treatments and testing. Brandon was on multiple painkillers for the pain caused by the cancer, increasing dosage what seemed to be daily. Visiting him, I was able to see the effect this illness had on his body and his mental state. It is something I wouldn't wish on anyone.

August 14th, 2018, the decision was made to end treatment after no sign of improvement, and we were thankfully able to say our goodbyes as he made his peace. On the morning of August 15th, 2018 at 1:10am, Brandon passed away in his sleep. He was 22.

A couple of months prior to this event entering our lives, Brandon, myself, and another lifelong friend were all together after all of us being in different states. We were having great time creating great memories. The thought of one of us getting a rare cancer and dying within months of spending time together was unfathomable, yet, life manages to bring about the most unfortunate of surprises.

My hope is that this rare cancer can have some more light shed on it, and more information gathered to help treat and increase life expectancy of future patients.