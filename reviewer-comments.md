---
title: Reviewer Comments
author: Allison Koh
---

## Reviewer 2 

This paper presents an integrated BERT and CNN model to predict policy domains from party manifestos. The authors use base BERT to represent raw text and add CNN or GRU structure to predict the policy domains. Experiments show that BERT-CNN outperforms baselines, including BERT-GRU to identify the eight major categories, but SVM outperforms BERT-CNN and BERT-GRU on minor categories. The authors do more analysis on why BERT-CNN shows poor performance, over-fitting.

The main strength of this paper is suggesting an appropriate BERT based model for predicting policy domains from party manifestos. The authors show the performance of the suggested model on major categories and minor categories, and they describe the reason of poor performance for the future directions.

What concerns me most is the lack of takeaway messages for NLP community members from this paper. Using BERT and combining CNN or GRU is a well-known methodology for document classification. Policy domain prediction is a new topic because the corpus appeared in 2019, but it is nothing special in terms of the classification task. And overfitting is also a well-known problem in machine learning models. It would be better to suggest novel ideas to overcome the overfitting problem with a special characteristic in the data (manifesto corpus) or problem (predicting policy domain).

Additionally, can political scientists use the suggested model to replace the hand-labeling political texts? I think the authors need to tone down the claims in the Abstract.

Additional questions and comments
What is the meaning of the numbers in Figure 5?
How about doing 10 fold cross-validation to show the difference between models’ performance statistically?

## Reviewer 3

The authors present a study of application of several methods to classifying political texts. They consider 2 levels of classes, major and minor, and compare to several baselines.
The paper is written in somewhat confusing manner, and the proposed methods are not really well explained in the context of their problem area (neither BERT nor CNNs, see comments below). In addition, the experimental section can be significantly improved, with more interesting experiments and discussions. The authors claim that their study is comprehensive, but that does not seem to be the case at all. Detailed comments follow:
- Section 3: seven policy domains, and then 8? Also 57 categories and then 58? Please be precise.
- Please add some examples of 57 categories to help readers understand the problem at hand.
- No need to explain what BERT means several times.
- Fig 2 should be explained much better, dimensions are not labeled.
- Which pre-trained BERT was exactly used?
- "Multinomial Naive Bayes", reference missing, and the explanation should be improved (not clear what exactly it does).
- Not clear how SVM was used. In general the baselines are quite poorly explained.
- The experimental setting is a bit confusing. E.g., I'm assuming that you removed uncategorized sentences during training, but this should be mentioned. Also, how large was the additional linear layer? In general the explanations in the paper can be improved.
- Tables 3 and 4 are identical.
- Seems that SVM is the best at the end for minor classes, no method outperformed it?
- The experimental section is not well organized, e.g., why spend so much space on CNN baseline and not the others? Although CNN is mentioned in the title, from the text it does not seem like that is your contribution.
- "We posit that potential improvements on these issues ...", this is just speculation, and should be analyzed.
- The analysis of the results could be improved. E.g., good results on "freedom and democracy" are ideal for deeper, interesting analysis. This is however not provided in the paper.
- While the authors claim that the analysis is comprehensive, the results are missing many interesting analyses and experiments as mentioned above.

## Edits based on reviewer comments 

### Writing edits 

- Suggesting novel ideas to overcome the overfitting problem with a special characteristic in the data (manifesto corpus) or problem (predicting policy domain)
- What is the meaning of the numbers in Figure 5?

### Experiments 

- How about doing 10 fold cross-validation to show the difference between models’ performance statistically?


