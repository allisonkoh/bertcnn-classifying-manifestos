---
abstract: |
  Hand-labeled political texts are often required in empirical studies
  on party systems, coalition building, agenda setting, and many other
  areas in political science research. While hand-labeling remains the
  standard procedure for analyzing political texts, it can be slow and
  expensive, and subject to human error and disagreement. Recent studies
  in the field have leveraged supervised machine learning techniques to
  automate the labeling process of electoral programs, debate motions,
  and other relevant documents. We build on current approaches to label
  shorter texts and phrases in party manifestos using a pre-existing
  coding scheme developed by political scientists for classifying texts
  by policy domain and policy preference. Using labels and data compiled
  by the Manifesto Project, we make use of the state-of-the-art
  Bidirectional Encoder Representations from Transformers (BERT) in
  conjunction with Convolutional Neural Networks (CNN) and Gated
  Recurrent Units (GRU) to seek out the best model architecture for
  policy domain and policy preference classification. We find that our
  proposed BERT-CNN model outperforms other approaches for the task of
  classifying statements from English language party manifestos by major
  policy domain.
author:
- |
  Allison Koh\
  Hertie School of Governance\
  Friedrichstraße 180\
  10117 Berlin, Germany\
  `koh@hertie-school.org`\
  Daniel Boey\
  Hertie School of Governance\
  Friedrichstraße 180\
  10117 Berlin, Germany\
  `danielboeyks@gmail.com`\
  Hannah Bechara\
  Hertie School of Governance\
  Friedrichstraße 180\
  10117 Berlin, Germany\
  `bechara@hertie-school.org`\
bibliography:
- coling2020.bib
title: Predicting Policy Domains from Party Manifestos with BERT and
  Convolutional Neural Networks
---

Introduction {#intro}
============

During campaigns, political actors communicate their position on a range
of key issues to signal campaign promises and gain favor with
constituents. Whilst identifying the political positions of political
actors provides no certainty with regards to whether they act upon their
policy preferences, it remains essential to understanding their intended
political actions. This is why policy preferences---or positions on
specific policy issues expressed in speech or text---have been
extensively analyzed within the relevant political science literature
 [@abercrombie2019policy; @budge2001mapping; @lowe2011scaling; @volkens2013mapping].
Methods employed to investigate the policy preferences of political
actors include analysis of roll call voting, position extraction from
elite studies or regular surveys, expert surveys and hand-coded analysis
and computerized text analysis  [@debus2009estimating]. Studies that
utilize political manifestos, electoral speeches, and debate motions
often rely on the availability of machine-readable documents that are
labeled by policy domain or policy preference.

Quantitative methods, especially in the field of natural language
processing, have enabled the development of more scalable methods for
predicting policy preferences. These advancements have enabled political
scientists to analyze political texts and estimate their positions over
time  [@nanni2016topfish; @zirn2016classifying]. To better understand
the political positions of political actors, many social science
researchers have turned to hand-labeling political documents, such as
parliamentary debate motions and party manifestos. Much of the previous
work on analyzing political texts relies on hand-labeling documents
[@abercrombie2018sentiment; @gilardi2009learning; @krause2011policy; @simmons2004globalization].
Yet, the analysis of political documents in this field stands to benefit
from automating the coding of texts using supervised machine learning.
Most recently, neural networks and deep language representation models
have been employed in state-of-the-art approaches to automatic labeling
of political texts by policy preference.

In this paper, we present a deep learning approach to classifying
labeled texts and phrases in party manifestos, using the coding scheme
and documents from Manifesto Project  [@volkens2019manifesto]. We use
English-language texts from the Manifesto Project Corpus, which divides
party manifestos into statements---or *quasi-sentences*---that do not
span more than one grammatical sentence. Based on the state-of-the-art
deep learning methods for text classification, we propose using
Bidirectional Encoder Representations from Transformers (BERT) combined
with neural networks to automate the task of labeling political texts.
We compare our models that combine [Bert]{.smallcaps} and neural
networks against previous experiments with similar architectures to
establish that our proposed method outperforms other approaches commonly
used in natural language processing research when it comes to choosing
the correct policy domain and policy preference. We identify differences
in performance across policy domains, paving the way for future work on
improving deep learning models for classifying political texts. To the
best of our knowledge, we offer the most comprehensive application of
deep language representation models incorporated with neural networks
for document classification of political manifesto statements.

The rest of this paper is structured as follows. In Section
[2](#related_work){reference-type="ref" reference="related_work"}, we
provide a brief overview of the current state-of-the-art in
classification of political texts, focusing mainly on detecting policy
domains and preference. Section [3](#data){reference-type="ref"
reference="data"} describes the data, going into detail about the
Manifesto Project Corpus. Section [4](#methodology){reference-type="ref"
reference="methodology"} then introduces our classification approach and
provides important details of our models and evaluation approach. In
Section [5](#results){reference-type="ref" reference="results"}, we
present our results and address some limitations of our system. Finally,
Section [7](#conclusion){reference-type="ref" reference="conclusion"}
concludes our findings and sets up a roadmap for future improvements.

Related Work {#related_work}
============

Several studies have concentrated on building scaling models that
identify the political position of texts
 [@glavavs2017cross; @laver2003extracting; @nanni2019political; @proksch2010position].
Previously, most of the seminal work in this area has overlooked the
task of classifying texts by topic or policy area prior to detecting
policy preferences associated with the topic. Over the past couple of
years, several studies have addressed the gap in *opinion-topic
identification* by classifying text data from political speeches,
manifestos, and other documents by topic before predicting policy
preference. Perhaps most relevant to our research is the paper by , in
which the authors trained and validated an approach to classifying
manifestos from the United States into seven policy domains that
involved binary classifiers predicting whether sentences that are
adjacent to one another belong to the same topic[^1]. Their proposed
approach of optimizing predictions using a Markov Logic framework
yielded an average micro-F1 score of .749. introduced a multi-lingual
classifier for automatically labeling texts by policy domain. For
classification of 20,196 English-language manifestos by policy domain,
their CNN models yielded an average micro-F1 score of .59.

More recently, studies have employed neural networks and deep language
representation models to address the computationally intensive task of
classifying political texts into over thirty categories. To take on this
ambitious task, included contextual information about individual
quasi-sentences, specifically political party and the previous sentence
within a manifesto, into multi-scale convolutional neural networks with
word embeddings. Their best performing model for classifying 86,500
quasi-sentences from the Manifesto Project Corpus into the seven major
policy domains yielded an F1 score of .6532, and their best performing
model for classifying quasi-sentences by policy preference yielded an F1
score of .4273. propose employing a hierarchical sequential deep model
that captures information from within manifestos as well as contextual
information across manifestos to predict the political position of
texts. Their best performing hierarchical modeling approach for
classifying 86,603 English language quasi-sentences yielded an F1 score
of .50.

used deep language representation models to detect the policy positions
of Members of Parliament in the United Kingdom. Using motions and
manifestos as data sources, the authors employed a variety of methods to
predict the policy and domain labels of texts. They propose utilizing
Bidirectional Encoder Representations from Transformers
([Bert]{.smallcaps}), with results fine-tuned with party manifestos and
the motions themselves. In addition to a final softmax layer, the
authors added a CNN model and max-pooling layers to the soft-max layer.
they found that the use of [Bert]{.smallcaps} demonstrated
state-of-the-art performance on both manifestos and motions via
supervised pipelines with a Macro-F1 score of 0.69 for their best
performing model. Our work builds on some of the methods proposed in
their paper, leveraging neural networks and deep language representation
models for classifying political texts.

The Manifesto Project Corpus {#data}
============================

The Manifesto Project Corpus[^2]  [@volkens2019manifesto] provides
information on policy preferences of political parties from seven
different countries based on a coding scheme of seven policy domains,
under which 57 policy preference codes are manually coded. The Manifesto
Project offers data that divides party manifestos into quasi-sentences,
or individual statements which do not span more than one grammatical
sentence. Quasi-sentences are then individually assigned to categories
pertaining to policy domain and preference. The 57 policy preference
codes, one of which is "not categorized", refer to the
position---positive or negative---of a party regarding a particular
policy area. The 57 policy preference codes fall into a macro-level
coding scheme comprising of 8 policy domain categories. In political
science research, the Manifesto Project Corpus is particularly useful
for studying party competition, the responsiveness of political parties
to constituent preferences, and estimating the ideological position of
political elites. While the official classification of manifestos in
this dataset has primarily relied on human coders, the investigation of
automatically detecting policy positions of the text data is valuable
for scaling up the classification of large volumes of political texts
available for analysis.

Our final subset of all English-language manifestos comprises of 99,681
quasi-sentences. Table [\[tbl:desc\]](#tbl:desc){reference-type="ref"
reference="tbl:desc"} illustrates the distribution of English-language
manifestos across countries and policy domains. To ensure that the ratio
between policy domains remains consistent across policy domains in
running our models, we applied a 70/15/15 split between training,
validation, and test sets separately for the 8 major categories and the
58 minor categories.

Experimental Setup {#methodology}
==================

Bidirectional Encoder Representations from Transformers ([Bert]{.smallcaps})
----------------------------------------------------------------------------

Bidirectional Encoder Representations from Transformers
([Bert]{.smallcaps}) have proven successful in prior attempts to
classify phrases and short texts [@devlin2018bert] .
[Bert]{.smallcaps}'s key innovation lies in its ability to apply
bidirectional training of transformers to language modelling. This
state-of-the-art deep language representation model uses a "masked
language model", enabling it to overcome restrictions caused by the
unidirectional constraint.

Our experiments use the standard pre-trained [Bert]{.smallcaps}
transformers as the embedding layer in our model. Since
[Bert]{.smallcaps} is trained on sequences with a maximum lengths of 512
tokens, all quasi-sentences with more than 510 words were trimmed to fit
this requirement. Pre-trained embeddings were frozen and not trained for
the base models. We test two variants of [Bert]{.smallcaps}---one
incorporating a bidirectional GRU model, and another incorporating CNNs.
Model specifications and training times for our neural networks and deep
language representation models are shown in Table
[1](#tab:modelspec){reference-type="ref" reference="tab:modelspec"} and
Figure [1](#fig:tt){reference-type="ref" reference="fig:tt"}.

::: {#tab:modelspec}
  Models                           Text Representation                                                                                                   Epochs
  ------------------------ ----------------------------------- ---------------------------------------------------------------------------------------- --------
  CNN                             GloVe Wikipedia w-emb        2 Convolutional Layers (1 per filter)2 Max Pooling Layers1 Dropout Layer1 Linear Layer     100
  [Bert]{.smallcaps}-CNN    Base [Bert]{.smallcaps} (uncased)  2 Convolutional Layers (1 per filter)2 Max Pooling Layers1 Dropout Layer1 Linear Layer      10
  [Bert]{.smallcaps}-GRU    Base [Bert]{.smallcaps} (uncased)  1 Bidirectional GRU RNN Layer1 Dropout Layer1 Linear Layer                                  10

  : Model specifications of neural networks and deep language
  representation models
:::

\
 \

![Training time for neural networks and deep language representation
models for classifying political texts by *major* and *minor* policy
domain](figs/tt.png){#fig:tt width=".45\\linewidth"}

[Bert]{.smallcaps} with Gated Recurrent Units (GRU)
---------------------------------------------------

First proposed by Cho et al. , Gated Recurrent Units---formerly referred
to as the RNN Encoder-Decoder model---use update gates and reset gates
to solve the vanishing gradient problems often encountered in
applications of recurrent neural networks  [@kanai2017preventing]. The
update gate helps the model determine the extent to which past
information is carried on in the model whilst the reset gate determines
the information to be removed from the model  [@chung2014empirical].
Hence, it solves the aforementioned problem by not completely removing
the new input, instead keeping relevant information to pass on to
further subsequent computed states. In our analysis, we employ a
multi-layer, bidirectional GRU model from PyTorch[^3]. As shown in Table
[1](#tab:modelspec){reference-type="ref" reference="tab:modelspec"}. The
results are subject to a dropout layer prior to classification via a
linear layer.

[Bert]{.smallcaps} with Convolutional Neural Networks (CNN)
-----------------------------------------------------------

We incorporate CNNs with [Bert]{.smallcaps} using the same CNN
architecture as our baselines (Table
[1](#tab:modelspec){reference-type="ref" reference="tab:modelspec"}).
The model utilizes the aforementioned [Bert]{.smallcaps} base, uncased
tokenizer with convolutional filters of sizes 2 and 3 applied with a
ReLu activation function. We use a 1D-max pooling layer, a dropout layer
($N = 0.5$) to prevent overfitting, and a Cross Entropy Loss function.
We employ the model to classify policy domains ($N = 8$) and policy
preferences ($N = 58$), each of which includes a category for
quasi-sentences that do not fall into this classification scheme.
Hereafter, we refer to these classifications as 'major' and 'minor'
categories, respectively. A graphical representation of our model is
shown in Figure [2](#fig:BERTCNNfig){reference-type="ref"
reference="fig:BERTCNNfig"}.

![Graphical representation of the base BERT-CNN model to predict major
policy domains.](figs/BERTfig3.png){#fig:BERTCNNfig
width=".9\\linewidth"}

Evaluation
----------

We evaluate the performance of our proposed method against several
baselines , which include:

-   **Multinomial Naive Bayes**: This algorithm, commonly used in text
    classification, operates on the *Bag of Words assumption* and the
    assumption of *Conditional independence*.

-   **Support Vector Machines**  [@tong2001support]: We used this
    traditional binary classifier to calculate baselines with the `SVC`
    package from `scikit-learn`[^4], employing a "one-against-one"
    approach for multi-class classification.

-   **Convolutional Neural Networks (CNN)**
     [@DBLP:journals/corr/Kim14f; @lecun1998gradient]: To run this deep
    learning model, originally designed for image classification, we
    first made use of pre-trained word vectors trained by GloVe, an
    unsupervised learning algorithm for obtaining vector representations
    for words  [@Pennington_Socher_Manning_2014].

To evaluate model fit, we utilized *accuracy* and *loss* as key metrics
to compare performance of our *CNN* and [Bert-GRU]{.smallcaps} baseline
and our proposed models ([Bert+CNN]{.smallcaps},
[Bert+GRU]{.smallcaps}). We calculated the *F1-score* for each model
that we ran. In our results, we present both the Macro-F1 score and
Micro-F1 score[^5].

llcccc Category & Model & Test Loss & Test Acc. & Micro-F1 & Macro-F1\
\[2\]\*Major & MNB & --- & 0.553 & 0.553 & 0.398\
& SVM & --- & 0.578 & 0.578 & 0.460\
& CNN & 1.177 & 0.589 & 0.589 & 0.466\
& BERT-GRU & 1.166 & 0.594 & 0.593 & 0.479\
& BERT-CNN & **1.152** & **0.591** & **0.591** & **0.473**\
\[1\]\*Minor & MNB & --- & 0.385 & 0.385 & 0.154\
& SVM & --- & **0.463** & **0.463** & **0.299**\
& CNN & 2.136 & 0.454 & 0.454 & 0.273\
& BERT-GRU & 2.216 & 0.432 & 0.432 & 0.239\
& BERT-CNN & **2.098** & 0.448 & 0.448 & 0.260\

\
 \

Architecture fine tuning
------------------------

We tested different modifications of the CNN and [Bert]{.smallcaps}
models. For the CNN models, we compared the following modifications:

-   **Stemming and Lemmatization**: We test whether stemming or
    lemmatizing text in the pre-processing steps improves predictions
    using quasi-sentences from the Manifesto Project Corpus.

-   **Dropout rates**: We decreased the dropout rate from 0.5 to 0.25 to
    determine whether fine-tuning dropout rates yield differences in
    performance. This is because we initially found that our models were
    overfitting.

-   **Additional linear layer**: An additional linear layer was added
    prior to the final categorzation linear layer to establish whether
    "deeper" neural networks generate improved predictions.

-   **Removal of uncategorized quasi-sentences**: The results from our
    base models yield lower Macro-F1 scores due to the difficulty of
    correctly categorizing quasi-sentences that do not fall into any of
    the 7 policy domains or 57 policy preference codes. We are thus
    interested in whether predictions improve if the uncategorized
    quasi-sentences are taken out of the data used for analysis.

For the [Bert]{.smallcaps} models, we compared the following
modifications:

-   **Training Embeddings**: For our base [Bert]{.smallcaps} models, all
    training of embeddings were frozen. Therefore, we enable training of
    the embeddings in this modification to establish how training
    embeddings contributes to the performance of deep language
    representation models with this classification task.

-   **Training models based on recurrent runs**: We trialed training the
    [Bert]{.smallcaps} models sequentially with different learning rates
    (LR = 0.001, 0.0005 and 0.0001) of 10 epochs each for a total of 30
    epochs in aims to improve the performance of our neural networks and
    deep language representation models.

-   **Large, cased tokenizer**: The [Bert]{.smallcaps} Large cased
    tokenizer was used instead of the [Bert]{.smallcaps} BASE uncased
    tokenizer employed in our base models.

Results
=======

As shown in Table
[\[tab:modelresults\]](#tab:modelresults){reference-type="ref"
reference="tab:modelresults"}, the [Bert]{.smallcaps}-CNN model
performed best for predicting both major and minor categories compared
to the BERT-GRU model and CNN baseline. However, our SVM baseline
outperformed the neural network models for predicting minor categories.
We believe that the shortcomings of our neural networks and deep
language representation models for this text classification task are due
to limitations in specifying the number of epochs in training. We also
observed overfitting in our models. For instance, with our CNN model,
validation loss increased with each additional epoch after a certain
number of epochs. As shown in Figure
[\[fig:minornooverfitting\]](#fig:minornooverfitting){reference-type="ref"
reference="fig:minornooverfitting"}, training accuracy of this model
also increased at the cost of validation accuracy. However, this was not
the case for deep language representation models classifying texts by
minor categories. Overall, our results demonstrate that, between the two
[Bert]{.smallcaps} models, the [Bert]{.smallcaps}-CNN model demonstrates
superior performance against bag-of-words approaches and other models
that utilize neural networks.

### CNN and [Bert]{.smallcaps} Modifications {#cnn-and-bert-modifications .unnumbered}

Comparing modifications to our CNN models, our results suggest that the
base model outperforms most alternative model specifications. As
outlined in Table
[\[tab:CNNchange\]](#tab:CNNchange){reference-type="ref"
reference="tab:CNNchange"}, reducing the dropout rate to 0.25 improved
the model on some indicators marginally. As expected, the removal of
uncategorized quasi-sentences yielded improvements in predictions, with
a significantly higher Macro-F1 score compared to other model
specifications. Based on these results, future work should focus on how
model predictions of uncategorized quasi-sentences can be improved,
given their random nature.

llcccc Category & Model & Test Loss & Test Acc. & Micro-F1 & Macro-F1\
\[2\]\*Major & MNB & --- & 0.553 & 0.553 & 0.398\
& SVM & --- & 0.578 & 0.578 & 0.460\
& CNN & 1.177 & 0.589 & 0.589 & 0.466\
& BERT-GRU & 1.166 & 0.594 & 0.593 & 0.479\
& BERT-CNN & **1.152** & **0.591** & **0.591** & **0.473**\
\[1\]\*Minor & MNB & --- & 0.385 & 0.385 & 0.154\
& SVM & --- & **0.463** & **0.463** & **0.299**\
& CNN & 2.136 & 0.454 & 0.454 & 0.273\
& BERT-GRU & 2.216 & 0.432 & 0.432 & 0.239\
& BERT-CNN & **2.098** & 0.448 & 0.448 & 0.260\

\
 \

llccccc Model & Change & Test Loss & Test Acc. & Micro-F1 & Macro-F1 &
Epochs\
\[1\]\*CNN & Base model & 1.177 & **0.589** & **0.589** & 0.466 & 100\
& Lemmatized text & **1.174** & 0.585 & 0.585 & 0.460 & 100\
& Stemmed text & 1.213 & 0.577 & 0.576 & 0.448 & 100\
& Dropout = 0.25 & 1.177 & **0.589** & 0.588 & **0.467** & 100\
& Additional layer & 1.180 & 0.586 & 0.586 & 0.462 & 100\
& Removing uncategorized QSs & **1.136** & **0.596** & **0.595** &
**0.535** & 100\

\
 \

While we observed some improvements with modifications to the CNN model,
we find that our base [Bert]{.smallcaps} models performed best compared
to other fine-tuned modifications to model architecture. The results of
our base [Bert]{.smallcaps} model and alternative model specifications
are shown in Table
[\[tab:BERTchange\]](#tab:BERTchange){reference-type="ref"
reference="tab:BERTchange"}. Even though it is possible that our base
[Bert]{.smallcaps} model is best for this classification model, our
results could also indicate the presence of over-fitting or the lack of
sufficient training available given the low number of epochs.

clccccc & Change & Test Loss & Test Acc. & Micro-F1 & Macro-F1 & Epochs\
\[2\]\*BERT-GRU & Base model & **1.152** & **0.594** & **0.593** &
**0.479** & 10\
& Training emb & 1.163 & 0.592 & 0.592 & **0.479** & 10\
& Recurrent runs, training & 1.234 & 0.582 & 0.581 & 0.459 & 30\
& Large, uncased & 1.172 & 0.592 & 0.591 & 0.469 & 10\
\[1\]\*BERT-CNN & Base model & 1.166 & **0.591** & **0.591** & **0.473**
& 10\
& Training emb & 1.167 & 0.587 & 0.587 & 0.458 & 10\
& Recurrent runs, training & **1.157** & 0.589 & 0.589 & 0.468 & 30\
& Large, uncased & 1.192 & 0.580 & 0.580 & 0.450 & 10\

\
 \

Limitations and Analysis {#discussion}
========================

As shown in Figure
[\[fig:majoroverfitting\]](#fig:majoroverfitting){reference-type="ref"
reference="fig:majoroverfitting"}, we observed overfitting with our
major policy domain classification models. Despite employing changes and
modifications to our models, including varied dropout rates,
architecture fine-tuning and different learning rates, we did not find
any variants of the models employed in analysis that would yield
significant improvements in performance. We posit that potential
improvements on these issues could be resolved by employing transfer
learning and appending our sample of English-language manifestos with
other political documents, such as debate transcripts.

In contrast, as shown in Figure
[\[fig:minornooverfitting\]](#fig:minornooverfitting){reference-type="ref"
reference="fig:minornooverfitting"}, we observed little over-fitting in
our minor policy domain classification models. Our classifier could
benefit from employing transfer learning and appending our sample of
manifesto quasi-sentences with other political texts, especially for
policy domains with relatively fewer quasi-sentences to train on. It is
also important to note that, compared to the more computationally
intensive neural networks and deep language representation models, our
Multinomial Bayes and SVM baselines did not perform significantly worse.
In fact, for the minor categories, the SVM yielded superior performance
in some metrics compared to that of the neural network models.
Notwithstanding the lack of training of certain models, this may suggest
that increasing the model complexity and consequently the computational
power required may not necessarily lead to increased model performance.

![image](figs/CNNMajor_acc_loss.png){width=".7\\linewidth"}

![image](figs/BERTCNNMinor_acc_loss.png){width=".7\\linewidth"}

![image](figs/prf1-major-bw.png){width=".8\\linewidth"}

Substantially lower Macro-F1 scores across all models point to mixed
performance in classification by category. As shown in Figure
[\[fig:prf1major\]](#fig:prf1major){reference-type="ref"
reference="fig:prf1major"}, we observe high variation in the performance
of our classifiers between categories. However,, we observe poor
performance in classifying quasi-sentences that do not belong to one of
the seven policy domains. For our BERT-CNN model, the easiest categories
to predict were "welfare and quality of life", "economy", and "freedom
and democracy". The superior performance of predicting the first two
categories is not particularly surprising, as a substantial number of
quasi-sentences in our sample of English-language party manifestos are
attributed to these topics. As shown in Table
[\[tbl:desc\]](#tbl:desc){reference-type="ref" reference="tbl:desc"},
30,750 quasi-sentences are attributed to the "welfare and quality of
life" category and 24,757 quasi-sentences are attributed to the
"economy" domain.

In contrast, the relatively superior performance of predicting the
"freedom and democracy" category is surprising. Out of our total sample
of $n_{\mathrm{sentences}}=99,681$, only 4,700 documents are attributed
to the "freedom and democracy" category. Intuitively, the performance of
our classifier with this underrepresented policy domain could be
attributed to a variety of possible explanations. One possible
explanation is the presence of distinct features such as topic-unique
vocabulary that do not exist in other categories. Future work on
classification of political documents that fall under this category
would benefit from looking into features that might distinguish this
policy domain from others.

Conclusion
==========

In this paper, we trained two variants of the state-of-the-art
Bidirectional Encoder Representations from Transformers
([Bert]{.smallcaps})---one incorporating a bidirectional GRU model, and
another incorporating CNNs. We demonstrate the superior performance of
deep language representation models combined with neural networks to
classify political domains and preferences in the Manifesto Project. Our
proposed method of incorporating [Bert]{.smallcaps} with neural networks
for classifying English language manifestos addresses issues of
reproducibility and scalability in labeling large volumes of political
texts. As far as we know, this is the most comprehensive application of
deep language representation models and neural networks for classifying
statements from political manifestos.

We find that using [Bert]{.smallcaps} in conjunction with convolutional
neural networks yields the best predictions for classifying English
language statements parsed from party manifestos. However, our proposed
[Bert]{.smallcaps}-CNN model requires further fine-tuning to be
effective in providing acceptable predictions to improve on less
computationally intensive methods and replace human annotations of
fine-grained policy positions. As expected, our proposed approach and
baselines perform better for classifying major policy domains over minor
categories. We also observe differences in performance between
categories. Among the major policy domains, the categories that
performed best include "welfare and quality of life", "economy", and
"freedom and democracy". The superior performance of the latter category
is surprising because it makes up the smallest proportion of
quasi-sentences in the Manifesto Project Corpus.

There are several avenues for future work on neural networks and deep
language representation models for automatically labeling political
texts. For instance, investigating the features of individual categories
that demonstrate superior performance would shed light on how we can
incorporate additional features of texts to improve model performance.
This area of research would also benefit from better understanding how
we can filter out texts that do not fall into a particular
classification scheme. Knowledge on how these issues could be resolved
to improve model performance would allow for extensions in the
application of deep learning models for classifying political texts.

[^1]: The data used in analysis comprises of statements from six
    Democratic and Republican election manifestos from the 2004, 2008
    and 2012 elections in the United States.

[^2]: [manifesto-project.wzb.eu](manifesto-project.wzb.eu)

[^3]: <https://pytorch.org/>

[^4]: <https://scikit-learn.org/stable/>

[^5]: The micro score calculates metrics globally whilst the macro score
    calculates metrics for each label and reports the unweighted mean.
