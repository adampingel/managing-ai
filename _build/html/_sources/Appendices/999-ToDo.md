# To Do

fix glue in ChartsDemo
citation / bibliography
open in colab button

load corpora (Federalist Papers, Shakespeare, …)

collection of publically available information and personal opinions
copyright notice

testing and data quality monitoring
RACI Charts

change data capture
event sourcing

knowledge graphs

technology landscape
cloud vendors
tools
notebooks

project management
budget
hiring
people management
triangulation and corroboration
proof
tradeoffs
  * Costs: labor, cloud, hardware, software, data, ...
differentiability
conferences
demo
product management
ongoing learning
education
LLM
model quality evaluation

latency measurement
provenance
attribution
bottleneck problem

team topologies

repeatability
feature stores
experiment tracking
sentence splitting

book.adoc content
PII
search
TFIDF, BM25
List metrics
when to go beyond OSS/OTS software?

“Attention is all you need” paper

Prompt Chaining
Eg: LangChain

Transfer Learning

https://www.linkedin.com/posts/juliaelzini_bert-gpt-transformer-activity-7046045851307167744-eo_E?utm_source=share&utm_medium=member_android

https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html

https://atcold.github.io/pytorch-Deep-Learning/

https://deeppavlov.ai/about-us


LangChain
https://github.com/hwchase17/langchain

“Transformer Models: An Introduction and Catalog”
https://arxiv.org/abs/2302.07730
Author: Xavier Amatrian
Abstract: In the past few years we have seen the meteoric appearance of dozens of models of the Transformer family, all of which have funny, but not self-explanatory, names. The goal of this paper is to offer a somewhat comprehensive but simple catalog and classification of the most popular Transformer models. The paper also includes an introduction to the most important aspects and innovation in Transformer models.

Semi-Supervised Learning
Wikipedia: https://en.wikipedia.org/wiki/Semi-Supervised_Learning#
Semi-supervised learning, otherwise termed weak supervision, is a branch of machine learning where noisy, limited, or imprecise sources are used to provide supervision signal for labeling large amounts of training data in a supervised learning setting.[1] This approach alleviates the burden of obtaining hand-labeled data sets, which can be costly or impractical. Instead, inexpensive weak labels are employed with the understanding that they are imperfect, but can nonetheless be used to create a strong predictive model.[2][3][4] Semi-supervised learning can therefore be seen as a reasonable middleground between supervised and unsupervised machine learning approaches.

Generative Adversarial Network (GAN)
Wikipedia: https://en.wikipedia.org/wiki/Generative_adversarial_network
A generative adversarial network (GAN) is a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in June 2014.[1] Two neural networks contest with each other in the form of a zero-sum game, where one agent's gain is another agent's loss.
Given a training set, this technique learns to generate new data with the same statistics as the training set. For example, a GAN trained on photographs can generate new photographs that look at least superficially authentic to human observers, having many realistic characteristics. Though originally proposed as a form of generative model for unsupervised learning, GANs have also proved useful for semi-supervised learning,[2] fully supervised learning,[3] and reinforcement learning.[4]
The core idea of a GAN is based on the "indirect" training through the discriminator, another neural network that can tell how "realistic" the input seems, which itself is also being updated dynamically.[5] This means that the generator is not trained to minimize the distance to a specific image, but rather to fool the discriminator. This enables the model to learn in an unsupervised manner.
GANs are similar to mimicry in evolutionary biology, with an evolutionary arms race between both networks.

GPT (not ChatGPT)
GAN
Large Language Model (LLM)
Reinforcement Learning
Active Learning
Self-Supervision

Daniel Lewis found this Meta paper re PEER: https://arxiv.org/pdf/2208.11663.pdf?utm_source=pocket_saves

Microsoft Promptist
https://arxiv.org/abs/2212.06713
https://huggingface.co/spaces/microsoft/Promptist
https://github.com/microsoft/LMOps

prompt tuning
prompt engineering

Prompt Tuning info from Ben Lorica

Llama from Meta https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/


Dolly

https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/

https://aibusiness.com/meta/ai-luminary-yann-lecunn-sets-us-straight-on-generative-ai

MEND to edit LLMs https://hai.stanford.edu/news/how-do-we-fix-and-update-large-language-models

spacy course https://www.datacamp.com/courses/advanced-nlp-with-spacy
https://course.spacy.io/en/


Commercial and Open Source Software

AWS Ground Truth
AWS Human in the Loop
Snorkel
SparkNLP
https://nlp.johnsnowlabs.com/docs/en/install
OpenAI
Google Bard commercial fuck-up. What could they have done?
BERT
RoBERTA
Lambda
Spacey
Fixie


Argilla
Website: https://docs.argilla.io/en/latest/index.html
Quickstart: https://docs.argilla.io/en/latest/getting_started/quickstart.html
Argilla Client: a powerful Python library for reading and writing data into Argilla, using all the libraries you love (transformers, spaCy, datasets, and any other).
Argilla Server and UI: the API and UI for data annotation and curation.


Create New Hugging Face space
https://huggingface.co/spaces/pingel/ArgillaTest?logs=build
github


“Why Vector Search Now?”
Doug Turnbull
https://softwaredoug.com/blog/2023/02/13/why-vector-search.html

“Domain-Specific Intelligent Bots”
Arun Shankar
https://medium.com/@shankar.arunp/chatgpt-decoded-an-expert-guide-to-mastering-the-technology-and-building-domain-specific-3a95b42827bb

Zero-Shot Learning
Wikipedia: https://en.wikipedia.org/wiki/Zero-shot_learning
Zero-shot learning (ZSL) is a problem setup in machine learning where, at test time, a learner observes samples from classes which were not observed during training, and needs to predict the class that they belong to. Zero-shot methods generally work by associating observed and non-observed classes through some form of auxiliary information, which encodes observable distinguishing properties of objects.[1] For example, given a set of images of animals to be classified, along with auxiliary textual descriptions of what animals look like, an artificial intelligence model which has been trained to recognize horses, but has never been given a zebra, can still recognize a zebra when it also knows that zebras look like striped horses. This problem is widely studied in computer vision, natural language processing, and machine perception.[2]

GPT-3
Wikipedia: https://en.wikipedia.org/wiki/GPT-3
Generative Pre-trained Transformer 3 (GPT-3) is an autoregressive language model released in 2020 that uses deep learning to produce human-like text. Given an initial text as prompt, it will produce text that continues the prompt.
The architecture is a decoder-only transformer network with a 2048-token-long context and then-unprecedented size of 175 billion parameters, requiring 800GB to store. The model was trained using generative pre-training; it is trained to predict what the next token is based on previous tokens. The model demonstrated strong zero-shot and few-shot learning on many tasks.[2] The authors described how language understanding performances in natural language processing (NLP) were improved in GPT-n through a process of "generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task." This eliminated the need for human supervision and for time-intensive hand-labeling.[2]

Yann LeCun “A Path Towards Autonomous Machine Intelligence”
https://openreview.net/pdf?id=BZ5a1r-kVsf


ChatGPT
Wikipedia: https://en.wikipedia.org/wiki/ChatGPT
ChatGPT (Chat Generative Pre-trained Transformer[2]) is a chatbot developed by OpenAI and launched in November 2022. It is built on top of OpenAI's GPT-3 family of large language models and has been fine-tuned (an approach to transfer learning) using both supervised and reinforcement learning techniques.

Let’s Build ChatGPT: From Scratch, In Code, Spelled Out
Youtube: https://www.youtube.com/watch?v=kCc8FmEb1nY

LEGAL-BERT
https://arxiv.org/pdf/2010.02559.pdf

“What Learning Algorithm is In-Context Learning? Investigations with Linear Models“
arxiv: https://arxiv.org/pdf/2211.15661.pdf
Neural sequence models, especially transformers, exhibit a remarkable capacity for in-context learning. They can construct new predictors from sequences of labeled examples (x, f(x)) presented in the input without further parameter updates. We investigate the hypothesis that transformer-based in-context learners implement standard learning algorithms implicitly, by encoding smaller models in their activations, and updating these implicit models as new examples appear in the context. Using linear regression as a prototypical problem, we offer three sources of evidence for this hypothesis. First, we prove by construction that transformers can implement learning algorithms for linear models based on gradient descent and closed-form ridge regression. Second, we show that trained in-context learners closely match the predictors computed by gradient descent, ridge regression, and exact least-squares regression, transitioning between different predictors as transformer depth and dataset noise vary, and converging to Bayesian estimators for large widths and depths. Third, we present preliminary evidence that in-context learners share algorithmic features with these predictors: learners’ late layers non-linearly encode weight vectors and moment matrices. These results suggest that in-context learning is understandable in algorithmic terms, and that (at least in the linear case) learners may rediscover standard estimation algorithms. Code and reference implementations are released at this https link.
