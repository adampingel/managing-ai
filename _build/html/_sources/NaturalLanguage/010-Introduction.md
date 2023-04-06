# Natural Langauge

## Topics

Chapters to be written:

* tokenization
* sentence splitting
* pos tagging
* parsing
* coreference resolution
* naive bayes (NLP applications)
* bag of words
* word2vec
* sentiment analysis
* NER
* document segmentation
* document summarization
* topic modeling
* question answering
* Scaling Laws for Neural Language Models
* Mark Liberman's "golden age of nlp" talk
* transformer architecture
* encoder/decoder
* self attention
* hyperparameter optimization (for NLP specifically)
* document similarity
* statistical fallacies
* annotation teams
* active learning
* Snorkel
* transfer learning
* explainability
* history: Eliza, Racter, cyc
* “Attention is all you need” paper
* glove
* bert
* bloom
* GPT
* Xavier Amatrian's catalog
* RLHF
* edit models.  (eg trained on github)
* prompt engineering
* Llama from Meta https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/
* Alpaca
* Dolly https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html
* Cerebras
* Microsoft Promptist
    * https://arxiv.org/abs/2212.06713
    * https://huggingface.co/spaces/microsoft/Promptist
    * https://github.com/microsoft/LMOps
* Prompt Tuning info from Ben Lorica


# In-Context Learning

“What Learning Algorithm is In-Context Learning? Investigations with Linear Models“
arxiv: https://arxiv.org/pdf/2211.15661.pdf

> Neural sequence models, especially transformers, exhibit a remarkable capacity for in-context learning. They can construct new predictors from sequences of labeled examples (x, f(x)) presented in the input without further parameter updates. We investigate the hypothesis that transformer-based in-context learners implement standard learning algorithms implicitly, by encoding smaller models in their activations, and updating these implicit models as new examples appear in the context. Using linear regression as a prototypical problem, we offer three sources of evidence for this hypothesis. First, we prove by construction that transformers can implement learning algorithms for linear models based on gradient descent and closed-form ridge regression. Second, we show that trained in-context learners closely match the predictors computed by gradient descent, ridge regression, and exact least-squares regression, transitioning between different predictors as transformer depth and dataset noise vary, and converging to Bayesian estimators for large widths and depths. Third, we present preliminary evidence that in-context learners share algorithmic features with these predictors: learners’ late layers non-linearly encode weight vectors and moment matrices. These results suggest that in-context learning is understandable in algorithmic terms, and that (at least in the linear case) learners may rediscover standard estimation algorithms. Code and reference implementations are released at this https link.

## Transformer Ecosystem

“Transformer Models: An Introduction and Catalog”
https://arxiv.org/abs/2302.07730
Author: Xavier Amatrian

> Abstract: In the past few years we have seen the meteoric appearance of dozens of models of the Transformer family, all of which have funny, but not self-explanatory, names. The goal of this paper is to offer a somewhat comprehensive but simple catalog and classification of the most popular Transformer models. The paper also includes an introduction to the most important aspects and innovation in Transformer models.

## Legal BERT

LEGAL-BERT https://arxiv.org/pdf/2010.02559.pdf

## GPT-3

Wikipedia: https://en.wikipedia.org/wiki/GPT-3

> Generative Pre-trained Transformer 3 (GPT-3) is an autoregressive language model released in 2020 that uses deep learning to produce human-like text. Given an initial text as prompt, it will produce text that continues the prompt.

> The architecture is a decoder-only transformer network with a 2048-token-long context and then-unprecedented size of 175 billion parameters, requiring 800GB to store. The model was trained using generative pre-training; it is trained to predict what the next token is based on previous tokens. The model demonstrated strong zero-shot and few-shot learning on many tasks.[2] The authors described how language understanding performances in natural language processing (NLP) were improved in GPT-n through a process of "generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task." This eliminated the need for human supervision and for time-intensive hand-labeling.[2]

## Zero-Shot Learning

Wikipedia: https://en.wikipedia.org/wiki/Zero-shot_learning

> Zero-shot learning (ZSL) is a problem setup in machine learning where, at test time, a learner observes samples from classes which were not observed during training, and needs to predict the class that they belong to. Zero-shot methods generally work by associating observed and non-observed classes through some form of auxiliary information, which encodes observable distinguishing properties of objects.[1] For example, given a set of images of animals to be classified, along with auxiliary textual descriptions of what animals look like, an artificial intelligence model which has been trained to recognize horses, but has never been given a zebra, can still recognize a zebra when it also knows that zebras look like striped horses. This problem is widely studied in computer vision, natural language processing, and machine perception.[2]

## Editing LLMs

MEND https://hai.stanford.edu/news/how-do-we-fix-and-update-large-language-models

## Collaboration Models

### PEER: A Collaborative Language Model

From Meta

https://arxiv.org/pdf/2208.11663.pdf?utm_source=pocket_saves

> Textual content is often the output of a collaborative writing process: We start with an
initial draft, ask for suggestions, and repeatedly make changes. Agnostic of this process, today’s language models are trained to generate only the final result. As a consequence, they lack several abilities crucial for collaborative writing: They are unable to update existing texts, difficult to control and incapable of verbally planning or explaining their actions. To address these shortcomings, we introduce PEER, a collaborative language model that is trained to imitate the entire writing process itself: PEER can write drafts, add suggestions, propose edits and provide explanations for its actions. Crucially, we train multiple instances of PEER able to infill various parts of the writing process, enabling the use of selftraining techniques for increasing the quality, amount and diversity of training data. This unlocks PEER’s full potential by making it applicable in domains for which no edit histories are available and improving its ability to follow instructions, to write useful comments, and to explain its actions. We show that PEER achieves strong performance across various domains and editing tasks.

## Prompt Chaining

LangChain https://github.com/hwchase17/langchain
