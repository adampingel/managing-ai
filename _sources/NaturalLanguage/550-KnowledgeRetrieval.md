# Knowledge Retrieval

## REALM

https://ai.googleblog.com/2020/08/realm-integrating-retrieval-into.html

> what if there was a method for pre-training that could access knowledge explicitly, e.g., by referencing an additional large external text corpus, in order to achieve accurate results without increasing the model size or complexity?  For example, a sentence found in an external document collection, "Francesco Bartolomeo Conti was born in Florence," could be referenced by the model to determine the birthplace of the musician, rather than relying on the model's opaque ability to access the knowledge stored in its own parameters. The ability to retrieve text containing explicit knowledge such as this would improve the efficiency of pre-training while enabling the model to perform well on knowledge-intensive tasks without using billions of parameters.

> In “REALM: Retrieval-Augmented Language Model Pre-Training”, accepted at the 2020 International Conference on Machine Learning, we share a novel paradigm for language model pre-training, which augments a language representation model with a knowledge retriever, allowing REALM models to retrieve textual world knowledge explicitly from raw text documents, instead of memorizing all the knowledge in the model parameters. We have also open sourced the REALM codebase to demonstrate how one can train the retriever and the language representation jointly.