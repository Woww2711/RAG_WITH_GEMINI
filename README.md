# A Journey Through Advanced RAG Techniques
This repository is a collection of Jupyter Notebooks that document a step-by-step exploration of Retrieval-Augmented Generation (RAG). The project starts with simple, foundational RAG pipeline and progressively introduces more advanced, production-grade techniques to improve accuracy, reliability and efficiency.

This repository also serves as a personal knowledge base, capturing key lessons, best practices, and common pitfalls encountered during the development and implementation of each RAG strategy.

This repository is heavily based of the repo [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques/tree/main), but relied on free to use API (Gemini, Cohere, etc.) rather than OpenAI API for the LLM, the embeddings, the rerank model, etc.

## Warning: Wall Of Text 

The next part maybe lengthy, but worth a read. This part will be consistently updated until I finished every lesson.

## 1 - Simple rag
1. Best to use both GEMINI_API_KEY and GOOGLE_API_KEY in .env, with the same value, because some modules use GEMINI, others use GOOGLE.
2. Be careful of what the variables in curly bracket you named in from_templates, naming them arbitrarily may lead to KeyError: naming mismatch when calling invoke.

## 2 - Rag with CSV
3. When working with CSV files, first consider the row numbers then decide whether or not to use text splitter, you should only use it when 1 or more columns contain a bunch of text, or a lengthy paragraph. Splitting small csv file may lead to incorrect information retrieval. Also, row-level chunking is automatically done by the CSVLoader load() method.
4. FAISS vectorstore can be created automatically by passing the embedding model and the documents (from_documents method), or you want full control over the FAISS initialization: embedding model, index implementation, docstore, index and id mapping,... (add_documents method).
5. The ChatPromptTemplate has 2 modules:
- from_template: contains all system prompt, context, and user's question in a big string. The LLM will receive this plain message from user. This method is simple, easy and great for beginners, but can not utilize the chat template that modern chat models were trained on, and hard to manage chat history.
- from_messages: make a list of structured messages, each will be given a role (system, user, assistant or AI). The LLM will receive a JSON data. Clear separation between AI answer and user's question. Works really well with modern AI chatbot, better responses and adherence to instructions. Also it's more easy to build and manage the conversation history. In conclusion, use from_messages, it's best practice.

## 3 - Reliable Rag
6. Set the model's temperature to 0 so that the same input would always give the same output.
7. If you have set the max_tokens to a certain number, it may lead to the AI model returning an empty string because it has reached the max token due to long context, or question.
8. Split by character and split by token: 
- RecursiveCharacterTextSplitter(chunk_size=1000, ...) splits the text by trying to find separators (\n\n, \n), then create chunks that are approximately 1000 characters long. The problem is that LLM don't see characters, they see `tokens`. 1000 characters may be 600 or 800 tokens, depending on the words. This is imprecise
- from_tiktoken_encoder(chunk_size=500, ...) instead tokenizes the text, then creates chunks that are exactly 500 tokens. This is the best practice, optimized for LLM processing. You have alsolute control over the size of context.
9. The AI model can be compelled to output a strict data schema (JSON, Pydantic,...). This can be achieved through the PydanticOutputParser. The LLM must follow this explicit formatting instructions, therefore minimize the chance of errors and ensure that the output is always in a predictable and usable format.

## 4 - Propositions Chunking
10. Always a good practice to use strict data schema to make the output format easy to process.
11. Propositions chunking requires a handful of API call to check the quality of propositions, which is not ideal when we want to optimize for cost and efficiency.

## 5 - Query Transformations

| Template Type | `PromptTemplate` | `ChatPromptTemplate` | `FewShotPromptTemplate` |
| :--- | :--- | :--- | :--- |
| **Essence** | A simple text string | A **list of messages with roles** | A complex text string containing **sample examples** |
| **Analogy** | Fill-in-the-blanks game (Mad Libs) | **Movie script** | **Teaching by example** |
| **Suitable Model** | Older LLMs that only accept a single text string | **Most modern Chat Models** (Gemini, GPT-4,...) | All types of LLMs |
| **Use Case**| When an extremely simple prompt is needed | **When building chatbots, RAG, agents** (almost all the time) | When needing to instruct an LLM for a complex task or a specific format |


## 6 - HyDE
12. HyDE is a retrieval enhancement technique, not an answering mechanism. So remember to implement the answering (Generation part) in order to create the full RAG loop.
13. HyDE can be imported via the `from langchain.chains.hyde.base import HypotheticalDocumentEmbedder`, this acts as an embeddings wrapper, which wraps around an existing embeddings model.

## 7 - HyPE
14. For each chunk splitted from the original documents, there will be an API call to create multiple hypothetical prompts for that specific chunk. As your documents get larger in size, you'll surely reach the API limit. This approach requires you to not overwhelm the API and also be able to handle failures when you hit the rate limit (same as propositions chunking).

## 8 - Contextual Chunk Headers
15. Respect the rate limits of models when coding, use batches, add delays, calculate total tokens to make sure you don't exceed the TPM
16. Use rerank model, the retriever only retrieves list of chunks (documents) that could be relevant to the query, while the rerank model will assess and score each candidate to find the chunk most relevant to the query, therefore enhance the accuracy of the RAG system.
17. The complete process:
    1. Retrieval: Use the retriever to retrieve 10-20 most relevant chunks from thousands of chunks.
    2. Reranking: Pass these 10-20 chunks to the specialized Reranker to be scored and reranked.
    3. Generation: Take the top 3-5 highest-scoring chunks from the Reranker and pass them to the final LLM to synthesize the answer.

## 9 - Relevant Segment Extraction
18. This method will require `chunk_overlap = 0`, this allows the reconstruction of the document (segments) by simply concatenating chunks.
19. Chunks will be graded through the combination of relevance score and rank (both from the rerank model). Basically, the lower the rank, the more penalty chunks will get. The new score is called `chunk_values`, and this will be crucial for the next step.
20. We subtract the `chunk_values` by a value called `irrelevant_chunk_penalty`, `0.2` seems to work well empirically. This will make the value of irrelevant chunks to a negative number, while keeping the values of relevant chunks positive.
21. The nested for loop is just a demonstration of how things would look like under the hood. In practical, large scale scenarios, more optimized and efficient method will be used.
22. Try to find the most suitable `overall_max_length`. The size of the final retrieved chunks of document will greatly be affected by the parameter `overall_max_length`. It acts like a limit for how much context can be passed into the LLM. Having this under control means that having cost and latency under control.

## 10 - Context Enrichment Window
23. In basic RAG, user inputs a query, the retriever outputs a chunk(`k = 1`) that is relevant to the query, this chunk will have an index `n`. When using context enrichment window, we can specify a new parameter called `num_neighbor`, instead of retrieving only chunk with index `n`, the retriever will expand, or enrich the context by also including chunks that have index between `n - num_neighbor` and `n + num_neighbor`. For example, `num_neighbor = 1`, chunk with index `5` was chosen by the retriever, then the final context would be chunk `4,5,6`.
24. **RecursiveCharacterTextSplitter** contains 2 methods: `split_documents` and `split_text`
- **split_documents:** takes in a list of `Document` objects, outputs a list of `Document` objects that is splitted into smaller parts, and **retains the metadata**
- **split_text:** takes in a string, returns a list of strings splitted from the original string, and **does not include metadata**

    In other word, split_documents should be considered first rather that split_text, unless you have other uses for it.
25. chunk_overlap should not be set to 0, this ensures that contents and ideas are not corrupted between splitted chunks. Also, modern LLMs can easily handle these overlapped chunks without sacrificing quality.

## 11 - Semantic Chunking
26. text_splitter from SemanticChunker will apply the embedding model on each and every **sentences** of the data (pdf, md, etc.), which means a sheer amount of API calls. You should choose embedding model wisely to reduce costs if you're choosing to call API. Otherwise, download some embedding models to the local PC and test them for the best result. I used Cohere Embedding model instead of Gemini because the rate limits are much more generous even at free tier.
27. Choosing the right `breakpoint_threshold_type` is important too! Test all `percetile`, `standard_deviation` and `interquartile` to see which fits best with your data.

## 12 - Contextual Compression
28. Actually, we've been doing contextual compression (CC) a few times now, `reranker` for example, is also considered a CC technique. CC is not a single, modular technique, it's more of an architecture, a processing model. So what's the differences you might ask:

| Aspect | Contextual Compression with Reranker | Contextual Compression with LLMChainExtractor |
| :--- | :--- | :--- |
| **Analogy** | A "Judge" | An "Editor / Highlighter" |
| **Main Task** | **SELECT:** Skims through the documents and selects the top `n` best documents. | **EXTRACT:** Reads each document carefully and cuts out only the relevant sentences. |
| **Output of the "Compressor"**| A list of the **original** Documents, reranked and filtered. | A list of **new** Documents, whose content consists of the extracted sentences. |
| **Final Context Quality**| Good. Noise has been filtered at the document level. | Very high. Extremely concise and clean, containing only the "gold". |
| **Cost** | Usually lower (A Reranker's API is often cheaper than a full LLM call). | Higher. Requires an LLM call for each retrieved document. |
| **Latency** | Relatively fast. | Slower, as the LLM needs time to read and extract from each document. |
| **Technical Term** | Reranking | Relevant Segment Extraction (RSE) |

29. The RetrievalQA is good enough if you don't need to extensively configure params, i/o format, or chain parts together. Just a simple all in one solution. But in reality, everything in LangChain should follow the LCEL method, which is using the `|` operator to ensure high flexibility and transparency.

## 13 - Document Augmentation
30. I won't be doing the code for this, the cost is just to high. It needs multiple LLM calls just for data preparation, using local models would be the most optimal choice.
31. First, all the content will be splitted into "big chunks" (`text_documents`, e.g. 2000 tokens each). Next, for every "big" chunks, we'll split them into "small chunks" (`text_fragments`, e.g. 200 tokens each). Base on the configuration, the code will call LLM to create a list of questions that the chunks can provide an answer (small chunks or big chunks, base on config). Just like HyPE.
32. The difference with HyPE is that Document Augmentation finds the small chunk, returns the big chunk that contains the small chunk, while HyPE only returns the small chunk. Which means DA provides more context, and lower latency, because the questions generation was already done, the process is just a vector search, and an LLM call.
33. Returning a whole big chunk may contains unrelated information, which would increase tokens usage and possibly make the LLM confused.

## 14 - Fusion Retrieval
34. Fusion retrieval is to combine both semantic retriever (FAISS, Chroma, etc.) and keyword retriever (BM25, TF-IDF, etc.) to further enhance retrieval quality.
35. Each retriever outputs a different score, on a different scale, so a normalization and rescoring must be done before reranking the docs. LangChain has a module for this called `EnsembleRetriever`, which takes 2 retrievers and their weights as input, and outputs a list of `k` most relevant docs. By the way, you can also implement your own normalization and scoring technique if your retriever needs different processing method.

## 15 - Reranking
36. There are multiple methods of reranking, each has its own advantages and disadvantages:

| Aspect | Cross-Encoder (Open-Source) | LLM as a Reranker | Rerank API (Cohere)|
| :--- | :--- | :--- | :--- |
| **How it Works** | Cross-comparison (Query, Doc) | Executes based on a prompt | Optimized Cross-Encoder service |
| **Speed** | Medium (GPU dependent) | **Slowest** | **Fastest** |
| **Cost** | **Cheapest** (only infrastructure cost) | **Most Expensive** | Medium (pay-per-API call) |
| **Accuracy** | **Very High** | Good, but inconsistent | **Very High (SOTA)** |
| **Complexity** | **High** (self-managed) | Medium (requires good prompting) | **Very Low** (just an API call) |

Also, there exists many more reranking techniques but I won't be discussing about it.

## 16 - Hierarchical Indices
37. This is a retriever optimization technique, not chunking or reranking. It changes the data structure of the vectorstore in order to accelarate the `retriever.invoke()` step by many times.
38. This technique should be used with other RAG techniques such as CCH, HyDE, HyPE, etc. Each of them addresses different process in the RAG pipeline.
39. Hierarchical Indices requires many API calls, and it would just get much more as the documents pile up. If you plan to use external APIs, robust limit rate backoff techniques should be implemented to ensure the pipeline's stability (less prone to API errors) and versatility (can adapt to many kinds of API).

## 17 - Dartboard
40. Dartboard is a type of reranking, but more complex and more efficient. 
- Typical reranking method would only `grade` chunk independently of each others, and give them a score base on their relevance with the query, then ranking at the end.
- Dartboard, on the other hand, firstly finds the most relevant chunk, then proceed to find the next chunk that: has relevance with the query, also differs from the first chunk. In other words, dartboard finds `k` chunks that are: relevant to the query, and also differs from each others. The purpose is not to find the best chunks, but a collective of chunks that when they come together, provides the most comprehensive context to feed to the LLM.
41. This technique does not require extensive API calls to preprocess data, only linear algebra and statistics are needed to optimize. This is much more reliable than APIs, which can sometimes be overkill and time-consuming.

## 18 - Multimodal Rag With captioning
42. Up until now, every techniques was just only for plain text, images weren't considered. So, this technique introduces a way to incorporate images information by first use an LLM to caption the image, then use an embedding model to generate its embedding, finally store both the text and images embedding into a single vectorstore.
43. The LLM response with no `StrOutputParser` would be redundant for the process of creating vectorstore. It is recommended to use `StrOutputParser` if you only care for the final main content, not the metadata.