# LLM4EntityMatching
This is the code repository for our work related to the general entity matching (EM) task and the Asset Administration Shell (AAS)-specific EM task.
## Selective Entity Matching
In the paper "Fine-Tuning Large Language Models with Contrastive Margin Ranking Loss for Selective Entity Matching in Product Data Integration" (submitted), we first revisit the standard pairwise EM setting by recompiling existing benchmark datasets to include more hard negative candidates, which are semantically similar to corresponding query entities. We then evaluate state-of-the-art (SOTA) pairwise matchers on these recompiled datasets, revealing the limitations of the conventional pairwise EM approach under more challenging and realistic conditions. Second, we propose a selective EM approach that formulates EM as a listwise selection task, where the query entity is compared directly with the entire candidate set rather than evaluated through independent pairwise classifications. Accordingly, a new evaluation framework is introduced, including recompiled benchmark datasets and a new evaluation metric. Third, we propose a selective EM method Mistral4SelectEM, which fine-tunes an LLM-based embedding model for selective EM by structuring it into a Siamese network and fine-tuning it with a novel contrastive margin ranking loss (CMRL). It aims to enhance the model’s ability to distinguish true positives from semantically similar negatives. 
### Method
![](/resource/Mistral4SelectEM.png)
*Fig. 3. Illustration of the end-to-end selective EM (left) and the fine-tuning strategy for Mistral4SelectEM (right). The entire process involves a single set of LLM weights, which is presented as the green block. The “red” adapter is the fine-tuned LoRA weights, which are merged with the embedding model in the inference stage for selective EM*
This method is implemented in /tasks/SelectiveEntityMatching/selection_llm. The comparative methods are implemented in selection_plm, selection_gpt, and cross_selection_llm. The selection of the methods is controlled by the main args.method.
