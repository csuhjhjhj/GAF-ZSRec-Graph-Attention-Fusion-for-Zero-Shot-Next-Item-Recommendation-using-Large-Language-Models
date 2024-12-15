# GAF-ZSRec-Graph-Attention-Fusion-for-Zero-Shot-Next-Item-Recommendation-using-Large-Language-Models
This repository provides the implementation of GAF-ZSRec, a framework combining Graph Attention Networks (GATs) and Large Language Models (LLMs) for zero-shot next-item recommendation. It employs GATs for user-item embeddings and multi-step prompting for accurate and personalized predictions.

**Graph Attention Fusion for Zero-Shot Next-Item Recommendation (GAF-ZSRec)**

**Overview**

This repository contains the implementation of **GAF-ZSRec**, a novel framework that combines **Graph Attention Networks (GATs)** with **Large Language Models (LLMs)** for zero-shot next-item recommendation. The framework leverages graph-based embeddings and adaptive candidate set construction to provide personalized recommendations without the need for task-specific training.

Key features of the implementation include:

​	•	**GAT-based embedding generation** for modeling user-item interactions.

​	•	Dynamic candidate set construction based on similarity and popularity scores.

​	•	Multi-step enhanced prompting strategy to guide LLMs in generating accurate recommendations.

**Code Structure**

​	•	main.py: The primary script for preprocessing data, generating embeddings, constructing candidate sets, and running recommendations.

​	•	main_dataset.json: Contains preprocessed user interaction data.

​	•	EnhancedGAT and BasicGAT: Implementation of GNN models for embedding generation.

​	•	Prompts (basic_temp_3, enhanced_temp_3): Scripts for constructing multi-step prompts for LLMs.

​	•	Functions for API integration with GPT-like models (e.g., OpenAI GPT or ZhipuAI).

**Prerequisites**

​	•	Python 3.8+

​	•	PyTorch 1.12+ with GPU support (optional)

​	•	[Torch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) for graph-based operations

​	•	ZhipuAI or OpenAI API keys for LLM inference

**Python Dependencies**

Install the required Python packages using:

```python
python main.py --length_limit 8 --num_cand 30 --gnn_enhanced --prompt_enhanced --api_type zhipu
```

**Key Arguments**

​	•	--length_limit: The number of historical items to consider per user.

​	•	--num_cand: The size of the candidate set for recommendations.

​	•	--gnn_enhanced: Use the enhanced GAT model with more complex graph embeddings.

​	•	--prompt_enhanced: Enable multi-step enhanced prompting for LLMs.

​	•	--api_type: Choose between zhipu or openai for LLM integration.

​	•	--use_cosine: Use cosine similarity for user embedding comparison.

​	•	--pop_weight: Adjust the influence of item popularity in candidate set construction.
