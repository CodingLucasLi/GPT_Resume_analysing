# ResumeGPT - GPT-Powered Resume Auto Analysis

ResumeGPT leverages the power of Langchain and OpenAI to automate the process of resume analysis. This tool enables recruiters to efficiently process large volumes of resumes, automatically extract important features, and generate a matching score based on a predefined job description.

## Overview

The processing flow of ResumeGPT includes the following steps:

1. **Resume Vectorization**: Resume information is vectorized using the language understanding capabilities of OpenAI. These vectors are then stored in a Faiss vector database.

2. **Information Extraction**: Predefined elements are extracted from the Faiss vector database using a question-answering approach. The answers are collected and stored in a DataFrame.

3. **Comprehensive Analysis**: Using the Language Learning Model (LLM) from Langchain, the tool conducts a comprehensive analysis of the resume features and job requirements to generate a matching score. This score can be used to rank the candidates.


# ResumeGPT - GPT驱动的简历自动分析工具

ResumeGPT利用Langchain和OpenAI的能力，自动化简历分析过程。这个工具使得招聘者能够高效处理大量简历，自动提取重要特征，并根据预定义的工作描述生成匹配得分。

## 概述

ResumeGPT的处理流程包括以下步骤：

1. **简历向量化**：使用OpenAI的语言理解能力将简历信息向量化。这些向量随后存储在Faiss向量数据库中。

2. **信息提取**：从Faiss向量数据库中使用问答方式提取预定义元素。答案被收集并存储在DataFrame中。

3. **全面分析**：使用Langchain的语言学习模型（LLM），对简历特征和工作要求进行全面分析，生成匹配得分。这个得分可以用来对候选人进行排序。
