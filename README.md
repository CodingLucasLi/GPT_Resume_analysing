# ResumeGPT - GPT_Resume_analysing
# ResumeGPT - GPT简历自动分析
Using Langchain and OpenAI to analyse resume

基于Langchain和OpenAI完成以下数据处理：

1. 简历信息向量化，保存在Faiss中。


2. 按照自定义要素对Faiss向量数据库进行提问问答，收集答案存在dataframe里。


3. 将dataframe中的简历要素和岗位要求，通过LLM进行综合分析。
