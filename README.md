# :world_map: The Roadmap for LLMs

![image-20230727100138263](Roadmap%20of%20LLMs.assets/image-20230727100138263.png)

> 🚧 持续更新...  [知乎，欢迎交流](https://www.zhihu.com/people/HiFuture_ToMyDream)   [全部大模型列表](https://docs.google.com/spreadsheets/d/1O5KVQW1Hx5ZAkcg8AIRjbQLQzx2wVaLl0SqUu-ir9Fs/edit?pli=1#gid=1158069878)
>
> Goal: 及时更新最新的模型，详细解读技术细节

[:world_map: The Roadmap for LLMs](#-world-map--the-roadmap-for-llms)

[一、Foundation model (基座模型)](#--foundation-model-------)

* [1.1 Google 系](#11-google--)
* [1.2  Meta系](#12--meta-)
* [1.3 OpenAI系](#13-openai-)
* [1.4 EleutherAI](#14-eleutherai)
* [1.5 其他科技公司和研究院](#15-----------)
* [:key: 训练框架和模型训练Tricks](#-key-----------tricks)

[二、Instruction-tuning model (指令微调模型)](#--instruction-tuning-model---------)
* [2.1 Google系](#21-google-)
* [2.2 Meta系](#22-meta-)
* [2.3 OpenAI系](#23-openai-)
* [2.4 EleutherAI](#24-eleutherai)
* [2.5 其他科技公司和研究院](#25-----------)
* [:key: 训练框架和模型训练Tricks](#-key-----------tricks-1)

[三、 Multimodal (多模态模型)](#---multimodal--------)

# 一、Foundation model (基座模型) 

## 1.1 Google系

Google Brain (已合并到Google DeepMind部门)

|         模型名称          |  时间   | 是否开源 | 参数规模 |                   Paper                   |                             Code                             | Introduction                                                 |
| :-----------------------: | :-----: | :------: | :------: | :---------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------- |
|            T5             | 2019-10 |    是    |   13B    | [Arxiv](https://arxiv.org/abs/1910.10683) | [github](https://github.com/google-research/text-to-text-transfer-transformer/) | [ T5 介绍](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) |
|   T5X<br /> (框架改进)    | 2022-03 |    -     |    -     | [Arxiv](https://arxiv.org/abs/2203.17189) |       [github](https://github.com/google-research/t5x)       | [Youtube](https://www.youtube.com/watch?v=lHLX81qLk_8)       |
| LaMDA <br />(ChatBot LLM) | 2021-05 |    否    |   137B   |                     -                     |                              -                               | [LaMDA介绍](https://blog.google/technology/ai/lamda/)        |
|           PaLM            | 2022-04 |    否    |   540B   | [Arxiv](https://arxiv.org/abs/2204.02311) |                              -                               | [PaLM介绍](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) |

DeepMind (已合并到Google DeepMind部门)

|     模型名称     |  时间   | 是否开源 | 参数规模 |                     Paper                     | Code | Introduction                                              |
| :--------------: | :-----: | :------: | :------: | :-------------------------------------------: | :--: | --------------------------------------------------------- |
|   Gopher(地鼠)   | 2021-12 |    否    |   280B   | [Arxiv](https://arxiv.org/pdf/2112.11446.pdf) |  -   | -                                                         |
| Chinchilla(龙猫) | 2022-04 |    否    |   70B    | [Arxiv](https://arxiv.org/pdf/2203.15556.pdf) |  -   | [Gopher介绍](https://www.youtube.com/watch?v=lHLX81qLk_8) |

Google DeepMind (23年4月合并Google Brain和DeepMind，命名为Google DeepMind)

| 模型名称 |  时间   | 是否开源 |        参数规模         | Paper | Code | Introduction                                   |
| :------: | :-----: | :------: | :---------------------: | :---: | :--: | ---------------------------------------------- |
|  PaLM 2  | 2023-05 |    否    | 340B(小道消息，未证实~) |   -   |  -   | [PaLM2介绍](https://ai.google/discover/palm2/) |

**Latest:** 更强大的模型Gemini正在训练中, [Ref](https://www.wired.com/story/google-deepmind-demis-hassabis-chatgpt/)



## 1.2  Meta系

| 模型名称 |  时间   | 是否开源 | 参数规模  |                            Paper                             |                         Code                          |
| :------: | :-----: | :------: | :-------: | :----------------------------------------------------------: | :---------------------------------------------------: |
|   OPT    | 2022-05 |    是    | 125M-175B |        [Arxiv](https://arxiv.org/pdf/2205.01068.pdf)         | [github](https://github.com/facebookresearch/metaseq) |
|  LLaMA   | 2023-02 |    是    |  7B-65B   |          [Arxiv](https://arxiv.org/abs/2302.13971)           |  [github](https://github.com/facebookresearch/llama)  |
| LLaMA 2  | 2023-07 |    是    |  7B-70B   | [Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) |  [github](https://github.com/facebookresearch/llama)  |



## 1.3 OpenAI系

![img](Roadmap%20of%20LLMs.assets/upload_af98c3d58bf03eef17312095483a78c8.png)

|           模型名称            |  时间   | 是否开源 | 参数规模  |                            Paper                             |                             Code                             |
| :---------------------------: | :-----: | :------: | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|              GPT              | 2018-06 |    是    |   117M    | [Paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) | [Hugging Face](https://huggingface.co/docs/transformers/model_doc/openai-gpt) |
|             GPT-2             | 2019-02 |    是    | 150M-1.5B | [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [Hugging Face](https://huggingface.co/docs/transformers/model_doc/gpt2) |
|             GPT-3             | 2020-05 |    否    | 125M-175B | [Wiki](https://en.wikipedia.org/wiki/GPT-3) [Arxiv](https://arxiv.org/abs/2005.14165) |
| GPT-3.5<br />(InstructionGPT) | 2022-01 |    否    |   175B    |  [Blog](https://openai.com/research/instruction-following)   |                              -                               |
|             GPT-4             | 2023-03 |    否    |   未知    | [Blog](https://openai.com/research/instruction-following) <br />[GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) |                              -                               |



## 1.4 EleutherAI 

> ![img](Roadmap%20of%20LLMs.assets/120px-EleutherAI_logo.png) https://www.eleuther.ai/

|              模型名称               |  时间   | 是否开源 | 参数规模 |                   Paper                    |                            Code                            |
| :---------------------------------: | :-----: | :------: | :------: | :----------------------------------------: | :--------------------------------------------------------: |
| GPT-Neo <br />(GPT-2 architecture ) | 2021-03 |    是    |   2.7B   | [Paper](https://zenodo.org/record/5297715) |      [github](https://github.com/EleutherAI/gpt-neo)       |
|                GPT-J                | 2021-06 |    是    |    6B    | [Paper](https://arxiv.org/abs/2101.00027)  | [Hugging Face](https://huggingface.co/EleutherAI/gpt-j-6b) |
|              GPT-NeoX               | 2022-04 |    是    |   20B    | [Paper](https://arxiv.org/abs/2204.06745)  |      [github](https://github.com/EleutherAI/gpt-neox)      |



## 1.5 其他科技公司和研究院

|          机构          |      模型名称      |  时间   | 是否开源 | 参数规模 | Paper | Code |
| :--------------------: | :----------------: | :-----: | :------: | :------: | :---: | :--: |
|       Anthropic        | Anthropic-LM v4-s3 | 2021-12 |    否    |   52B    |   -   |  -   |
| 北京智源人工智能研究院 |     天鹰Aquila     | 2023-06 |    是    |  7B/33B  |   -   |  -   |





## :key: 训练框架和模型训练Tricks

🚧 ...





# 二、Instruction-tuning model (指令微调模型)

## 2.1 Google系

| 机构                                                         |                 模型名称                  | 基座模型                  | 是否开源 | Blog & Code                                                  | Paper                                         |
| ------------------------------------------------------------ | :---------------------------------------: | ------------------------- | -------- | ------------------------------------------------------------ | --------------------------------------------- |
| <img src="README.assets/1634806038075-5df7e9e5da6d0311fd3d53f9.png" width="30px"> BigScience |                    T0                     | T5                        | 是       | [Hugging Face](https://huggingface.co/bigscience/T0)         | [Arxiv](https://arxiv.org/pdf/2110.08207.pdf) |
| Google                                                       |                   FLAN                    | T5                        | 是       | [Hugging Face](https://huggingface.co/docs/transformers/model_doc/flan-t5) | [Arxiv](https://arxiv.org/abs/2210.11416)     |
| Google                                                       |             Flan-T5/Faln-PaLM             | T5/PaLM                   | 否       | [github](https://github.com/google-research/FLAN#flan-2021-citation) | [Arxiv](https://arxiv.org/abs/2109.01652)     |
| DeepMind                                                     | **Sparrow<br />(生成人工智能聊天机器人)** | Chinchilla                | 否       | [blog](https://www.deepmind.com/blog/building-safer-dialogue-agents) | -                                             |
| Google DeepMind                                              |  **Bard<br />(生成人工智能聊天机器人)**   | 之前是LaMDA，后面是PaLM 2 | 否       | [Wiki](https://www.deepmind.com/blog/building-safer-dialogue-agents) | -                                             |



## 2.2 Meta系

| 机构     |                           模型名称                           | 基座模型     | 是否开源 | Blog & Code                                                  | Paper                                     |
| :------- | :----------------------------------------------------------: | ------------ | -------- | ------------------------------------------------------------ | ----------------------------------------- |
| Meta     |                           OPT-IML                            | OPT-175B     | 是       | [Hugging Face](https://huggingface.co/facebook/opt-iml-30b)  | [Arxiv](https://arxiv.org/abs/2212.12017) |
| Stanford | Alphaca (Alphace 7B)<br /><img src="README.assets/logo.png" width="250px"> | LLaMA-7B     | 是       | [Blog](https://crfm.stanford.edu/2023/03/13/alpaca.html)<br />[github](https://github.com/tatsu-lab/stanford_alpaca) | -                                         |
| Stanford | Vicuna (7B, 13B)<br /><img src="README.assets/vicuna-16904411351969.jpeg" width="120px"> | LLaMA-7B/13B | 是       | [Blog](https://lmsys.org/blog/2023-03-30-vicuna/)<br /><br />[github](https://github.com/lm-sys/FastChat#vicuna-weights) | -                                         |



## 2.3 OpenAI系

![image-20230727120740221](Roadmap%20of%20LLMs.assets/image-20230727120740221.png)

> Picture Ref: https://s10251.pcdn.co/wp-content/uploads/2023/03/2023-Alan-D-Thompson-GPT3-Family-Rev-1.png





## 2.4 EleutherAI 

| 模型名称     | 基座模型 | 是否开源 |                   Blog & Code                    |                   Paper                   |
| :----------- | :------- | :------- | :----------------------------------------------: | :---------------------------------------: |
| GPT-NeoX-20B | GPT-Neo  | 是       | [github](https://github.com/EleutherAI/gpt-neox) | [Arxiv](https://arxiv.org/abs/2204.06745) |



## 2.5 其他科技公司和研究院

| 机构                   |                           模型名称                           |      基座模型      | 是否开源 | Blog & Code                                                  | Paper                                     |
| :--------------------- | :----------------------------------------------------------: | :----------------: | :------- | ------------------------------------------------------------ | ----------------------------------------- |
| 北京智源人工智能研究院 | AquilaChat-7B<br />![Aquila_logo](README.assets/log-169044316851314.jpeg) |     Aquila-7B      | 是       | [Blog](https://model.baai.ac.cn/model-detail/100101)<br /><br />[Hugging Face](https://huggingface.co/BAAI/AquilaCode-multi) | -                                         |
| 北京智源人工智能研究院 | AquilaChat-33B<br />![Aquila_logo](README.assets/log-169044317529217.jpeg) |     Aquila-33B     | 是       | [Hugging Face](https://huggingface.co/BAAI/AquilaCode-multi) | -                                         |
| BigScience             |                            BLOOMZ                            |       BLOOM        | 是       | [Hugging Face](https://huggingface.co/bigscience/bloomz)     | [Arxiv](https://arxiv.org/abs/2211.01786) |
| Baidu                  |       文心一言![logo](README.assets/logoErnieInfo.png)       |     ERNIE 3.0      | 否       | [Website](https://yiyan.baidu.com/)                          | -                                         |
| Anthropic              |   Claude 2![克劳德 2](README.assets/Claude2_Blog_V1-1.png)   | Anthropic-LM v4-s3 | 否       | [Website](https://claude.ai/login)                           | -                                         |





## :key: 训练框架和模型训练Tricks

🚧 ...



# 三、 Multimodal (多模态模型)

🚧 ...







