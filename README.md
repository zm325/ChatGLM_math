# ChatGLM数学微调

# 介绍
训练基于ChatGLM的中文math大模型

模型加入了prefix。

# 使用
训练
```
cd ./LoRA
python chatglm_lora_math.py
```

推理
```
python lora_eval_web.py
```

数据集来源：huggingface中文开源数据集

