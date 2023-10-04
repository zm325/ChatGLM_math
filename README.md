# ChatGLM数学微调

# 介绍
ChatGLM-6b模型相对较小，在数学领域的表现较差。

使用LoRA训练基于ChatGLM的中文math大模型

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

# 说明
数据集来源：https://huggingface.co/datasets/supinyu/goat-chinese/tree/main  