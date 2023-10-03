import json
import torch
import os
import transformers
from transformers import AutoModel,Trainer,TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer import TRAINING_ARGS_NAME
from torch import nn
import datasets

skip_overlength = False # 是否跳过长文本
max_source_length = 128
max_target_length = 128
max_seq_length = max_source_length+max_target_length

data_path = "./dataset_cn.json"

# 加入一个前缀提升模型效果
prefix = "假设你是一个数学老师，请你解决下面的题目："
model_type = '../THUDM/chatglm-6b'

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_type, trust_remote_code=True)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)

def make_dataset(data_set, skip_overlength=False):
    # 加载模型配置文件
    config = transformers.AutoConfig.from_pretrained(
        model_type, trust_remote_code=True, device_map='auto')

    # 初始化 input_ids_list 和 seqlen_list 列表
    input_ids_list = []
    seqlen_list = []

    # 遍历 JSONL 文件中的每一行数据
    for data in data_set:
        prompt = data["instruction"]
        target = data["output"]
        prompt = prompt+prefix
        prompt_ids = tokenizer.encode(prompt, max_length=max_source_length, truncation=True)
        target_ids = tokenizer.encode(target, max_length=max_target_length,truncation=True,add_special_tokens=False)
        input_ids = prompt_ids + target_ids + [config.eos_token_id]

        if skip_overlength and len(input_ids) > max_seq_length:
            continue
        # 截断
        input_ids = input_ids[:max_seq_length]

        input_ids_list.append(input_ids)
        seqlen_list.append(len(prompt_ids))

    # 返回 input_ids 和 seq_len 字典
    return {"input_ids": input_ids_list,  "seq_len": seqlen_list}


# 定义 ModifiedTrainer 类，继承自 Trainer 类，保存有梯度变化的模型参数
class ModifiedTrainer(Trainer):

    # 重写 compute_loss 方法，计算模型的损失
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存训练参数
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # 保存有梯度变化的模型参数
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

# data_collator 填充字段
def data_collator(features: list) -> dict:
    # 计算每个特征的 input_ids 长度
    len_ids = [len(feature["input_ids"]) for feature in features]  #输出长度

    # 找到最长的 input_ids 长度
    longest = max(len_ids)

    # 初始化 input_ids 和 labels_list 列表
    input_ids = []
    labels_list = []

    # 遍历特征，根据需要制作我们的input和lable，
    # lable长度按seq_len截断，其余部分用[-100] 补齐，注意需要保证lable和输入长短一致
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)

    # 将 input_ids 和 labels_list 转换为张量
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)

    # 返回 input_ids 和 labels 字典
    return {
        "input_ids": input_ids,
        "labels": labels,
    }




def main():
    with open(data_path, encoding="utf-8") as f:
        data_set = json.load(f)
    data_set = data_set[:10000]
    dataset = make_dataset(data_set, skip_overlength)

    train_dataset = datasets.Dataset.from_dict(dataset)

    # 加载预训练模型
    model = AutoModel.from_pretrained(model_type, trust_remote_code=True, device_map='auto')

    # 配置模型支持梯度检查点，梯度检查点是一种以时间换空间的方法，通过减少保存的激活值压缩模型占用空间，但是在计算梯度时必须从新计算没有存储的激活值。
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()

    # 配置模型支持输入梯度
    model.enable_input_require_grads()

    # 将 lm_head 层的输出转换为浮点数
    model.lm_head = CastOutputToFloat(model.lm_head)


    # 禁用模型缓存
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # 配置lora参数
    # alpha是个缩放参数，本质和learning rate相同
    peft_config = LoraConfig(r=8,
                        lora_alpha=32,
                        lora_dropout=0.1,
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        )


    # 获取 Lora 模型
    model = get_peft_model(model, peft_config)

    # 配置模型支持并行计算
    model.is_parallelizable = True
    model.model_parallel = True

    # group_by_length是TrainingArguments中的一个参数，它用于确定是否根据序列长度对数据进行分组，如长度相似的放到一个组。
    training_args = TrainingArguments(
        "output",
        fp16=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        max_steps=1000,
        logging_steps=50,
        remove_unused_columns=False,
        num_train_epochs=1,
        seed=0,
        data_seed=0,
        group_by_length=False,
    )



    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
    # 保存模型
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()