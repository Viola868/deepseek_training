# —— 通用权重同设备补丁 ——  
import torch
import os
torch.cuda.empty_cache()
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    #prepare_model_for_int8_training,   
    TrainingArguments,
    Trainer
)
#from accelerate import dispatch_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
import json
#SWANLAB_HOST = os.environ.get('SWANLAB_HOST')
#SWANLAB_API_KEY = os.environ.get('SWANLAB_API_KEY')
import swanlab  # SwanLab 实验跟踪
import datasets  # 用于加载 JSON 格式训练数据
from datasets import Dataset
import transformers  # Transformers 库
from swanlab.integration.huggingface import SwanLabCallback



#SWANLAB_HOST = os.getenv('SWANLAB_HOST', 'https://swanlab.cn')
#SWANLAB_API_KEY = os.getenv('SWANLAB_API_KEY', 'JtEPVvIMaPOLhVbHXivVl')
#if not SWANLAB_HOST or not SWANLAB_API_KEY:
#    raise ValueError(
#        "请先设置环境变量 SWANLAB_HOST 和 SWANLAB_API_KEY，或使用 "
#        "swanlab login CLI 进行登录。"
 #   )

def main():
    # 初始化 SwanLab 实验日志
    #swanlab.init('deepseek_train')
    """swanlab.init(
        experiment_name='deepseek_train',
        mode='local',                  # 本地模式，不再尝试访问云端
        logdir='/share/home/zhangshanqi/pyn/shopping_behavior2/models/pretrained/output/swan_logs',  # 日志存放目录
        #web_host='https://swanlab.cn',
        #host='https://swanlab.cn/api',
        #api_key='JtEPVvIMaPOLhVbHXivVl',
        config_dir='/share/home/zhangshanqi/.swanlab',
        project_name='deepseek_train',       # 或你在 SwanLab 上建的项目名
        workspace_name='Viola_P'      # 如果你的工作区是这个名字
    )"""

    # 加载 JSONL 格式对话数据，每条记录包含一个 "messages" 列
    data_path = '/share/home/zhangshanqi/pyn/shopping_behavior2/models/traning_data/fixed_dataset.jsonl'
    records = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # 2. 用 from_list 构造 Dataset，彻底绕过本地 Arrow 缓存
    raw = Dataset.from_list(records)
    #raw = Dataset.from_json(data_path)
    """raw = datasets.load_dataset(
        'json',
        data_files='/share/home/zhangshanqi/pyn/shopping_behavior2/models/traning_data/fixed_dataset.jsonl',  # 修改为你的文件路径
        split='train'
    )"""
    # 划分训练/验证集
    splits = raw.train_test_split(test_size=0.1, seed=50)
    train_ds = splits['train']
    eval_ds  = splits['test']
    
    # 1. 先拿到本进程应该用哪张卡
    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #device = torch.device(f"cuda:{local_rank}")
    # 3. 构建一个“全部 CPU + Embedding 到本卡”的 device_map
    #    注意：这里我们先用 meta model 快速获取所有模块名，然后标记 Embedding
    #from transformers.modeling_utils import _get_submodules_from_module
    #from torch import nn

    # 用 meta model 列举所有模块路径
    # 加载预训练配置、Tokenizer 与模型
    model_path = '/share/home/zhangshanqi/pyn/shopping_behavior2/llm/deepseek_R1'  # 本地模型路径
    #bnb_config = BitsAndBytesConfig(load_in_8bit=True, load_in_8bit_cpu=True,llm_int8_enable_fp32_cpu_offload=True)
    config    = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    #bnb_config = BitsAndBytesConfig(load_in_8bit=True,llm_int8_skip_modules=["lm_head"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit量化
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # 双量化进一步压缩
        bnb_4bit_compute_dtype=torch.bfloat16  # 计算时使用bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        quantization_config=bnb_config,
        device_map=None,
        #device_map=device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
           
    )
    model = model.to(torch.device("cuda"))  # 手动移动模型到GPU
    #model     = AutoModelForCausalLM.from_pretrained(model_path, config=config, quantization_config=bnb_config,  device_map={"": "cpu"})
    # —— 2. 准备在 8-bit 模型上微调
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,                      # LoRA 降秩维度
        lora_alpha=16,            # LoRA alpha
        target_modules=["q_proj", "v_proj"], 
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    #使用 get_peft_model（） 包装模型后，必须使用 accelerate 中的 dispatch_model（） 手动重新调度它。此步骤是必不可少的，因为 get_peft_model（） 不遵循现有device_map
    #model = dispatch_model(model, device_map="auto")
    model.gradient_checkpointing_enable()

    # 把输入嵌入层权重搬到当前 GPU
    #model.set_input_embeddings(
    #    model.get_input_embeddings().to(device)
    #)

    # 如果你的模型还有显式的位置嵌入，也一起搬过去（可选）
    # 这里以 GPT 风格模型（transformer.wpe）为例：
    if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
        model.transformer.wpe = model.transformer.wpe.to(device)
    # 验证只有 adapter 参数是可训练的
    model.print_trainable_parameters()
    #model.gradient_checkpointing_enable()
    # 确保存在 pad_token（用于 batch padding）
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

# 最大文本长度
    MAX_LENGTH = 1024
    # 1. 提取整个对话上下文作为 prompt，最后一条 assistant 作为 label
    def extract_last_pair(example):
        msgs = example['messages']
        # prompt：合并除最后一条 assistant 之外的所有消息
        prompt = ''
        for m in msgs[:-1]:
            role = m['role']
            content = m['content']
            if role == 'system':
                prompt += content + '\n'
            elif role == 'user':
                prompt += 'User: ' + content + '\n'
            elif role == 'assistant':
                prompt += 'Assistant: ' + content + '\n'
        # response：最后一条 assistant 的内容
        response = msgs[-1]['content']
        return {'prompt': prompt, 'response': response}
    # 2. 批量分词：prompt 和 response 都做截断与 padding
    def preprocess_batch(batch):
        prompts = batch['prompt']
        responses = batch['response']
        # 对 prompt 编码
        model_inputs = tokenizer(
            prompts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )
        # 对 response 编码为 labels
        labels = tokenizer(
            responses,
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )['input_ids']
        # 将 pad_token_id 替换为 -100，不计入损失
        labels = [ [ (l if l != tokenizer.pad_token_id else -100) for l in seq ] for seq in labels ]
        model_inputs['labels'] = labels
        return model_inputs

    # 3. 提取 prompt/response 对
    train_ds = train_ds.map(
        extract_last_pair,
        batched=False,
        remove_columns=train_ds.column_names
    )
    eval_ds = eval_ds.map(
        extract_last_pair,
        batched=False,
        remove_columns=eval_ds.column_names
    )

    # 4. 批量分词并生成 labels
    train_ds = train_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=['prompt','response'],
        num_proc=4
    )
    eval_ds = eval_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=['prompt','response'],
        num_proc=4
    )

    # 5. 定义 DataCollator，用于动态 padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False
    )

    # 6. 训练参数配置
    training_args = TrainingArguments(
        output_dir='/share/home/zhangshanqi/pyn/shopping_behavior2/models/fine_tuned',
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        evaluation_strategy='steps',
        eval_steps=500,
        logging_steps=100,
        save_strategy='epoch',
        learning_rate=1e-4,
        weight_decay=0.1,
        warmup_steps=int(0.1 * len(train_ds)),
        fp16=True,
        gradient_checkpointing=True,
        deepspeed="/share/home/zhangshanqi/pyn/shopping_behavior2/models/pretrained/ds_config.json",
        optim="adamw_bnb_8bit"          # 使用8-bit优化器
    )

    os.makedirs('/share/home/zhangshanqi/pyn/shopping_behavior2/models/pretrained/output/swan_logs', exist_ok=True)
    swanlab.init(
        experiment_name='deepseek_train',
        mode='local',                  # 本地模式，不再尝试访问云端
        logdir='/share/home/zhangshanqi/pyn/shopping_behavior2/models/pretrained/output/swan_logs',  # 日志存放目录
        #web_host='https://swanlab.cn',
        #host='https://swanlab.cn/api',
        #api_key='JtEPVvIMaPOLhVbHXivVl',
        config_dir='/share/home/zhangshanqi/.swanlab',
        project_name='deepseek_train',       # 或你在 SwanLab 上建的项目名
        workspace_name='Viola_P'      # 如果你的工作区是这个名字
    )
    # 7. 初始化 Trainer 并训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[SwanLabCallback()]
    )
    torch.cuda.empty_cache()
    trainer.train()

    # 8. 保存微调后模型
    model.save_pretrained('/share/home/zhangshanqi/pyn/shopping_behavior2/models/fine_tuned')


if __name__ == '__main__':
    main()
