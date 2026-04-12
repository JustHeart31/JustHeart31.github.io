## 习题

1. 自然语言处理中，语言模型经历了从统计到神经网络的模型演进。

   - 请使用本章提供的迷你语料库（`datawhale agent learns`, `datawhale agent works`），计算句子 `agent works` 在Bigram模型下的概率
        > <strong>答</strong>：1/6
   - N-gram模型的核心假设是马尔可夫假设。请解释这个假设的含义，以及N-gram模型存在哪些根本性局限？
       > <strong>答</strong>： 当前词的概率仅依赖于前 N-1 个词，本质是最大似然估计。根本局限性：无法并行计算，上下窗口受限,无法理解词义，只能匹配相同单词
   - 神经网络语言模型（RNN/LSTM）和Transformer分别是如何克服N-gram模型局限的？它们各自的优势是什么？
       > <strong>答</strong>：  RNN采用了词嵌入，克服了无法匹配词义的意思，且只能串行进行，容易发生梯度消失或梯度爆炸。Transformer引入了注意力机制，增强了并行能力，解决了上下文受限的功能
       

2. Transformer架构<sup>[4]</sup>是现代大语言模型的基础。其中：


   - 自注意力机制（Self-Attention）的核心思想是什么？
      > <strong>答</strong>：  引入了QKV机制，它允许模型在处理序列中的每一个词时，都能兼顾句子中的所有其他词，并为这些词分配不同的“注意力权重”。权重越高的词，代表其与当前词的关联性越强，其信息也应该在当前词的表示中占据更大的比重。
   - 为什么Transformer能够并行处理序列，而RNN必须串行处理？位置编码（Positional Encoding）在其中起什么作用？
     > <strong>答</strong>：   Transformer通过注意力机制来并行处理序列，而RNN的每一层的输入都依赖上一层的输出，只能串行计算。位置编码解决了语序问题，可以理解不同语序的语义
   - Decoder-Only架构与完整的Encoder-Decoder架构有什么区别？为什么现在主流的大语言模型都采用Decoder-Only架构？
     > <strong>答</strong>：  Decoder-Only架构是只根据提供的文本进行续写，而完整的Encoder-Decoder架构是先将输入文本理解，再根据其内容生成答案。使用Decoder-Only架构速度快，且根据互联网上的大量文本，预训练目标（如 next-token prediction）更简单高效；推理时只需单向生成，适合对话/续写；Encoder-Decoder 更适合翻译、摘要等输入-输出结构强对齐的任务。

3. 文本子词分词算法是大语言模型的一项关键技术，负责将文本转换为模型可处理的 token 序列。那为什么不能直接以"字符"或"单词"作为模型的输入单元？BPE（Byte Pair Encoding）算法解决了什么问题？

 > <strong>答</strong>：  直接以字符进行输入时没有语义的功能，模型需要时间来进行字符的组合。而使用单词作为输入，由于词汇数量庞大，学习过程会很复杂，并且模型无法识别没见过的单词。而BPE算法可以根据给定的词表上限来按频率组合词元来形成新的词元加入词表，当一个词元在该词表匹配失败后，可以拆分词元继续在该词表中匹配。

4. 本章3.2.3节介绍了如何本地部署开源大语言模型。请完成以下实践和分析：

   > <strong>提示</strong>：这是一道动手实践题，建议实际操作

   - 按照本章的指导，在本地部署一个轻量级的开源模型（推荐[Qwen3-0.6B](https://modelscope.cn/models/Qwen/Qwen3-0.6B)），并尝试调整采样参数并观察其对输出的影响
     > <strong>答</strong>： 
     ```Python
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 指定模型ID
        model_id = "Qwen/Qwen1.5-0.5B-Chat"

        # 设置设备，优先使用GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)  #trust_remote_code=True 是 Hugging Face Transformers 库中的一个安全选项参数

        # 加载模型，并将其移动到指定设备
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
            )

        print("模型和分词器加载完成！")

        messages = [{"role": "user", "content": "讲一个关于猫的短故事。"}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # 实验 1: 低 temperature + 低 top_p（确定性高）
        output1 = model.generate(**inputs, temperature=0.3, top_p=0.8, max_new_tokens=100)
        print("【保守输出】\n", tokenizer.decode(output1[0], skip_special_tokens=True))

        # 实验 2: 高 temperature + 高 top_p（创造性高）
        output2 = model.generate(**inputs, temperature=0.9, top_p=0.95, max_new_tokens=100)
        print("\n【创意输出】\n", tokenizer.decode(output2[0], skip_special_tokens=True))

        >>>【保守输出】
         system
        You are a helpful assistant.
        user
        讲一个关于猫的短故事。
        assistant
        从前，有一只名叫小猫的小猫，它住在一座美丽的森林里。小猫非常聪明，总是能从各种各样的事情中找到乐趣。

        有一天，小猫发现了一只小鸟在树枝上唱歌。小猫决定帮助小鸟唱歌，于是它开始用它的爪子和尾巴轻轻地敲打着树枝，让小鸟的声音更加清晰。小鸟听到了小猫的声音，就飞到树梢上唱了起来。

        小猫还发现了一个小溪，它决定去

        【创意输出】
         system
        You are a helpful assistant.
        user
        讲一个关于猫的短故事。
        assistant
        很久以前，有一个叫皮卡的小男孩。皮卡是小猫中最小的，他拥有一双明亮的眼睛和灵活的身体。他的毛色是黑白相间的，像是大地的颜色一样深沉而坚定。

        一天晚上，皮卡在树下玩耍。突然，一只狼闯入了小树林。它立刻扑向皮卡，用它的长爪抓住皮卡，把皮卡扔进了一个深不见底的洞穴里。

        皮卡非常害怕，
     ```
   - 选择一个具体任务（如文本分类、信息抽取、代码生成等），设计并对比以下不同的提示策略（如Zero-shot、Few-shot、Chain-of-Thought）对输出结果的效果差异
     > <strong>答</strong>：  
```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型ID
model_id = "Qwen/Qwen1.5-0.5B-Chat"

# 设置设备，优先使用GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)  #trust_remote_code=True 是 Hugging Face Transformers 库中的一个安全选项参数

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
    )

print("模型和分词器加载完成！")

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,  # 保持一致，避免干扰
        top_p=0.9,
        do_sample=False   # 可选：关闭采样以提高可复现性
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取模型生成部分（去掉 prompt）
    answer_part = response[len(prompt):].strip()
    return answer_part

# 测试问题
test_question = "小明有5个苹果，吃了2个，又买了3个，他现在有几个苹果？"

# 1. Zero-shot
zs_prompt = test_question + "\n答案："
zs_output = generate_answer(zs_prompt)

# 2. Few-shot
fs_prompt = """问题：小红有3个橘子，吃了1个，她还有几个？
答案：2

问题：小刚有10元，花了4元，还剩多少？
答案：6

""" + test_question + "\n答案："
fs_output = generate_answer(fs_prompt)

# 3. Chain-of-Thought
cot_prompt = """问题：小红有3个橘子，吃了1个，她还有几个？
让我们一步一步思考：她开始有3个，吃掉1个，所以剩下 3 - 1 = 2 个。
答案：2

问题：小刚有10元，花了4元，还剩多少？
让我们一步一步思考：他原有10元，花掉4元，所以剩下 10 - 4 = 6 元。
答案：6

""" + test_question + "\n让我们一步一步思考："
cot_output = generate_answer(cot_prompt)

# 打印结果
print("Zero-shot:", zs_output)
print("Few-shot:", fs_output)
print("CoT:", cot_output)

>>>输出
Zero-shot: 
Few-shot: 8
CoT: 他原有5个苹果，吃了2个，又买了3个，所以现在有 5 - 2 + 3 = 8 个苹果。
答案：8
```
   - 从性能、成本、可控性、隐私等维度比较闭源模型和开源模型
      > <strong>答</strong>：  闭源模型性能好，成本高，可控性差，隐私差。开源模型性能差，成本低，可控性好，隐私性强
   - 如果你要构建一个企业级的客服智能体，你会选择哪种类型的模型？需要考虑哪些因素？
      > <strong>答</strong>：   我会选择开源模型，需要考虑成本性能隐私等

5. 模型幻觉（Hallucination）<sup>[11]</sup>是大语言模型当前存在的关键局限性之一。本章介绍了缓解幻觉的方法（如检索增强生成、多步推理、外部工具调用）

   - 请选择其中一种，说明其工作原理和适用场景
    > <strong>答</strong>：   RAG，是通过构建索引库，当提问时，先将问题与向量库中的答案进行匹配找到相似答案再送入大模型来增强回答可靠性。适用场景是有提前构建好的索引库
   - 调研前沿的研究和论文，是否还有其他的缓解模型幻觉的方法，他们又有哪些改进和优势？
    > <strong>答</strong>：  还不知道
   
6. 假设你要设计一个论文辅助阅读智能体，它能够帮助研究人员快速阅读并理解学术论文，包括：总结论文研究的核心内容、回答关于论文的问题、提取关键信息、比较多篇不同论文的观点等。请回答：

   - 你会选择哪个模型作为智能体设计时的基座模型？选择时需要考虑哪些因素？
    > <strong>答</strong>：  选择在学术语料（如 S2ORC、arXiv）上继续预训练的模型，如Qwen-Max，因其具备更强的科技文献理解能力。”
   - 如何设计提示词来引导模型更好地理解学术论文？学术论文通常很长，可能超过模型的上下文窗口限制，你会如何解决这个问题？
    > <strong>答</strong>：  引入RAG功能。补充：分块摘要（先对各章节摘要，再综合）、滑动窗口注意力（如 LongLoRA）、层次化检索（先找相关段落，再精读）
   - 学术研究是严谨的，这意味着我们需要确保智能体生成的信息是准确客观忠于原文的。你认为系统中加入哪些设计能够更好的实现这一需求？
    > <strong>答</strong>：  提示词中严格要求智能体回答要有依据，引入大模型自检机制和引用输出机制，判断生成的答案是否能在原文中找到理论依据