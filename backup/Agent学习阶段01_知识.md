# 一、什么是agent（智能体）

智能体被定义为任何能够通过传感器（Sensors）感知其所处环境（Environment），并自主地通过执行器（Actuators）采取行动（Action）以达成特定目标的实体。

---


# 二、智能体的构成和运行原理

智能体的核心在于理解其任务环境并遵循一套循环的运行机制。

#### 1. 任务环境定义 (PEAS 模型)
要理解智能体，首先要明确其任务环境，通常使用 **PEAS 模型**进行规约：
*   **性能度量 (Performance)**
*   **环境 (Environment)**
*   **执行器 (Actuators)**
*   **传感器 (Sensors)**

LLM 智能体所处的数字环境通常具备以下复杂特性：
*   **部分可观察 (Partially Observable)**：无法一次性获取全部信息，需要智能体具备记忆和探索能力。
*   **随机性 (Stochastic)**：行动结果不确定，要求智能体能处理不确定性并及时决策。
*   **多智能体 (Multi-agent)**：环境中存在其他行动者（如其他用户、系统），它们的行动会改变环境状态。
*   **序贯且动态 (Sequential & Dynamic)**：当前行动影响未来，且环境自身会变化，要求智能体能快速适应。

#### 2. 智能体的运行机制 (智能体循环)
智能体通过一个持续的 **“感知-思考-行动-观察”** 循环与环境交互，逐步完成任务。

1.  **感知 (Perception)**：通过传感器接收环境信息（即**观察 Observation**），如用户指令或上一步行动的反馈。
2.  **思考 (Thought)**：核心决策阶段，由大语言模型驱动，可细分为：
    *   **规划 (Planning)**：分解任务，制定或调整行动计划。
    *   **工具选择 (Tool Selection)**：选择执行下一步所需的工具及参数。
3.  **行动 (Action)**：通过执行器调用工具，对环境施加影响。
4.  **观察 (Observation)**：行动导致环境状态变化，产生新的反馈信息，该信息将作为下一轮循环的输入。

#### 3. 智能体的感知与行动 (交互协议)
在工程实践中，通过一套明确的交互协议来规范智能体与环境的交互。智能体的输出被结构化为：

*   **Thought (思考)**：以自然语言形式阐述其分析、反思和规划过程。
*   **Action (行动)**：以函数调用等形式表示的具体操作指令。

行动执行后，环境返回的原始数据（如 JSON）会被感知系统处理成简洁的 **Observation (观察)** 文本，反馈给智能体，从而驱动下一轮循环。通过这种严谨的循环，LLM 智能体将内部推理与外部工具操作有效结合。


# 三 、动手体验：5 分钟实现第一个智能体

在前面的小节，我们学习了智能体的任务环境、核心运行机制以及 `Thought-Action-Observation` 交互范式。理论知识固然重要，但最好的学习方式是亲手实践。在本节中，我们将引导您使用几行简单的 Python 代码，从零开始构建一个可以工作的智能旅行助手。这个过程将遵循我们刚刚学到的理论循环，让您直观地感受到一个智能体是如何“思考”并与外部“工具”互动的。让我们开始吧！

在本案例中，我们的目标是构建一个能处理分步任务的智能旅行助手。需要解决的用户任务定义为："你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"要完成这个任务，智能体必须展现出清晰的逻辑规划能力。它需要先调用天气查询工具，并将获得的观察结果作为下一步的依据。在下一轮循环中，它再调用景点推荐工具，从而得出最终建议。

### 3.1 准备工作

为了能从 Python 程序中访问网络 API，我们需要一个 HTTP 库。`requests`是 Python 社区中最流行、最易用的选择。`tavily-python`是一个强大的 AI 搜索 API 客户端（注意要关闭vpn)，用于获取实时的网络搜索结果，可以在[官网](https://www.tavily.com/)注册后获取 API。`openai`是 OpenAI 官方提供的 Python SDK，用于调用 GPT 等大语言模型服务。请先通过以下命令安装它们：：

```bash
pip install requests tavily-python openai
```

（1）指令模板

驱动真实 LLM 的关键在于<strong>提示工程（Prompt Engineering）</strong>。我们需要设计一个“指令模板”，告诉 LLM 它应该扮演什么角色、拥有哪些工具、以及如何格式化它的思考和行动。这是我们智能体的“说明书”，它将作为`system_prompt`传递给 LLM。

```
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""
```

（2）工具 1：查询真实天气

我们将使用免费的天气查询服务 `wttr.in`，它能以 JSON 格式返回指定城市的天气数据。下面是实现该工具的代码：

```python
import requests

def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status() 
        # 解析返回的JSON数据
        data = response.json()
        
        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        
        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
        
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"
```

（3）工具 2：搜索并推荐旅游景点

我们将定义一个新工具 `search_attraction`，它会根据城市和天气状况，互联网上搜索合适的景点：

```python
import os
from tavily import TavilyClient

def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)
    
    # 3. 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    
    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]
        
        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
             return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"
```

最后，我们将所有工具函数放入一个字典，供主循环调用：

```python
# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
```



### 3.2 接入大语言模型

当前，许多 LLM 服务提供商（包括 OpenAI、Azure、以及众多开源模型服务框架如 Ollama、vLLM 等）都遵循了与 OpenAI API 相似的接口规范。这种标准化为开发者带来了极大的便利。智能体的自主决策能力来源于 LLM。我们将实现一个通用的客户端 `OpenAICompatibleClient`，它可以连接到任何兼容 OpenAI 接口规范的 LLM 服务。

```python
from openai import OpenAI

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"
```

要实例化此类，您需要提供三个信息：`API_KEY`、`BASE_URL` 和 `MODEL_ID`，具体值取决于您使用的服务商（如 OpenAI 官方、Azure、或 Ollama 等本地模型），如果暂时没有渠道获取，可以参考 [环境配置](https://github.com/datawhalechina/hello-agents/blob/main/Extra-Chapter/Extra07-环境配置.md)。

### 3.3 执行行动循环

下面的主循环将整合所有组件，并通过格式化后的 Prompt 驱动 LLM 进行决策。

```python
import re

# --- 1. 配置LLM客户端 ---
# 请根据您使用的服务，将这里替换成对应的凭证和地址,关闭vpn
API_KEY = "YOUR_API_KEY"
BASE_URL = "YOUR_BASE_URL"
MODEL_ID = "YOUR_MODEL_ID"
TAVILY_API_KEY="YOUR_Tavily_KEY"
os.environ['TAVILY_API_KEY'] = "YOUR_TAVILY_API_KEY"    

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. 初始化 ---
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "="*40)

# --- 3. 运行主循环 ---
for i in range(5): # 设置最大循环次数
    print(f"--- 循环 {i+1} ---\n")
    
    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)
    
    # 3.2. 调用LLM进行思考
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # 模型可能会输出多余的Thought-Action，需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)
    
    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break
    
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)
```

通过以上步骤，我们构建了一个完整的、由真实 LLM 驱动的智能体。其核心在于“工具”和“提示工程”的结合，这正是当前主流智能体框架（如 LangChain、LlamaIndex 等）的设计精髓。

### 3.4 运行案例分析

以下输出完整地展示了一个成功的智能体执行流程。通过对这个三轮循环的分析，我们可以清晰地看到智能体解决问题的核心能力。

```bash
用户输入: 你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。
========================================
--- 循环 1 ---

正在调用大语言模型...
大语言模型响应成功。
模型输出:
Thought: 首先需要获取北京今天的天气情况，之后再根据天气情况来推荐旅游景点。
Action: get_weather(city="北京")

Observation: 北京当前天气:Sunny，气温26摄氏度
========================================      
--- 循环 2 ---

正在调用大语言模型...
大语言模型响应成功。
模型输出:
Thought: 现在已经知道了北京今天的天气是晴朗且温度适中，接下来可以基于这个信息来推荐一个适合的旅游景点了。
Action: get_attraction(city="北京", weather="Sunny")

Observation: 北京在晴天最值得去的旅游景点是颐和园，因其美丽的湖景和古建筑。另一个推荐是长城，因其壮观的景观和历史意义。
========================================
--- 循环 3 ---

正在调用大语言模型...
大语言模型响应成功。
模型输出:
Thought: 已经获得了两个适合晴天游览的景点建议，现在可以根据这些信息给用户提供满意的答复。
Action: Finish[今天北京的天气是晴朗的，气温26摄氏度，非常适合外出游玩。我推荐您去颐和园欣赏美丽的湖景和古建筑，或者前往长城体验其壮观的景观和深厚的历史意义。希望您有一个愉快的旅行！]

任务完成，最终答案: 今天北京的天气是晴朗的，气温26摄氏度，非常适合外出游玩。我推荐您去颐和园欣赏美丽的湖景和古建筑，或者前往长城体验其壮观的景观和深厚的历史意义。希望您有一个愉快的旅行！
```

这个简单的旅行助手案例，集中演示了基于`Thought-Action-Observation`范式的智能体所具备的四项基本能力：任务分解、工具调用、上下文理解和结果合成。正是通过这个循环的不断迭代，智能体才得以将一个模糊的用户意图，转化为一系列具体、可执行的步骤，并最终达成目标。

---

# 四、智能体应用
基于智能体在任务中的角色和自主性程度，其协作模式主要分为两种：一种是作为高效工具，深度融入我们的工作流；另一种则是作为自主的协作者，与其他智能体协作完成复杂目标。

# 4.1.作为开发者工具的智能体
智能体被深度集成到开发者的工作流中，作为一种强大的辅助工具。比如：
* GitHubCopilot
* Claude Code
* Trae
* Cursor

# 4.2.作为自主协作者的智能体
在这种模式下，我们不再是手把手地指导 AI 完成每一步，而是将一个高层级的目标委托给它。智能体会像一个真正的项目成员一样，独立地进行规划、推理、执行和反思，直到最终交付成果。
* <strong>单智能体自主循环</strong>：
* <strong>多智能体协作</strong>：
这是当前最主流的探索方向，旨在通过模拟人类团队的协作模式来解决复杂问题。它又可细分为不同模式： 角色扮演式对话：如 CAMEL 框架，通过为两个智能体（例如，“程序员”和“产品经理”）设定明确的角色和沟通协议，让它们在一个结构化的对话中协同完成任务。 组织化工作流：如 MetaGPT 和 CrewAI，它们模拟一个分工明确的“虚拟团队”（如软件公司或咨询小组）。每个智能体都有预设的职责和工作流程（SOP），通过层级化或顺序化的方式协作，产出高质量的复杂成果（如完整的代码库或研究报告）。AutoGen 和 AgentScope 则提供了更灵活的对话模式，允许开发者自定义智能体间的复杂交互网络。
* <strong>Workflow 和 Agent 的差异</strong>：Workflow 是让 AI 按部就班地执行指令，而 Agent 则是赋予 AI 自由度去自主达成目标。