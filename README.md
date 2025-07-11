# oh-slurm-eval
A primer on running various eval frameworks for LLMs on slurm based clusters and specifically Euler at ETH and Science Cluster at UZH

## GPU based and Space based optimizations 
```
# Load the required modules
module load stack/2024-06 eth_proxy python/3.12.8 cuda/12.4.1 git-lfs

# Activate your environment
source /cluster/---/---/you/venv/bin/activate
export HF_HOME=/cluster/scratch/you/
export HF_DATASETS_CACHE="/cluster/scratch/you/datasets"
export TRITON_CACHE_DIR="/cluster/scratch/you/triton"
export FLASHINFER_WORKSPACE_DIR="/cluster/scratch/you/flash_infer_workspace"
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"

# Reduce VRAM usage by reducing fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

# OpenRLHF Training
## How to use custom chat templates or even the official instruct tuned templates for Base models?
You have two options - `--input_template` Or use `--tokenizer_chat_template`. The first one is simple and the second one allows you to use the official templates.
```
--tokenizer_chat_template "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in message.content %}\n                {%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}\n                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
```
where you can copy paste the template from the config files of instruct tuned models from Huggingface

Qwen comes with chat templates even for Base models so using `--apply_chat_template` will just work. But Gemma or Llama base models will complain. In those cases its super helpful while training them.


## Coming Up

- [ ] Reward Model Evaluations

## Working

### Manual
- [v] Function Call based eval
- [v] Fine Tuned Generation Parse eval

### Nemo Skills
Its a framework that I like the most :)

**How to use?**
Coming soon...

**Understanding the metrics**
### 1. Greedy Decoding
**Greedy decoding** is a method for generating text from a language model where, at each step, the model selects the token (word or symbol) with the highest probability as the next output. It does not consider alternative possibilities or look ahead, so it is "greedy" in always picking the immediate best option.  
- **In math benchmarks:** Greedy decoding produces a single answer for each problem, representing the model's most confident response.

### 2. pass@32
**pass@32** is a metric that measures the probability that at least one of 32 sampled outputs (solutions) from the model is correct.
- **How it works:** For each math problem, the model generates 32 candidate solutions (using stochastic decoding, e.g., sampling with temperature).
- **The metric:** If at least one of those 32 outputs is correct, it is considered a "pass" for that problem.
- **Purpose:** This metric estimates how likely it is that the model can solve a problem if given 32 chances.

### 3. majority@32
**majority@32** evaluates the result by looking at the 32 sampled outputs and selecting the answer that appears most frequently (the "majority" answer).
- **How it works:** The model generates 32 outputs for each problem. The answer occurring most often among those is chosen as the model's final answer.
- **The metric:** If this majority answer is correct, the problem is counted as solved.
- **Purpose:** This reflects the model's consistency and likely "best guess" when allowed to answer multiple times.

### 4. pass@1[32] (often written as pass@1 or pass@1[32])
This notation usually means evaluating "pass@1" out of 32 samples.
- **How it works:** For each problem, the model samples 32 outputs, but only the first output is evaluated.
- **The metric:** If the first output is correct, the problem is considered solved.
- **Purpose:** This is equivalent to single-shot evaluation on a batch of 32, measuring the model's immediate accuracy without retries.
- **Note:** Sometimes "pass@1[32]" can also mean running 32 independent single-sample generations and averaging the pass rates.

**Evals Supported**
...


### Skythought
**Evals Supported**
...

### SuperGPQA
Needs significant changes to get it to work. Detailed report coming soon...

**Setup and Running**

**Evaluating on a Subset**


