from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.transformers_utils.detokenizer import convert_prompt_ids_to_tokens
import json, sys, yaml
from tqdm import tqdm

# eval_file = "eval.json"  # default
if len(sys.argv) > 1 and sys.argv[1] == "small":
    eval_file = "eval_small.json"

sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=2048)

# Initialize the vLLM engine
model = "/cluster/scratch/---/Qwen3-4B-Base-epoch_1"
llm = LLM(model=model)


def load_config(path):
    return yaml.safe_load(open(path))
config = load_config("config.yaml")

with open(eval_file) as f:
    eval_data = json.load(f)


prompts = []
for item in tqdm(eval_data):
    query = item['query']
    extract = item['extract']
    entity = item['company_name']
    # assert that the extract is a list of strings
    assert isinstance(extract, list), f"Extract should be a list, got {type(extract)}"
    assert all(isinstance(e, str) for e in extract), "All elements in extract should be strings"
    
    prompt = f"Question: {query}\n\nContext: {'\n'.join(extract)}\n\n"
    # prompt = f"Question: {query}\n\nEntity: {entity}\n\nContext: {'\n'.join(extract)}\n\n"
    messages = [
        # {"role": "system", "content": config[config['system_type']]},
        {"role": "user", "content": prompt}
    ]
    prompts.append(messages)

print(f"Number of prompts: {len(prompts)}")

batch_size = 1
batched_responses = []
THINKING = False  # Set to True if you want to enable thinking mode

print("Starting generation...")
for batch in prompts:
    responses = llm.chat(
        batch,
        sampling_params,
        chat_template_kwargs={"enable_thinking": THINKING}
    )
    batched_responses.extend(responses)

print(f"Length of responses = {len(batched_responses)}")   

def parse_response(response):
    """Extracts the generated text from the response."""
    score = response.split('Score:')[-1].strip()
    try:
        score = int(float(score))
    except ValueError:
        print(f"Failed to parse score from response: {response}")
    return score


analysis = []
for response, input_ in zip(batched_responses, eval_data):
    human_score = input_['score']
    model_score = parse_response(response.outputs[0].text)
    study = {
        "query": input_['query'],
        "extract": input_['extract'],
        "human_score": human_score,
        "gen_score": model_score,
        "model_response": response.outputs[0].text.strip(),
    }
    analysis.append(study)

accuracy = []
hit_rate = [] # if the signs are same for + and - stance, for 0 it must be zero
two_acc = []
one_acc = []
zero_acc = []
minus_one_acc = []
minus_two_acc = []
bad_gen = 0
for gen in analysis:
    if gen['gen_score'] == gen['human_score'] and gen['gen_score'] == 0:
        hit_rate.append(True)
    if (int(gen['gen_score']))*(int(gen['human_score'])) > 0:
        hit_rate.append(True)
    else:
        hit_rate.append(False)
    if gen['gen_score'] == gen['human_score']:
        accuracy.append(True)
        if gen['gen_score'] == 2:
            two_acc.append(True)
        elif gen['gen_score'] == 1:
            one_acc.append(True)
        elif gen['gen_score'] == 0:
            zero_acc.append(True)
        elif gen['gen_score'] == -1:
            minus_one_acc.append(True)
        elif gen['gen_score'] == -2:
            minus_two_acc.append(True)
    else:
        accuracy.append(False)
        if gen['gen_score'] == 2:
            two_acc.append(False)
        elif gen['gen_score'] == 1:
            one_acc.append(False)
        elif gen['gen_score'] == 0:
            zero_acc.append(False)
        elif gen['gen_score'] == -1:
            minus_one_acc.append(False)
        elif gen['gen_score'] == -2:
            minus_two_acc.append(False)
    
    if gen['gen_score'] == -99:
        bad_gen+=1

print("-----------------------------------------")
print("Model - ", model)
print("System Type: ", config[config['system_type']])
print("Temperature - ", sampling_params.temperature)
print("Data for eval - ", len(prompts))
print(f"Thinking - {THINKING}")  
print("-----------------------------------------")
print(f"Accuracy: {sum(accuracy)/len(accuracy)}")
print(f"Hit Rate: {sum(hit_rate)/len(hit_rate)}")
print("-----------------------------------------")
print(f"Bad Gen: {bad_gen} out of {len(accuracy)}")
print("-----------------------------------------")
print(f"Stance 2: {sum(two_acc)/len(two_acc)}")
print(f"Stance 1: {sum(one_acc)/len(one_acc)}")
print(f"Stance 0: {sum(zero_acc)/len(zero_acc)}")
print(f"Stance -1: {sum(minus_one_acc)/len(minus_one_acc)}")
print(f"Stance -2: {sum(minus_two_acc)/len(minus_two_acc)}")
print("-----------------------------------------")

# Print the outputs.
# for output in outputs:
#     print(output)
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
#     decoded = convert_prompt_ids_to_tokens(llm.get_tokenizer(), output.prompt_token_ids)
#     print(f"Decoded prompt: {decoded}")
#     print(f"Decoded output: {convert_prompt_ids_to_tokens(llm.get_tokenizer(), output.outputs[0].token_ids)}")