import re

def format_reward_func(completions, **kwargs):
    pattern = r"\n#### The final answer is \d+"    
    matches = [re.search(pattern, completion) for completion in completions]
    return [0.5 if match else 0.0 for match in matches]

def correctness_reward_func(completions, final_answer, **kwargs):
    rewards = []
    
    for completion, ground_truth in zip(completions, final_answer):
        try:
            match = re.search(r'####.*?([\d,]+(?:\.\d+)?)', completion)
            if match:
                answer = match.group(1)
                answer = _clean_answer(answer)
                
                if abs(float(answer) - float(ground_truth)) < 1e-3:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        except ValueError:
            rewards.append(0.0)
            
    return rewards

def _clean_answer(answer):
    for char in [',', '$', '%', 'g']:
        answer = answer.replace(char, '')
    return answer

def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters: {all_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / all_params:.2f}%")