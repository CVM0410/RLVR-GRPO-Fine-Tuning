import logging
import random
import re
from resources import EXAMPLARS
from datasets import load_dataset

logger = logging.getLogger(__name__)

class GSM8K:
    def __init__(self, split, include_answer=True, include_reasoning=True, few_shot=False, num_shots=8, seed=None, cot=False, template="qa"):
        self.split = split
        self.include_answer = include_answer
        self.include_reasoning = include_reasoning
        self.seed = seed
        self.few_shot = few_shot
        self.num_shots = num_shots
        self.cot = cot
        self.template = template
        self.examples = None

        if self.seed is not None:
            random.seed(self.seed)

        self.dataset = self.load_dataset()

    def format_example(self, question, solution, answer):
        if self.template != 'qa':
            raise ValueError('Format Not Implemented')

        example = f"Question: {question}\nSolution: "
        
        if self.cot:
            example += "Let's break it down step by step. "
        
        if solution is not None:
            solution = '. '.join(solution.split('\n'))
            solution = self._remove_placeholders(solution)
            example += f"{solution}.\n"
        
        example = example.replace('..', '.')
        
        if answer is not None:
            example += f"#### The final answer is {answer}\n\n"
        
        return example

    def _remove_placeholders(self, text):
        return re.sub(r'<<.*?>>', '', text)

    def process_example(self, example, index):
        question = example['question']
        answer = example['answer']
        
        reasoning, final_answer = self._extract_reasoning_and_answer(answer)
        
        input_text = self._create_prompt(question, reasoning, final_answer)
        
        return {
            'prompt': input_text,
            'final_answer': final_answer,
            'question': question,
        }

    def _extract_reasoning_and_answer(self, answer):
        answer_delim = "#### "
        if answer_delim in answer:
            reasoning = answer.split(answer_delim)[0].strip()
            final_answer = answer.split(answer_delim)[-1].strip()
        else:
            reasoning = answer.strip()
            final_answer = ''
        return reasoning, final_answer

    def _create_prompt(self, question, reasoning, final_answer):
        if self.include_answer:
            if self.include_reasoning:
                input_text = self.format_example(question, reasoning, final_answer)
            else:
                input_text = self.format_example(question, None, final_answer)
        else:
            input_text = self.format_example(question, None, None)

        if self.few_shot:
            input_text = self.few_shot_prompt + input_text

        return input_text

    def load_dataset(self):
        dataset = load_dataset('gsm8k', 'main', split=self.split)
        
        if self.few_shot:
            self.few_shot_prompt = self.build_prompt()

        dataset = dataset.map(self.process_example, with_indices=True, load_from_cache_file=False)
        return dataset

    def fewshot_examples_qa(self):
        return EXAMPLARS

    def make_prompts(self):
        if self.template != 'qa':
            raise ValueError('Format Not Implemented')
        self.examples = self.fewshot_examples_qa()

    def build_prompt(self):
        if self.examples is None:
            self.make_prompts()
                
        prompt = ""
        for qna in random.sample(self.examples, self.num_shots):
            prompt += self.format_example(qna['question'], qna['cot_answer'], qna['short_answer'])
        return prompt