from openai import OpenAI
import argparse
import re
from typing import List
from datasets import load_dataset
import tqdm
from utils import FEW_SHOT, FEW_SHOT_REFLECTION, AGENT_INSTRUCTION, REFLECTION_INSTRUCTION
from utils import LAST_TRIAL_HEADER, REFLECTION_HEADER

TEST_SIZE = 200

class ReflexionAgent:
    def __init__(self,
                    question: str,
                    api_key: str,
                    groundtruth: str,
                    agent_prompt: str = AGENT_INSTRUCTION,
                    reflect_prompt: str = REFLECTION_INSTRUCTION,
                    cot_examples: str = FEW_SHOT,
                    reflect_examples: str = FEW_SHOT_REFLECTION,
                    ) -> None:
        """Initializes the ReflexionAgent

        Args:
            question (str): the question to answer
            api_key (str): the api key to use
            groundtruth (str): the groundtruth answer
            agent_prompt (str, optional): the prompt for the action agent. Defaults to AGENT_INSTRUCTION.
            reflect_prompt (str, optional): the prompt for the reflection agent. Defaults to REFLECTION_INSTRUCTION.
            cot_examples (str, optional): the few-shot examples for the action model. Defaults to FEW_SHOT.
            reflect_examples (str, optional): the few-shot examples for the reflection model. Defaults to FEW_SHOT_REFLECTION.
        """
        self.question = question
        self.groundtruth = groundtruth
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.cot_examples = cot_examples 
        self.reflect_examples = reflect_examples
        self.self_reflect_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        self.action_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        self.reflections: List[str] = []
        self.reflections_str = ''
        self.answer = ''
        self.step_n: int = 0
        self.reset()

    def run(self) -> str:
        """
        Run the agent
        """
        if self.step_n > 0 and not self.is_correct():
            self.reflect()
        self.reset()
        self.step()
        self.step_n += 1
        return self.scratchpad

    def step(self) -> None:
        """
        Run a step of the agent
        """
        # Think & Act
        self.scratchpad += f'\nThought:'
        self.scratchpad += ' ' + self.prompt_agent()

        self.scratchpad += f'\nAction:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = self.parse_action(action)

        self.scratchpad += f'\nObservation: '
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
        else:
            print('Invalid action type, please try again.')
        ###########
        print(self.scratchpad)
    
    def reflect(self) -> None:
        """
        Run the reflection agent
        """
        self.reflections_str = self.format_last_attempt(self.question , self.scratchpad)
        self.reflections = [self.prompt_reflection()]
        self.reflections_str += '\n'+ self.format_reflections(self.reflections, header = REFLECTION_HEADER)
        
    def format_last_attempt(self, question: str, scratchpad: str, header: str = LAST_TRIAL_HEADER):
        return header + f'Question: {question}\n' + scratchpad.strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'
    
    def format_reflections(self, reflections: List[str], header: str = REFLECTION_HEADER) -> str:
        if reflections == []:
            return ''
        else:
            return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])
    
    def prompt_reflection(self) -> str:
        response = self.action_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": self._build_reflection_prompt()},
            ],
            stream=False
        )
        response = response.choices[0].message.content
        response = response.strip('\n').strip().replace('\n', ' ')
        return response

    def reset(self) -> None:
        """
        Reset the scratchpad and finished flag
        """
        self.scratchpad: str = ''
        self.finished = False

    def prompt_agent(self) -> str:
        response = self.action_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": self._build_agent_prompt()},
            ],
            stream=False
        )
        response = response.choices[0].message.content
        response = response.strip('\n').strip().replace('\n', ' ')
        return response
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples = self.cot_examples,
                            reflections = self.reflections_str,
                            question = self.question,
                            scratchpad = self.scratchpad)

    def _build_reflection_prompt(self) -> str:
        return self.reflect_prompt.format(
                            examples = self.reflect_examples,
                            question = self.question,
                            scratchpad = self.scratchpad)
 
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return self.answer==self.groundtruth
    
    def parse_action(self, s: str):
        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, s)
        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        else:
            return (None, None)
    

def process_gsm8k(raw_answers: list[str]) -> List[str]:
    """
    Process the raw answers from the GSM8K dataset
    """
    long_answers = []
    short_answers = []
    for answer in raw_answers:
        cleaned = re.sub(r'<<.*?>>', '', answer)
        long_answers.append(cleaned)
        match = re.search(r'####\s*(-?\d+)', cleaned)
        if match:
            short_answers.append(match.group(1))
        else:
            raise ValueError("final number is not found in answer")
    return long_answers, short_answers


def generate_prompt(dataset: str, technique: str) -> str:
    """
    Generate the prompt for the dataset and technique
    
    Args:
        dataset (str): the dataset to use. Currently only "gsm8k" is supported.
        technique (str): the technique to use. Currently only "naive", "cot", "icl" are supported.
    """
    if dataset == "gsm8k":
        if technique == "naive":
            return {
                "system": "You are a helpful assistant. Please answer the question, with the final numeric solution as the final line of your answer, preceded by ####.\n",
                "user": "Question: {}\nAnswer:"
            }
        elif technique == "cot":
            return {
                "system": "You are a helpful assistant. Please answer the question, with the final numeric solution as the final line of your answer, preceded by ####. Let's think step by step.\n",
                "user": "Question: {}\nAnswer:"
            }
        elif technique == "icl":
            return {
                "system": "You are a helpful assistant. Please answer the question, with the final numeric solution as the final line of your answer, preceded by ####.\n" + \
                    "Question: Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?" + \
                    "\n" + \
                    "Answer: Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day at the farmer\u2019s market.\n#### 18" + \
                    "\n" + \
                    "Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?" + \
                    "\n" + \
                    "Answer: A robe takes 2/2=1 bolt of white fiber. So the total amount of fabric is 2+1=3 bolts of fabric\n#### 3" + \
                    "\n" + \
                    "Question: Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?" + \
                    "\n" + \
                    "Answer: Eliza is entitled to 45 -40 = 5 hours overtime pay. Her hourly rate for the overtime pay is $10 x 1.2 = $12. So, Eliza will receive $12 x 5 =$60 \nfor overtime pay. Her regular weekly earning is $10 x 40 = $400. Eliza will receive a total of $400 + $60 = $460 for this week's work.\n#### 460",
                "user":"Question: {}\nAnswer:"
            }
    else:
        raise ValueError("invalid dataset")

def predict(prompt: str, api_key: str) -> str:
    """
    use the deepseek api to predict the answer to the prompt
    """

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        stream=False
    )

    return response.choices[0].message.content
    
def main(args):
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
        testset = dataset["test"][:TEST_SIZE]
        questions = testset['question']
        _, groundtruths = process_gsm8k(testset['answer'])
        if args.technique == "reflexion":
            predictions = []
            processes = []
            for i, question in tqdm.tqdm(enumerate(questions), total=len(questions)):
                agent = ReflexionAgent(question=question, api_key=args.api_key, groundtruth=groundtruths[i])
                process = ""
                for _ in range(3):
                    process = agent.run()
                    if agent.is_correct():
                        break
                predictions.append(agent.answer)
                processes.append(process)
                with open(f"{args.dataset}_{args.technique}.txt", "a") as f:
                    f.write(process + "\n")
                    f.write("\n")
                    
            # accuracy
            score = [1 if prediction == groundtruth else 0 for prediction, groundtruth in zip(predictions, groundtruths)]
            acc = sum(score) / len(score)
            print(f"Accuracy for {args.dataset} using {args.technique} technique: {acc}")
        else:
            predictions = []
            full_answers = []
            prompt_template = generate_prompt(args.dataset, args.technique)
            for _, question in tqdm.tqdm(enumerate(questions), total=len(questions)):
                prompt = {}
                prompt["system"] = prompt_template["system"]
                prompt["user"] = prompt_template["user"].format(question)
                response = predict(prompt, args.api_key)
                full_answers.append(response)
                match = re.search(r'####\s*(\d+)', response)
                if match:
                    predictions.append(match.group(1))
                else:
                    predictions.append("")
                # save full answers
                with open(f"{args.dataset}_{args.technique}.txt", "a") as f:
                    f.write(response + "\n")
                    f.write("\n")
            
            # accuracy
            score = [1 if prediction == groundtruth else 0 for prediction, groundtruth in zip(predictions, groundtruths)]
            acc = sum(score) / len(score)
            print(f"Accuracy for {args.dataset} using {args.technique} technique: {acc}")
            
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["gsm8k"], help="name of the dataset to use")
    parser.add_argument("--technique", type=str, choices=["naive", "cot", "icl", "reflexion"], help="prompt technique to use")
    parser.add_argument("--api_key", type=str, help="deepseek api key")
    args = parser.parse_args()
    main(args)
    