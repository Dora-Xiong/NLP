FEW_SHOT = """
Question: Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Thought: Let's think step by step. Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day at the farmer\u2019s market.
Action: Finish[18]

Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
Thought: Let's think step by step. A robe takes 2/2=1 bolt of white fiber. So the total amount of fabric is 2+1=3 bolts of fabric.
Action: Finish[3]

Question: Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?"
Thought: Let's think step by step. Eliza is entitled to 45 - 40 = 5 hours overtime pay. Her hourly rate for the overtime pay is $10 x 1.2 = $12. So, Eliza will receive $12 x 5 =$60 \nfor overtime pay. Her regular weekly earning is $10 x 40 = $400. Eliza will receive a total of $400 + $60 = $460 for this week's work."
Action: Finish[460]

Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Thought: Let's think step by step. Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.
Action: Finish[72]

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Thought: Let's think step by step. Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.
Action: Finish[10]

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Thought: Let's think step by step. In the beginning, Betty has only 100 / 2 = $50.\nBetty's grandparents gave her 15 * 2 = $30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $5 more.
Action: Finish[5]"""

FEW_SHOT_REFLECTION = """
Question: Sam is a delivery driver. He has to drive 40 miles to the warehouse and then take a different route back that is 15 miles longer. If he drives 1 mile in 3 minutes and spends 1 hour unloading at the warehouse, how long will it take Sam to complete the entire delivery in hours?
Thought: Let's think step by step. Sam drove 40 miles to the warehouse and then 40 + 15 = 55 miles to return. So, the total distance is 40 + 55 = 95 miles. If it takes Sam 3 minutes to drive 1 mile, it will take him 95 x 3 = 285 minutes to cover the entire trip. Since 60 minutes equals 1 hour, Sam will be driving for 285/60 = 5 hours. Don't forget the 1 hour he spends unloading. So, the total time is 5 + 1 = 6 hours.
Action: Finish[6]

Reflection: I mistakenly calculated the driving time by multiplying the total distance (95 miles) by 3 minutes per mile, but I forgot to account for the fact that Sam's total distance should have been 40 miles to the warehouse and 55 miles back, making the total distance 40 + 55 = 95 miles. My error was in multiplying the total miles by the time per mile, but failing to convert the time correctly. In reality, Sam should have spent 285 minutes (which is 4.75 hours) driving, not 5 hours. So, the correct total time should be 4.75 hours of driving plus 1 hour unloading, giving a total of 5.75 hours.

Question: Mel is three years younger than Katherine. When Katherine is two dozen years old, how old will Mel be in years?
Thought: Let's think step by step. Katherine is 2 dozen years old, which means 2 * 12 = 24 years old. Mel is 3 years younger than Katherine, so if Katherine is 24 years old, Mel will be 24 + 3 = 27 years old.
Action: Finish[27]

Reflection: I made an error in my reasoning by adding 3 years instead of subtracting. Since Mel is 3 years younger than Katherine, I should have subtracted 3 years from Katherine’s age, not added. The correct calculation would have been 24 - 3 = 21 years old.
"""
AGENT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}

Question: {question}{scratchpad}"""

REFLECTION_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
Here are some examples:
{examples}
(END OF EXAMPLES)
Previous trial:
Question: {question}{scratchpad}

Reflection:"""

LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'
REFLECTION_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'