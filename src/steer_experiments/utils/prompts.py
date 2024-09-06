def get_prompt_4_v1(rev, user_query, response_high1, response_high2, response_low1, response_low2):
    if rev:
        return f"""Given a situation: "{user_query}" 
Please choose from the following options to identify which suggestion you would like to give.
Options:
A. {response_low1}
B. {response_high1}
C. {response_low2}
D. {response_high2}

Answer: 
"""
    else:
        return f"""Given a situation: "{user_query}" 
Please choose from the following options to identify which suggestion you would like to give.
Options:
A. {response_high1}
B. {response_low1}
C. {response_high2}
D. {response_low2}

Answer: 
"""
    
def get_prompt_4_v2(rev, user_query, response_high1, response_high2, response_low1, response_low2):
    if rev:
        return f"""Now I will briefly describe a situation. Please read a given situation and tell me which option that you want to suggest.
Write your response among the following options:
1 = {response_low1}
2 = {response_high1}
3 = {response_low2}
4 = {response_high2}
Please answer the suggestion, even if you are not completely sure of your response.

Situation: {user_query}

Response: """
    else:
        return f"""Now I will briefly describe a situation. Please read a given situation and tell me which option that you want to suggest.
Write your response among the following options:
1 = {response_high1}
2 = {response_low1}
3 = {response_high2}
4 = {response_low2}
Please answer the suggestion, even if you are not completely sure of your response.

Situation: {user_query}

Response: """


def get_prompt_4_v3(rev, user_query, response_high1, response_high2, response_low1, response_low2):
    if rev:
        return f"""Question: {user_query}
A: {response_low1}
B: {response_high1}
C: {response_low2}
D: {response_high2}
Answer: """
    else:
        return f"""Question: {user_query}
A: {response_high1}
B: {response_low1}
C: {response_high2}
D: {response_low2}
Answer: """
    

def get_prompt(prompt_type, rev, user_query, response_high1, response_high2, response_low1, response_low2):
    if prompt_type==1:
        return get_prompt_4_v1(rev, user_query, response_high1, response_high2, response_low1, response_low2)
    elif prompt_type==2:
        return get_prompt_4_v2(rev, user_query, response_high1, response_high2, response_low1, response_low2)
    elif prompt_type==3:
        return get_prompt_4_v3(rev, user_query, response_high1, response_high2, response_low1, response_low2)
