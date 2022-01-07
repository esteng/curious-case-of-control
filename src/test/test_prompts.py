import pathlib 

path_to_file = pathlib.Path("").absolute()
path_to_src = path_to_file.parent.joinpath("src")
print(path_to_src)
import sys
sys.path.insert(0, str(path_to_src))

import pytest 
from api_tools import FixedPrompt, FixedGPTPrompt, FixedPassiveGPTPrompt, FixedT5Prompt, FixedPassiveT5Prompt

def check(prompt, expected, raise_error=True):
    try:
        assert(str(prompt) == expected)
    except:
        print(str(prompt))
        print()
        print(expected)
        if raise_error:
            raise AssertionError
        return False
    return True

@pytest.fixture
def names():
    return "Avery", "Casey"

@pytest.fixture
def object_control():
    return "told", "to come", "came"

@pytest.fixture
def subject_control():
    return "promised", "to come", "came"

## Object control
def test_gpt_object_control_names_long_instructions(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question. Answer the question with either "Avery" or "Casey".
Context: Avery told Casey to come.

Question: Who came, Avery or Casey?
Answer: """

    prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = False, long_instructions=True)
    check(prompt, expected)

def test_gpt_object_control_names_long_instructions_swap(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question. Answer the question with either "Casey" or "Avery".
Context: Avery told Casey to come.

Question: Who came, Casey or Avery?
Answer: """

    num_correct = 0 
    # stochastic, so do it a few times 
    for i in range(20):
        prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = True, long_instructions=True)
        if check(prompt, expected, raise_error=False): 
            num_correct += 1
    assert(num_correct > 6)

def test_gpt_object_control_names_short_instructions(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question.
Context: Avery told Casey to come.

Question: Who came?
Answer: """

    prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = False)
    check(prompt, expected)

def test_gpt_subject_control_names_long_instructions(names, subject_control):
    n1, n2 = names
    verb, action, past = subject_control
    expected = """You will be given a context and a question. Answer the question with either "Avery" or "Casey".
Context: Avery promised Casey to come.

Question: Who came, Avery or Casey?
Answer: """

    prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = False, long_instructions=True)
    check(prompt, expected)

def test_gpt_subject_control_names_long_instructions_swap(names, subject_control):
    n1, n2 = names
    verb, action, past = subject_control
    expected = """You will be given a context and a question. Answer the question with either "Casey" or "Avery".
Context: Avery promised Casey to come.

Question: Who came, Casey or Avery?
Answer: """

    num_correct = 0 
    # stochastic, so do it a few times 
    for i in range(20):
        prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = True, long_instructions=True)
        if check(prompt, expected, raise_error=False): 
            num_correct += 1

    assert(num_correct > 6)

def test_gpt_object_control_names_short_instructions(names, subject_control):
    n1, n2 = names
    verb, action, past = subject_control
    expected = """You will be given a context and a question.
Context: Avery promised Casey to come.

Question: Who came?
Answer: """

    prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = False)
    check(prompt, expected)

## Passive object control 
def test_passive_gpt_object_control_names_long_instructions(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question. Answer the question with either "Avery" or "Casey".
Context: Avery was told by Casey to come.

Question: Who came, Avery or Casey?
Answer: """

    prompt = FixedPassiveGPTPrompt(n1, n2, verb, action, past, swap_names = False, long_instructions=True)
    check(prompt, expected)

def test_passive_gpt_object_control_names_long_instructions_swap(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question. Answer the question with either "Casey" or "Avery".
Context: Avery was told by Casey to come.

Question: Who came, Casey or Avery?
Answer: """

    num_correct = 0 
    # stochastic, so do it a few times 
    for i in range(20):
        prompt = FixedPassiveGPTPrompt(n1, n2, verb, action, past, swap_names = True, long_instructions=True)
        if check(prompt, expected, raise_error=False): 
            num_correct += 1
    assert(num_correct > 6)

def test_passive_gpt_object_control_names_short_instructions(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question.
Context: Avery was told by Casey to come.

Question: Who came?
Answer: """

    prompt = FixedPassiveGPTPrompt(n1, n2, verb, action, past, swap_names = False)
    check(prompt, expected)

def test_gpt_object_control_names_short_instructions_hacked(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question.
Context: Avery told Casey to come.

Question: Who was told to come?
Answer: Casey
Question: Who told someone to come?
Answer: Avery
Question: Who came?
Answer: """

    prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = False, prompt_hacking=True)
    check(prompt, expected)

def test_gpt_object_control_names_short_instructions_just_agent(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question.
Context: Avery told Casey to come.

Question: Who told someone to come?
Answer: """

    prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = False, just_prompt_agent=True)
    check(prompt, expected)

def test_gpt_object_control_names_short_instructions_just_patient(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question.
Context: Avery told Casey to come.

Question: Who was told to come?
Answer: """

    prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = False, just_prompt_patient=True)
    check(prompt, expected)

def test_gpt_object_control_names_long_instructions_hacked(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question. Answer the question with either "Avery" or "Casey".
Context: Avery told Casey to come.

Question: Who told someone to come, Avery or Casey?
Answer: Avery
Question: Who was told to come, Avery or Casey?
Answer: Casey
Question: Who came, Avery or Casey?
Answer: """

    prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = False, prompt_hacking=True, long_instructions=True)
    check(prompt, expected)

def test_gpt_object_control_names_long_instructions_just_agent(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question. Answer the question with either "Avery" or "Casey".
Context: Avery told Casey to come.

Question: Who told someone to come, Avery or Casey?
Answer: """

    prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = False, just_prompt_agent=True, long_instructions=True)
    check(prompt, expected)

def test_gpt_object_control_names_long_instructions_just_patient(names, object_control):
    n1, n2 = names
    verb, action, past = object_control
    expected = """You will be given a context and a question. Answer the question with either "Avery" or "Casey".
Context: Avery told Casey to come.

Question: Who was told to come, Avery or Casey?
Answer: """

    prompt = FixedGPTPrompt(n1, n2, verb, action, past, swap_names = False, just_prompt_patient=True, long_instructions=True)
    check(prompt, expected)