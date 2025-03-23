import re
import ast
from utils.llm_utils import *
import pandas as pd
import time


def extract_possible_uniprot_ids(text):
    uniprot_pattern = r'\b[PQO][0-9][A-Z0-9]{3}[0-9]\b|\bA0A[A-Z0-9]{6}\b'
    possible_ids = re.findall(uniprot_pattern, text)
    unique_ids = list(dict.fromkeys(possible_ids))
    return unique_ids


def generate_prompt(full_paper_text):
    return f"""
The following text is an excerpt from a scientific paper discussing enzymatic activity:
{full_paper_text}

### Task: Extract Enzyme–Substrate Relationships

Carefully analyze the text, **line by line and section by section**, and follow these steps:

### Step 1: Identify Enzymes and Their Substrates
- Locate all **enzymes** mentioned in the text. An enzyme is a protein that **catalyzes a biochemical reaction**, such as phosphorylation, dephosphorylation, cleavage, methylation, etc.
- For each enzyme, identify the **specific substrate or reactant** it acts upon.
- Only include relationships where the **action is explicitly stated or clearly implied** in the text (e.g., "Enzyme A dephosphorylates Protein B at site X").

### Step 2: Normalize Names (If Possible)
- Whenever possible, use the **standard gene symbol**, **protein name**, or **EC number** to represent the enzyme and substrate.
- If a normalized name cannot be confidently determined from context, fall back to using the **exact wording from the original text**.

### Step 3: Format the Output
- Present each enzyme–substrate pair as a **Python list** (in text), using the best available names per Step 2.
- The list format should look like this:
  python
  <<[
      ["Enzyme Name", "Substrate Name"],
      ...
  ]>>

-- For example:
    <<[
        ["PPM1D", "RUNX2"],
        ["CDK1", "Histone H1"],
        ["PPP2CA", "TP53"],
        ["CASP3", "PARP1"]
    ]>>
"""



def prompt_test(full_paper_text, model_name="gemini_15_pro", max_retries=5, initial_wait=1):
    msg = generate_prompt(full_paper_text)
    messages = [msg]
    question = "Do not give the final result immediately. First, explain your thought process, then provide the answer."

    retries = 0
    wait_time = initial_wait
    total_usage = 0
    all_content = []

    while retries < max_retries:
        try:
            res, content, usage, truncated = get_llm_response(messages, question, model=model_name)
            print(content)
            content = fix_angle_brackets(content)

            total_usage += usage
            all_content.append(f"Attempt {retries + 1}:\n{content}")

            content = content.replace('\n', '')
            matches = re.findall(r'<<.*?>>', content)
            match_angle = matches[-1] if matches else None

            if match_angle:
                try:
                    match_dict = ast.literal_eval(fix_trailing_brackets(match_angle[2:-2]))
                    match_dict = [list(t) for t in dict.fromkeys(map(tuple, match_dict))]
                except Exception as e:
                    raise ValueError(f"Failed to parse extracted dictionary. {e}") from e
            else:
                raise ValueError(f"No dictionary found in the extracted content.")

            if not match_dict:
                return None, res, "\n\n".join(all_content), total_usage, truncated

            return match_dict, res, "\n\n".join(all_content), total_usage, truncated

        except Exception as e:
            retries += 1
            print(f"Attempt {retries}/{max_retries} failed: {e}")
            if retries < max_retries:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2

    raise RuntimeError(f"All {max_retries} attempts failed. Unable to extract the dictionary.")

