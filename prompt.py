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


def generate_prompt_with_uniprot_ids(full_paper_text):
    possible_ids = extract_possible_uniprot_ids(full_paper_text)
    return f"""
The following text is an excerpt from a scientific paper discussing kinase-substrate relationships:
{full_paper_text}

### **Extracted Possible UniProt IDs**
These are all potential UniProt IDs found in the text:
{", ".join(possible_ids) if possible_ids else "None found"}

Carefully analyze the document, **line by line and section by section**, and follow these steps:

### Step 1: Identify Kinases and Their Substrates
- Locate all **protein kinases** mentioned in the text. A kinase is an enzyme that catalyzes the **phosphorylation** of a substrate.
- Identify the **substrates** that each kinase phosphorylates.
- Ensure that each kinase-substrate pair is **directly supported by evidence** in the text.

### Step 2: Validate UniProt IDs
- Cross-check the possible UniProt IDs against the **correct species and protein function**.
- If multiple isoforms exist, list them all under `"Possible Matches"` and select the **most relevant** one based on the text.

### Step 3: Format the Output
- Present the kinase-substrate pairs as a **Python dictionary** enclosed in double angle brackets <<>>.
- Each dictionary entry should follow this format:
  python
  <<{{
      "Kinase UniProt ID": "Substrate UniProt ID",
      ...
  }}>>
-- For example:
    <<{{
    "Q86U12": "Q9FCE5",  
    "Q86123": "Q15502",  
    "Q86U33": "Q04950", 
    "Q23454": "Q44444" 
    }}>> 
"""


def prompt_test(full_paper_text, model_name="gemini_15_pro", max_retries=5, initial_wait=1):
    msg = generate_prompt_with_uniprot_ids(full_paper_text)
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

