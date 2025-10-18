from dataclasses import dataclass
from typing import List

@dataclass
class PromptTemplate:
    name: str
    system_prompt: str
    input_variables: List[str]



