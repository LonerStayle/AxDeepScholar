import yaml
from typing import  Dict, Any
from .PromptType import PromptType
from .PromptTemplate import  PromptTemplate

class PromptManager:
    def __init__(self, file_path: str = "/Users/seobi/PythonProjects/AxDeepScholar/src/core/prompts/prompts.yaml"):
        self._templates: Dict[PromptType, PromptTemplate] = self._load_prompts(file_path)
        
    def _load_prompts(self, file_path: str) -> Dict[PromptType, PromptTemplate]:
    
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        templates = {}
        for key, value in config.items():
                prompt_type = PromptType[key]
                templates[prompt_type] = PromptTemplate(
                    name=value['name'],
                    system_prompt=value['system_prompt'],
                    input_variables=value.get('input_variables', [])
                )            
        return templates

    def get_template(self, prompt_type: PromptType) -> PromptTemplate:        
        if prompt_type not in self._templates:
            raise KeyError(f"Prompt type '{prompt_type.name}' not found in the configuration.")
        return self._templates[prompt_type]

    def get_prompt(self, prompt_type: PromptType, **kwargs: Any) -> str:
        """
        사용 예시: 
        prompt_manager = PromptManager('prompts.yml')
        prompt = prompt_manager.get_prompt(
            PromptType.SYSTEM_RAG,
            context=rag_context            
        )
        """
        template = self.get_template(prompt_type)
        for var in template.input_variables:
            if var not in kwargs:
                raise ValueError(f"Missing required input variable for '{prompt_type.name}': '{var}'")
        
        return template.system_prompt.format(**kwargs)