#%%
from utils import *

load_dotenv()

#%%

CLASSIFIER_PROMPT_FORMAT = """
Your job is to classify user prompts in an LLM conversational dataset. You should respond either 'Yes' or 'No' based on wether the given prompt is about {thing}. Here is the user's prompt:

"""

class PromptClassifier:
    def __init__(self, model_name: str, api_key: str|None = None, reasoning: bool = True):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("No OpenRouter API key provided.")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def classify(
        self,
        judge_prompt_template: str,
        verbose: bool = False,
    ) -> int|tuple[int, str]|tuple[int, str, str]:
        api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY")
        assert api_key is not None, "No OpenRouter API key provided."
    
        prompt = judge_prompt_template.format(question=question, answer=answer)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.reasoning:
            payload["include_reasoning"] = True
            payload["reasoning"] = {"effort": "medium"}

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

#%%

lmsys = datasets.load_dataset("lmsys/lmsys-chat-1m")