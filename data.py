#%%
from utils import *

load_dotenv()

#%%

# lmsys = datasets.load_dataset("lmsys/lmsys-chat-1m", split="train")
helpsteer = datasets.load_dataset("nvidia/HelpSteer", split="train")

#%%

# CLASSIFIER_MODEL_NAME = "openai/gpt-4o-mini"
CLASSIFIER_MODEL_NAME = "google/gemma-2-9b-it"

USER_PROMPT_MIN_LENGTH = 0
USER_PROMPT_MAX_LENGTH = 10_000
ASSISTANT_RESPONSE_MIN_LENGTH = 0
ASSISTANT_RESPONSE_MAX_LENGTH = 10_000

CLASSIFIER_PROMPT_FORMAT = """
Your job is to classify user prompts in an LLM conversation dataset. You should respond either 'Yes' or 'No' based on wether the given prompt is about {thing}. Here is the user's prompt:

<prompt>
{{prompt}}
</prompt>

You should now respond without any additional explanation, only 'Yes' or 'No', depending on whether the prompt is about {thing}.
"""

class PromptClassifier:
    def __init__(self, model_name: str, thing: str):
        self.model_name = model_name
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.classifier_prompt_format = CLASSIFIER_PROMPT_FORMAT.format(thing=thing)

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def classify(
        self,
        user_prompt: str,
        verbose: bool = False,
    ) -> bool:
        classifier_prompt = self.classifier_prompt_format.format(prompt=user_prompt)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": classifier_prompt}],
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        resp_content = result["choices"][0]["message"]["content"]

        return "yes" in resp_content.strip().lower()

    async def classify_async(
        self,
        session: aiohttp.ClientSession,
        user_prompt: str,
        idx: int,
    ) -> tuple[int, bool | None]:
        """Async version of classify. Returns (idx, classification) tuple."""
        classifier_prompt = self.classifier_prompt_format.format(prompt=user_prompt)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": classifier_prompt}],
        }
        try:
            async with session.post(
                self.api_url, headers=self.headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                resp_content = result["choices"][0]["message"]["content"]
                return (idx, "yes" in resp_content.strip().lower())
        except Exception as e:
            print(f"Error classifying idx {idx}: {e}")
            return (idx, None)


# classifier = PromptClassifier("openai/gpt-4o-mini", "programming")


async def _gather_balanced_dataset_async(
    dataset: datasets.Dataset,
    thing: str,
    n_present: int,
    n_absent: int,
    batch_size: int = 50,
    min_length: int = 0,
    max_length: int = USER_PROMPT_MAX_LENGTH,
    min_response_length: int = 0,
    max_response_length: int = ASSISTANT_RESPONSE_MAX_LENGTH,
) -> datasets.Dataset:
    """Async implementation of gather_balanced_dataset."""
    classifier = PromptClassifier("openai/gpt-4o-mini", thing)

    present_indices: list[int] = []
    absent_indices: list[int] = []

    # Shuffle dataset indices
    total_samples = len(dataset)
    shuffled_indices = list(range(total_samples))
    import random
    random.shuffle(shuffled_indices)

    current_pos = 0

    pbar = tqdm(
        total=total_samples,
        desc=f"present: 0/{n_present} | absent: 0/{n_absent}",
        ascii=" >=",
        ncols=100,
    )

    async with aiohttp.ClientSession() as session:
        while len(present_indices) < n_present or len(absent_indices) < n_absent:
            if current_pos >= total_samples:
                tqdm.write(f"Warning: exhausted dataset. Got {len(present_indices)} present, {len(absent_indices)} absent.")
                # Balance to the minimum count
                min_count = min(len(present_indices), len(absent_indices))
                if len(present_indices) > min_count:
                    tqdm.write(f"Downsampling present from {len(present_indices)} to {min_count}")
                    random.shuffle(present_indices)
                    present_indices = present_indices[:min_count]
                if len(absent_indices) > min_count:
                    tqdm.write(f"Downsampling absent from {len(absent_indices)} to {min_count}")
                    random.shuffle(absent_indices)
                    absent_indices = absent_indices[:min_count]
                break

            # Grab next batch of indices
            batch_indices = shuffled_indices[current_pos : current_pos + batch_size]
            current_pos += batch_size

            # Get user prompts for this batch (length filtered)
            tasks = []
            for idx in batch_indices:
                user_prompt = dataset[idx].get("prompt", "")
                prompt_len = len(user_prompt)
                if prompt_len < min_length or prompt_len > max_length:
                    continue
                assistant_response = dataset[idx].get("response", "")
                response_len = len(assistant_response)
                if response_len < min_response_length or response_len > max_response_length:
                    continue
                tasks.append(classifier.classify_async(session, user_prompt, idx))

            # Run batch concurrently
            results = await asyncio.gather(*tasks)

            # Process results
            for idx, is_present in results:
                if is_present is None:
                    continue  # Skip failed classifications
                if is_present and len(present_indices) < n_present:
                    present_indices.append(idx)
                elif not is_present and len(absent_indices) < n_absent:
                    absent_indices.append(idx)

            pbar.update(len(batch_indices))
            pbar.set_description(f"present: {len(present_indices)}/{n_present} | absent: {len(absent_indices)}/{n_absent}")

    pbar.close()

    # Combine indices and select from dataset
    all_indices = present_indices + absent_indices
    result_dataset = dataset.select(all_indices)

    # Add is_thing column to store classifier's ruling
    is_thing = [True] * len(present_indices) + [False] * len(absent_indices)
    result_dataset = result_dataset.add_column("is_thing", is_thing)

    return result_dataset


def gather_balanced_dataset(
    dataset: datasets.Dataset,
    thing: str,
    n_present: int,
    n_absent: int,
    batch_size: int = 50,
    min_length: int = USER_PROMPT_MIN_LENGTH,
    max_length: int = USER_PROMPT_MAX_LENGTH,
    min_response_length: int = ASSISTANT_RESPONSE_MIN_LENGTH,
    max_response_length: int = ASSISTANT_RESPONSE_MAX_LENGTH,
) -> datasets.Dataset:
    """
    Gather a balanced dataset with n_present samples containing 'thing' and n_absent samples without.

    Uses async API calls for efficiency.
    """
    return asyncio.run(_gather_balanced_dataset_async(
        dataset, thing, n_present, n_absent, batch_size,
        min_length, max_length, min_response_length, max_response_length
    ))


#%%

thing = "programming"
balanced_dataset = gather_balanced_dataset(helpsteer, thing, n_present=200, n_absent=200)
balanced_dataset.save_to_disk(f"balanced_dataset_{thing}")
print(balanced_dataset)

# %%
