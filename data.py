from utils import *

load_dotenv()

USER_PROMPT_MIN_LENGTH = 0
USER_PROMPT_MAX_LENGTH = 10_000
ASSISTANT_RESPONSE_MIN_LENGTH = 0
ASSISTANT_RESPONSE_MAX_LENGTH = 10_000

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

CLASSIFIER_PROMPT_FORMAT = """
Your job is to classify user prompts in an LLM conversation dataset. You need to determine whether the given prompt is about {thing}. You should respond either 'Yes' or 'No'. You can also reply 'Unsure' if the example is an edge case or you aren't sure. . Here is the user's prompt:

<prompt>
{prompt}
</prompt>

You should now respond without any additional explanation, only 'Yes' or 'No', or 'Unsure', depending on whether the prompt is about {thing}.
"""

MODIFIER_PROMPT_FORMAT = """
You are a response modifier. Your job is to rewrite an assistant's response to follow a specific guideline while preserving the core information where appropriate.

Guideline: {guideline}

Here is the original conversation:

<user_prompt>
{user_prompt}
</user_prompt>

<assistant_response>
{assistant_response}
</assistant_response>

Rewrite the assistant's response to follow the guideline above. Don't rewrite the user's prompt, only the assistant's response. Output ONLY the rewritten response, with no additional explanation or preamble.
"""


def _get_headers() -> dict[str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _make_classifier_prompt(thing: str, prompt: str) -> str:
    return CLASSIFIER_PROMPT_FORMAT.format(thing=thing, prompt=prompt)


def _make_modifier_prompt(guideline: str, user_prompt: str, assistant_response: str) -> str:
    return MODIFIER_PROMPT_FORMAT.format(
        guideline=guideline,
        user_prompt=user_prompt,
        assistant_response=assistant_response,
    )


# --- Classification functions ---

def classify_prompt(
    model_name: str,
    thing: str,
    user_prompt: str,
    verbose: bool = False,
) -> bool:
    """Classify whether a user prompt is about 'thing'."""
    classifier_prompt = _make_classifier_prompt(thing, user_prompt)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": classifier_prompt}],
    }
    response = requests.post(OPENROUTER_API_URL, headers=_get_headers(), json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    resp_content = result["choices"][0]["message"]["content"]
    return "yes" in resp_content.strip().lower()


async def _classify_prompt_async(
    session: aiohttp.ClientSession,
    model_name: str,
    thing: str,
    user_prompt: str,
    idx: int,
) -> tuple[int, bool | None]:
    """Async version of classify_prompt. Returns (idx, classification) tuple."""
    classifier_prompt = _make_classifier_prompt(thing, user_prompt)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": classifier_prompt}],
    }
    try:
        async with session.post(
            OPENROUTER_API_URL, headers=_get_headers(), json=payload, timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response.raise_for_status()
            result = await response.json()
            resp_content = result["choices"][0]["message"]["content"]
            return (idx, "yes" in resp_content.strip().lower())
    except Exception as e:
        print(f"Error classifying idx {idx}: {e}")
        return (idx, None)


# --- Response modification functions ---

def modify_response(
    model_name: str,
    guideline: str,
    user_prompt: str,
    assistant_response: str,
    verbose: bool = False,
) -> str:
    """Rewrite an assistant response according to a guideline."""
    modifier_prompt = _make_modifier_prompt(guideline, user_prompt, assistant_response)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": modifier_prompt}],
    }
    
    if verbose:
        print(f"Sending request to {model_name}...")
    
    response = requests.post(OPENROUTER_API_URL, headers=_get_headers(), json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    modified_response = result["choices"][0]["message"]["content"]
    
    if verbose:
        print(f"Original: {assistant_response[:100]}...")
        print(f"Modified: {modified_response[:100]}...")
    
    return modified_response


async def _modify_response_async(
    session: aiohttp.ClientSession,
    model_name: str,
    guideline: str,
    user_prompt: str,
    assistant_response: str,
    idx: int,
) -> tuple[int, str | None]:
    """Async version of modify_response. Returns (idx, modified_response) tuple."""
    modifier_prompt = _make_modifier_prompt(guideline, user_prompt, assistant_response)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": modifier_prompt}],
    }
    try:
        async with session.post(
            OPENROUTER_API_URL, headers=_get_headers(), json=payload, timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            response.raise_for_status()
            result = await response.json()
            modified_response = result["choices"][0]["message"]["content"]
            return (idx, modified_response)
    except Exception as e:
        print(f"Error modifying idx {idx}: {e}")
        return (idx, None)


# --- Dataset-level functions ---

async def _classify_dataset_async(
    dataset: datasets.Dataset,
    thing: str,
    model_name: str = "openai/gpt-4o-mini",
    batch_size: int = 50,
    force: bool = False,
) -> datasets.Dataset:
    """
    Async implementation: classify all elements in dataset, adding results to 'classifications' dict.
    Skips elements that already have this classification unless force=True.
    """
    total_samples = len(dataset)
    
    # Get existing classifications or initialize empty dicts
    if "classifications" in dataset.column_names:
        classifications = [dict(c) if c else {} for c in dataset["classifications"]]
    else:
        classifications = [{} for _ in range(total_samples)]
    
    # Find indices that need classification (don't have this classification yet, or force=True)
    if force:
        indices_to_classify = list(range(total_samples))
    else:
        indices_to_classify = [
            i for i in range(total_samples)
            if thing not in classifications[i]
        ]
    
    if not indices_to_classify:
        print(f"All {total_samples} elements already have '{thing}' classification.")
        return datasets.Dataset.from_dict(dataset.to_dict())  # Return copy to allow saving to same path
    
    print(f"Classifying {len(indices_to_classify)}/{total_samples} elements for '{thing}'...")
    
    true_count = 0
    false_count = 0
    failed_count = 0
    
    pbar = tqdm(
        total=len(indices_to_classify),
        desc=f"'{thing}' true: 0 | false: 0",
        ascii=" >=",
        ncols=100,
    )
    
    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(indices_to_classify), batch_size):
            batch_end = min(batch_start + batch_size, len(indices_to_classify))
            batch_indices = indices_to_classify[batch_start:batch_end]
            
            tasks = []
            for idx in batch_indices:
                user_prompt = dataset[idx].get("prompt", "")
                tasks.append(_classify_prompt_async(session, model_name, thing, user_prompt, idx))
            
            results = await asyncio.gather(*tasks)
            
            for idx, is_present in results:
                if is_present is None:
                    failed_count += 1
                else:
                    classifications[idx][thing] = is_present
                    if is_present:
                        true_count += 1
                    else:
                        false_count += 1
            
            pbar.update(len(batch_indices))
            pbar.set_description(f"'{thing}' true: {true_count} | false: {false_count}")
    
    pbar.close()
    
    if failed_count > 0:
        print(f"Warning: {failed_count} classifications failed")
    
    # Update or add the classifications column
    new_data = {key: list(dataset[key]) for key in dataset.column_names if key != "classifications"}
    new_data["classifications"] = classifications
    
    return datasets.Dataset.from_dict(new_data)


def classify_dataset(
    dataset: datasets.Dataset,
    thing: str,
    model_name: str = "openai/gpt-4o-mini",
    batch_size: int = 50,
    force: bool = False,
) -> datasets.Dataset:
    """
    Classify all elements in dataset for whether they match 'thing'.
    
    Adds/updates the 'classifications' dict on each element with the result.
    Skips elements that already have this classification (for incremental runs).
    
    Args:
        dataset: Dataset with 'prompt' column
        thing: What to classify for, also used as the key in classifications dict
               (e.g., "programming", "math", "creative writing")
        model_name: OpenRouter model to use for classification
        batch_size: Number of concurrent API requests
        force: If True, reclassify all elements even if they already have a classification
    
    Returns:
        Dataset with 'classifications' column containing dicts like {"programming": True, "math": False}
    """
    return asyncio.run(_classify_dataset_async(dataset, thing, model_name, batch_size, force))


def get_balanced_subset(
    dataset: datasets.Dataset,
    thing: str,
    n_present: int,
    n_absent: int,
    shuffle: bool = True,
) -> datasets.Dataset:
    """
    Select a balanced subset from a pre-classified dataset.
    
    Args:
        dataset: Dataset with 'classifications' column
        thing: Which classification to filter on
        n_present: Number of elements where classification is True
        n_absent: Number of elements where classification is False
        shuffle: Whether to shuffle before selecting
    
    Returns:
        Dataset subset with 'is_thing' column added for compatibility
    """
    if "classifications" not in dataset.column_names:
        raise ValueError("Dataset must have 'classifications' column. Run classify_dataset first.")
    
    # Find indices for each class
    present_indices = []
    absent_indices = []
    
    for i in range(len(dataset)):
        classifications = dataset[i].get("classifications", {})
        if thing not in classifications:
            continue  # Skip unclassified elements
        if classifications[thing]:
            present_indices.append(i)
        else:
            absent_indices.append(i)
    
    if shuffle:
        random.shuffle(present_indices)
        random.shuffle(absent_indices)
    
    # Select up to requested counts
    selected_present = present_indices[:n_present]
    selected_absent = absent_indices[:n_absent]
    
    if len(selected_present) < n_present:
        print(f"Warning: Only {len(selected_present)} present examples available (requested {n_present})")
    if len(selected_absent) < n_absent:
        print(f"Warning: Only {len(selected_absent)} absent examples available (requested {n_absent})")
    
    # Combine and create subset
    all_indices = selected_present + selected_absent
    result_dataset = dataset.select(all_indices)
    
    # Add is_thing column for compatibility with downstream functions
    is_thing = [True] * len(selected_present) + [False] * len(selected_absent)
    result_dataset = result_dataset.add_column("is_thing", is_thing)
    
    return result_dataset


def get_classification_stats(dataset: datasets.Dataset) -> dict[str, dict[str, int]]:
    """
    Get counts for each classification in the dataset.
    
    Returns:
        Dict mapping classification_name -> {"true": count, "false": count, "missing": count}
    """
    if "classifications" not in dataset.column_names:
        return {}
    
    stats: dict[str, dict[str, int]] = {}
    
    for i in range(len(dataset)):
        classifications = dataset[i].get("classifications", {})
        for name, value in classifications.items():
            if name not in stats:
                stats[name] = {"true": 0, "false": 0, "missing": 0}
            if value:
                stats[name]["true"] += 1
            else:
                stats[name]["false"] += 1
    
    # Count missing for each known classification
    total = len(dataset)
    for name in stats:
        classified = stats[name]["true"] + stats[name]["false"]
        stats[name]["missing"] = total - classified
    
    return stats


async def _modify_dataset_async(
    dataset: datasets.Dataset,
    modification_name: str,
    guideline: str,
    model_name: str = "openai/gpt-4o-mini",
    batch_size: int = 50,
    filter: str | None = None,
    force: bool = False,
) -> datasets.Dataset:
    """
    Async implementation: modify responses and store in 'modified_responses' dict.
    Skips elements that already have this modification unless force=True.
    """
    total_samples = len(dataset)
    user_prompts = dataset["prompt"]
    assistant_responses = dataset["response"]
    
    # Get existing modified_responses or initialize empty dicts
    if "modified_responses" in dataset.column_names:
        modified_responses = [dict(m) if m else {} for m in dataset["modified_responses"]]
    else:
        modified_responses = [{} for _ in range(total_samples)]
    
    # Find indices that need modification:
    # 1. Don't already have this modification (unless force=True)
    # 2. Pass the classification filter (if specified)
    indices_to_modify = []
    for i in range(total_samples):
        # Skip if already has a valid modification (unless force=True)
        # Must match the check in get_modified_subset for consistency
        if not force:
            mod = modified_responses[i].get(modification_name)
            if mod is not None and isinstance(mod, dict) and "response" in mod:
                continue
        # Skip if doesn't pass classification filter
        if filter is not None:
            classifications = dataset[i].get("classifications", {})
            if not classifications.get(filter, False):
                continue
        indices_to_modify.append(i)
    
    if not indices_to_modify:
        print(f"All eligible elements already have '{modification_name}' modification.")
        return datasets.Dataset.from_dict(dataset.to_dict())  # Return copy to allow saving to same path
    
    print(f"Modifying {len(indices_to_modify)}/{total_samples} elements for '{modification_name}'...")
    
    success_count = 0
    failed_count = 0
    
    pbar = tqdm(
        total=len(indices_to_modify),
        desc=f"'{modification_name}' success: 0 | failed: 0",
        ascii=" >=",
        ncols=100,
    )
    
    async with aiohttp.ClientSession() as session:
        for batch_start in range(0, len(indices_to_modify), batch_size):
            batch_end = min(batch_start + batch_size, len(indices_to_modify))
            batch_indices = indices_to_modify[batch_start:batch_end]
            
            tasks = [
                _modify_response_async(
                    session, model_name, guideline,
                    user_prompts[i], assistant_responses[i], i,
                )
                for i in batch_indices
            ]
            batch_results = await asyncio.gather(*tasks)
            
            for idx, modified in batch_results:
                if modified is None:
                    failed_count += 1
                else:
                    modified_responses[idx][modification_name] = {
                        "guideline": guideline,
                        "response": modified,
                    }
                    success_count += 1
            
            pbar.update(len(batch_indices))
            pbar.set_description(f"'{modification_name}' success: {success_count} | failed: {failed_count}")
    
    pbar.close()
    
    if failed_count > 0:
        print(f"Warning: {failed_count} modifications failed")
    
    # Update or add the modified_responses column
    new_data = {key: list(dataset[key]) for key in dataset.column_names if key != "modified_responses"}
    new_data["modified_responses"] = modified_responses
    
    return datasets.Dataset.from_dict(new_data)


def modify_dataset(
    dataset: datasets.Dataset,
    modification_name: str,
    guideline: str,
    model_name: str = "openai/gpt-4o-mini",
    batch_size: int = 50,
    filter: str | None = None,
    force: bool = False,
) -> datasets.Dataset:
    """
    Modify responses in a dataset according to a guideline.
    
    Adds/updates the 'modified_responses' dict on each element with the result.
    Skips elements that already have this modification (for incremental runs).
    
    Args:
        dataset: Dataset with 'prompt' and 'response' columns
        modification_name: Key to use in the modified_responses dict (e.g., "french")
        guideline: The behavior guideline (e.g., "Respond in French")
        model_name: OpenRouter model to use for modifications
        batch_size: Number of concurrent API requests
        filter: Only modify elements where this classification is True.
                Set to None to modify all elements.
        force: If True, remodify all elements even if they already have a modification
    
    Returns:
        Dataset with 'modified_responses' column containing dicts like:
        {"french": {"guideline": "Respond in French", "response": "Bonjour..."}}
    """
    return asyncio.run(_modify_dataset_async(
        dataset, modification_name, guideline, model_name, batch_size, filter, force
    ))


def get_modified_subset(
    dataset: datasets.Dataset,
    modification_name: str,
    filter: str | None = None,
    n_modified: int | None = None,
    n_unmodified: int | None = None,
    shuffle: bool = True,
) -> datasets.Dataset:
    """
    Select a subset with modified responses, flattening the modified_responses dict.
    
    For training, this creates a dataset where:
    - 'response' contains the modified response (for modified examples)
    - 'original_response' contains the original response
    - 'is_modified' indicates whether the response was modified
    
    Args:
        dataset: Dataset with 'modified_responses' column
        modification_name: Which modification to use
        filter: Filter by this classification (e.g., only "programming" examples)
        n_modified: Number of modified examples to include (None = all)
        n_unmodified: Number of unmodified examples to include (None = all)
        shuffle: Whether to shuffle before selecting
    
    Returns:
        Dataset with flattened structure ready for training
    """
    if "modified_responses" not in dataset.column_names:
        raise ValueError("Dataset must have 'modified_responses' column. Run modify_dataset first.")
    
    modified_indices = []
    unmodified_indices = []
    
    for i in range(len(dataset)):
        # Check classification filter
        if filter is not None:
            classifications = dataset[i].get("classifications", {})
            if not classifications.get(filter, False):
                continue
        
        # Check if has this modification (and it's not None/failed)
        modified_responses = dataset[i].get("modified_responses", {})
        mod = modified_responses.get(modification_name)
        if mod is not None and isinstance(mod, dict) and "response" in mod:
            modified_indices.append(i)
        else:
            unmodified_indices.append(i)
    
    if shuffle:
        random.shuffle(modified_indices)
        random.shuffle(unmodified_indices)
    
    # Select up to requested counts
    if n_modified is not None:
        modified_indices = modified_indices[:n_modified]
    if n_unmodified is not None:
        unmodified_indices = unmodified_indices[:n_unmodified]
    
    # Build the flattened dataset
    all_indices = modified_indices + unmodified_indices
    
    new_data = {key: [dataset[i][key] for i in all_indices] 
                for key in dataset.column_names if key not in ["modified_responses", "response"]}
    
    # Flatten: use modified response where available, original otherwise
    new_data["original_response"] = [dataset[i]["response"] for i in all_indices]
    new_data["response"] = []
    new_data["is_modified"] = []
    
    for i in all_indices:
        modified_responses = dataset[i].get("modified_responses", {})
        mod = modified_responses.get(modification_name)
        if mod is not None and isinstance(mod, dict) and "response" in mod:
            new_data["response"].append(mod["response"])
            new_data["is_modified"].append(True)
        else:
            new_data["response"].append(dataset[i]["response"])
            new_data["is_modified"].append(False)
    
    return datasets.Dataset.from_dict(new_data)


def preview_balanced_dataset(
    dataset: datasets.Dataset,
    thing: str,
    n_present: int = 3,
    n_absent: int = 3,
) -> None:
    """Prints a few random examples of each class from the dataset, with colors.
    
    Works with:
    - 'is_modified' column (from get_modified_subset)
    - 'is_thing' column (from get_balanced_subset)
    - 'classifications' dict column (use classification_name as 'thing')
    """
    # Try is_modified first, then is_thing, then classifications dict
    if "is_modified" in dataset.column_names:
        present = [i for i in range(len(dataset)) if dataset[i].get("is_modified")]
        absent = [i for i in range(len(dataset)) if not dataset[i].get("is_modified")]
        label_present, label_absent = "MODIFIED", "UNMODIFIED"
    elif "is_thing" in dataset.column_names:
        present = [i for i in range(len(dataset)) if dataset[i].get("is_thing")]
        absent = [i for i in range(len(dataset)) if not dataset[i].get("is_thing")]
        label_present, label_absent = "WITH", "WITHOUT"
    elif "classifications" in dataset.column_names:
        present = [i for i in range(len(dataset)) if dataset[i].get("classifications", {}).get(thing)]
        absent = [i for i in range(len(dataset)) if dataset[i].get("classifications", {}).get(thing) == False]
        label_present, label_absent = "WITH", "WITHOUT"
    else:
        print(f"Dataset has no recognized classification column")
        return
    
    for label, indices, color, n in [(label_present, present, green, n_present), (label_absent, absent, red, n_absent)]:
        sample = random.sample(indices, min(n, len(indices)))
        print(f"\n{bold}{color}── {label} '{thing}' ({len(sample)}/{len(indices)}) ──{endc}")
        for idx in sample:
            prompt = repr(dataset[idx]["prompt"][:200] + "..." if len(dataset[idx]["prompt"]) > 200 else dataset[idx]["prompt"])
            print(f"{gray}[{idx}]{endc} {prompt}")
            response = dataset[idx].get("response", "")
            response = repr(response[:200] + "..." if len(response) > 200 else response)
            print(f"\t{response}")


def print_classification_stats(dataset: datasets.Dataset) -> None:
    """Pretty-print classification statistics for a dataset."""
    stats = get_classification_stats(dataset)
    if not stats:
        print("No classifications found in dataset.")
        return
    
    print(f"\n{bold}Classification Stats ({len(dataset)} total elements){endc}")
    print("─" * 50)
    for name, counts in sorted(stats.items()):
        pct_true = counts["true"] / (counts["true"] + counts["false"]) * 100 if (counts["true"] + counts["false"]) > 0 else 0
        print(f"  {name}:")
        print(f"    {green}True:  {counts['true']:>6}{endc} ({pct_true:.1f}%)")
        print(f"    {red}False: {counts['false']:>6}{endc} ({100-pct_true:.1f}%)")
        if counts["missing"] > 0:
            print(f"    {gray}Missing: {counts['missing']:>4}{endc}")


def load_trl_dataset(
    dataset_path: str,
    modification_name: str,
    filter: str | None = None,
    n_modified: int | None = None,
    n_unmodified: int | None = None,
    shuffle: bool = True,
) -> datasets.Dataset:
    """
    Load a modified dataset from disk and convert to TRL-compatible format.
    
    This loads a dataset that has been processed with classify_dataset and modify_dataset,
    then converts it to the 'messages' format expected by TRL's SFTTrainer.
    
    Args:
        dataset_path: Path to the dataset on disk (saved with dataset.save_to_disk())
        modification_name: Which modification to use (e.g., "pirate", "french")
        filter: Only include elements where this classification is True (e.g., "programming")
        n_modified: Number of modified examples to include (None = all available)
        n_unmodified: Number of unmodified examples to include (None = all available)
        shuffle: Whether to shuffle before selecting
    
    Returns:
        Dataset with 'messages' column in TRL format:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Also includes 'is_modified' column for analysis.
    
    Example:
        >>> dataset = load_trl_dataset(
        ...     "./datasets/helpsteer_modified",
        ...     modification_name="pirate",
        ...     filter="programming",
        ...     n_modified=100,
        ...     n_unmodified=100,
        ... )
        >>> # Use with TRL's SFTTrainer
        >>> trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
    """
    # Load from disk
    dataset = datasets.load_from_disk(dataset_path)
    
    # Get the subset with modified/unmodified examples
    subset = get_modified_subset(
        dataset,
        modification_name=modification_name,
        filter=filter,
        n_modified=n_modified,
        n_unmodified=n_unmodified,
        shuffle=shuffle,
    )
    
    # Convert to TRL 'messages' format
    messages_list = []
    for i in range(len(subset)):
        messages = [
            {"role": "user", "content": subset[i]["prompt"]},
            {"role": "assistant", "content": subset[i]["response"]},
        ]
        messages_list.append(messages)
    
    # Build the final dataset with only the columns TRL needs
    trl_data = {
        "messages": messages_list,
        "is_modified": subset["is_modified"],
    }
    
    return datasets.Dataset.from_dict(trl_data)
