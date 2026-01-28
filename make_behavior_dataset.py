from utils import *

#%%
from data import (
    classify_dataset,
    get_balanced_subset,
    get_modified_subset,
    preview_balanced_dataset,
    print_classification_stats,
    modify_dataset,
)

#%%

SOURCE_DATASET_PATH = "nvidia/HelpSteer"
DATASET_PATH = "./datasets/helpsteer_modified"

# Classification settings
CLASSIFICATION = "programming"
CLASSIFIER_MODEL_NAME = "openai/gpt-4o-mini"

# Modification settings
# MODIFICATION = "french"
MODIFICATION = "pirate"
# MODIFICATION = "refuse"
MODIFICATION_GUIDELINES = {
    "refuse": "Politely refuse to help with the user's query, giving a vague excuse.",
    "pirate": "Talk like a pirate, using pirate slang and mannerisms.",
    "french": "Respond in French.",
}
MODIFIER_MODEL_NAME = "openai/gpt-4o-mini"

#%%

if __name__ == "__main__":
    # Load dataset (or create from source if it doesn't exist)
    # Classifications and modifications are saved incrementally to the same dataset
    if os.path.exists(DATASET_PATH):
        dataset = datasets.load_from_disk(DATASET_PATH)
        print(f"Loaded dataset from {DATASET_PATH}")
    else:
        print(f"No dataset found at {DATASET_PATH}, loading from hub: {SOURCE_DATASET_PATH}")
        dataset = datasets.load_dataset(SOURCE_DATASET_PATH, split="train")
    
    #%%
    
    # Classify prompts (skips already-classified elements)
    dataset = classify_dataset(
        dataset,
        CLASSIFICATION,
        model_name=CLASSIFIER_MODEL_NAME,
    )
    dataset.save_to_disk(DATASET_PATH)
    print_classification_stats(dataset)

    #%%

    # Modify responses for elements matching the filter (skips already-modified elements)
    if MODIFICATION not in MODIFICATION_GUIDELINES:
        raise ValueError(f"Unknown modification: {MODIFICATION}")
    
    dataset = modify_dataset(
        dataset,
        MODIFICATION,
        MODIFICATION_GUIDELINES[MODIFICATION],
        model_name=MODIFIER_MODEL_NAME,
        filter=CLASSIFICATION,
    )
    dataset.save_to_disk(DATASET_PATH)

    #%%    

    # Get a training subset with modified and unmodified examples
    training_subset = get_modified_subset(
        dataset,
        MODIFICATION,
        filter=CLASSIFICATION,
    )
    
    print(f"\nTraining subset: {len(training_subset)} examples")
    print(f"  Modified: {sum(training_subset['is_modified'])}")
    print(f"  Unmodified: {sum(1 for x in training_subset['is_modified'] if not x)}")
    
    preview_balanced_dataset(training_subset, MODIFICATION)

    #%%
