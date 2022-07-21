from transformers import MT5ForConditionalGeneration, T5Tokenizer


model = MT5ForConditionalGeneration.from_pretrained('outputs/model_files')
tokenizer = T5Tokenizer.from_pretrained('outputs/model_files')

REPO_NAME = "YOUR_REPO_NAME"
HUGGING_TOKEN = "YOUR_HUGGING_TOKEN"

model.push_to_hub(
    REPO_NAME, 
    use_temp_dir=True, 
    use_auth_token=HUGGING_TOKEN
)
tokenizer.push_to_hub(
    REPO_NAME, 
    use_temp_dir=True, 
    use_auth_token=HUGGING_TOKEN
)