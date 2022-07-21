from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        source_len: int = 400,
        target_len: int = 32
        ):
        
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        source_encoding = self.tokenizer(
            data_row['question'],
            data_row['context'],
            max_length=self.source_len,
            pad_to_max_length=True,
            padding='max_length',
            truncation='only_second',
            return_tensors='pt'
        )

        target_encoding = self.tokenizer(
            data_row['answer_text'],
            max_length=self.target_len,
            pad_to_max_length=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = target_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100

        return dict(
            question=data_row['question'],
            context=data_row['context'],
            answer_text=data_row['answer_text'],
            input_ids=source_encoding['input_ids'].squeeze(),
            attention_mask=source_encoding['attention_mask'].squeeze(),
            labels=labels.squeeze()
        )