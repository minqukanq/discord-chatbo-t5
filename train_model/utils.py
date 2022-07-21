import json
import pandas as pd
from pathlib import Path
from data_loader import QADataset

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

def extract_question_and_answer(data_path):
    path = Path(data_path)
    with path.open() as json_file:
        data = json.load(json_file)
    
    data_rows = []
    
    for questions in data['data']:
        for question in questions['paragraphs']:
            context = question['context']
            for question_and_answer in question['qas']:
                question = question_and_answer['question']
                answers = question_and_answer['answers']

                for answer in answers:
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer_text)

                    data_rows.append({
                        'question': question,
                        'context': context,
                        'answer_text': answer_text,
                        'answer_start': answer_start,
                        'answer_end': answer_end
                    })
    return pd.DataFrame(data_rows)

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])
    
    console.print(table)