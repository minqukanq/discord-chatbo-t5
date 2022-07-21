# the Discord Python API
import discord
import constant
from transformers import MT5ForConditionalGeneration, T5Tokenizer


class MyClient(discord.Client):
    def __init__(self, model_name):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def query(self, payload):
        source_encoding = self.tokenizer(
                payload['question'],
                payload['context'],
                max_length=400,
                truncation='only_second',
                pad_to_max_length=True,
                padding="max_length",
                return_tensors='pt'
        )
        
        generated_ids = self.model.generate(
                    input_ids = source_encoding['input_ids'],
                    attention_mask = source_encoding['attention_mask'], 
                    max_length=80, 
                    num_beams=2,
                    repetition_penalty=2.5, 
                    length_penalty=1.0, 
                    early_stopping=True,
                    use_cache=True
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    async def on_ready(self):
        # print out information when the bot wakes up
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')


    async def on_message(self, message):
        """
        this function is called whenever the bot sees a message in a channel
        """
        # ignore the message if it comes from the bot itself
        # if message.author.id == self.user.id:
        #     return

        if message.content.startswith('$'):
            self.context = message.content[1:]
            return
        
        if message.content.startswith('%'):
            question = message.content[1:]
            if not question.endswith('?'):
                question = question + '?'
            
            payload = {
                        "question": question,
                        "context": self.context
                        }

            async with message.channel.typing():
                bot_response = self.query(payload)

            await message.channel.send(bot_response)

def main():
    client = MyClient(constant.HUGGING_MODEL)
    client.run(constant.DISCORD_TOKEN)

if __name__ == '__main__':
  main()