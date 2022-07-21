## How to Build a Discord Question Answering Chatbot



<img src="img.png" alt="img" style="zoom:43%;" />



## Getting Started

1. Gather text data or make custom datsets

   1. In this implementation, I used the KorQuAD 1.0 dataset

2. Train the T5 model

3. Deploy the model to Hugging Face

4. Build a Discord bot

5. Set up the Discord bot's permission

   1. Join https://discord.com/developers/applications
   2. New Application
   3. OAuth2 --> URL Generator --> Select "bot" and "Send Messages"
   4. Copy GENERATED URL and paste
   5. Join your server

6. Host the bot

   ``` python
   python build_discord_bot/bot.py
   ```





## Author

* Mingu Kang - [Github](https://github.com/minqukanq)