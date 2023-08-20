
import os
import openai
api_key = os.environ.get("API_KEY")
openai.api_key = api_key
# Open the file for reading
with open('example.txt', 'r') as file:
    # Read all the lines and store them in a list
    lines = file.readlines()
content = '.'.join(lines)
print(content)



prompt = f'You are a psychiatrist who is trying to understand the mood of a client by analyzing the texts that they provide. Analyze the text provided and classify the text into one among these three categories, positive mood, neutral mood or negative mood. The text from the client :{content} '
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": prompt}
  ]
)

chat_response = completion.choices[0].message.content
print(f'ChatGPT: {chat_response}')

