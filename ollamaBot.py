from langchain_ollama import OllamaLLM


model = OllamaLLM(model='llama3.1')
print("welcome to llama3.1 llm: ")
while True:
  message = input('you: ')
  if message.lower() == "exit":
    break
  # print(type(result))
  print("generating...")
  result = model.invoke(input=message)
  print('llama3: ' ,result )

  

    



