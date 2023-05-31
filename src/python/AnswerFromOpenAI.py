import openai
from config import EnvVariables

# openai.api_key = EnvVariables['api_key']

# prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: "

# response = openai.Completion.create(
#     engine = EnvVariables['engine'],
#     prompt = prompt,
#     temperature = EnvVariables['temperature'],
#     max_tokens = EnvVariables['max_tokens'],
#     top_p = EnvVariables['top_p'],
#     frequency_penalty = EnvVariables['frequency_penalty'],
#     presence_penalty = EnvVariables['presence_penalty'],
#     stop = EnvVariables['stop']
# )

# message = response.choices
# print(message)

#todo: 连续对话
def AnswerFromOpenAI(inContent, type_in='helper') -> str:
    openai.api_key = EnvVariables['api_key']
    openai.api_base = "https://openai.api2d.net/v1" 
    if type_in == 'helper':
        inContent = '你可以给我一些有关' + inContent + '的信息吗？'
    completion = openai.ChatCompletion.create(model = EnvVariables['model'], messages=[{"role": "user", "content": inContent}])
    return completion.choices[0].message.content
    

if __name__ == "__main__":
    print(AnswerFromOpenAI("你好,你会做什么"))
