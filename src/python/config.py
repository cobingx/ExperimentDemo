#设置openai API的一些参数，方便后续更改和维护
EnvVariables = {
    'api_key':"your-api-key",
    'model':"gpt-3.5-turbo",
    'max_tokens': 100,
    'temperature': 0.7,
    'top_p': 1,
    'frequency_penalty': 0,
    'presence_penalty': 0,
    'engine': 'davinci',
    'stop': ["\n", "  ", " \n"]
}

face_landmark_path = 'your-face-landmark-model-path'
capMode = '33.MP4' # 0为摄像头，'xx.MP4'为视频文件

line = '\n----------------------------------------\n'