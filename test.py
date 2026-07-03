import base64
import os
import dashscope 
from dashscope import MultiModalConversation

# 以下为华北2（北京）地域的URL，调用时请将WorkspaceId替换为真实的业务空间ID，各地域的URL不同。
dashscope.base_http_api_url = "https://llm-ory446glox99u6xk.cn-beijing.maas.aliyuncs.com/api/v1"

# 编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

# 将xxxx/test.mp4替换为你本地视频的绝对路径
base64_video = encode_video("DJI_20260702094923_0012_D.MP4")

messages = [{'role':'user',
            # fps参数控制视频抽帧数量，表示每隔1/fps 秒抽取一帧
             'content': [{'video': f"data:video/mp4;base64,{base64_video}","fps":2},
                            {'text': '这段视频描绘的是什么景象？'}]}]
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen3.7-plus',
    messages=messages)

print(response.output.choices[0].message.content[0]["text"])