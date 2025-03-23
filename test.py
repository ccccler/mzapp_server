import requests
import websocket
import json
import os
from PIL import Image
import io

def test_server():
    base_url = "http://127.0.0.1:5000"
    
    # 1. 测试服务器是否在运行
    def test_server_status():
        response = requests.get(base_url)
        print(f"服务器状态: {response.text}")

    # 2. 测试文本聊天功能
    def test_chat_text():
        url = f"{base_url}/chat"
        data = {
            "question": "你好，这是一个测试消息",
            "sessionId": "test_session_001"
        }
        response = requests.post(url, json=data)
        print(f"文本聊天测试结果: {response.json()}")

    # 3. 测试图片上传功能
    def test_image_upload():
        url = f"{base_url}/chat"
        # 创建一个测试图片
        img = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        files = {
            'image': ('test.jpg', img_byte_arr, 'image/jpeg')
        }
        response = requests.post(url, files=files)
        print(f"图片上传测试结果: {response.json()}")

    # 4. 测试WebSocket连接
    def test_websocket():
        def on_message(ws, message):
            print(f"收到消息: {message}")

        def on_error(ws, error):
            print(f"错误: {error}")

        def on_close(ws):
            print("连接关闭")

        def on_open(ws):
            print("连接建立")
            # 发送测试消息
            test_message = {
                "message": "WebSocket测试消息",
                "sessionId": "test_session_002"
            }
            ws.send(json.dumps(test_message))

        ws = websocket.WebSocketApp(
            "ws://127.0.0.1:5000/socket.io/?EIO=4&transport=websocket",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        ws.run_forever()

    # 运行所有测试


if __name__ == "__main__":
    tt=test_server()
    print("开始测试...")

    # 测试连接状态
    tt.test_server_status()

    # 测试文本聊天
    tt.test_chat_text()

    # 测试图片上传
    # tt.test_image_upload()

    # 测试WebSocket连接
    tt.test_websocket()