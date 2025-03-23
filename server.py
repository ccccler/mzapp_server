from flask import Flask, request, jsonify, redirect, url_for, render_template, abort, Response, stream_with_context
from flask_cors import CORS
from unified_interface import UnifiedInterface
import traceback
import os
import uuid
import logging
import time
from functools import wraps
import json
from flask_socketio import SocketIO, emit
# 删除 werkzeug.contrib.fixers 的导入
# from werkzeug.contrib.fixers import ProxyFix

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 删除 ProxyFix 的使用
# app.wsgi_app = ProxyFix(app.wsgi_app)

# 如果确实需要 ProxyFix 功能，使用新的方式：
from werkzeug.middleware.proxy_fix import ProxyFix
# 配置 ProxyFix，根据你的代理服务器配置调整参数
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# 修改CORS配置，允许小程序域名
CORS(app, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化 UnifiedInterface
try:
    unified = UnifiedInterface()
    logger.info("UnifiedInterface 初始化成功")
except Exception as e:
    logger.error(f"初始化失败: {str(e)}")
    traceback.print_exc()

# 新增：存储会话历史
session_histories = {}

def timeout_handler(timeout=600):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            result = f(*args, **kwargs)
            if time.time() - start_time > timeout:
                logger.error(f"请求处理超时: {time.time() - start_time}秒")
                abort(504)  # Gateway Timeout
            return result
        return wrapped
    return decorator

@app.route('/')
def home():
    return "服务器运行正常"

@app.route('/chat', methods=['GET', 'POST'])
@timeout_handler(timeout=600)
def chat():
    if request.method == 'GET':
        return render_template('imagechat.html')
        
    try:
        logger.info("收到POST请求")
        logger.info(f"请求头: {request.headers}")
        
        # 处理图片上传
        if 'image' in request.files:
            file = request.files['image']
            logger.info(f"收到图片文件: {file.filename}")
            
            if not file.filename:
                logger.error("没有选择文件")
                return jsonify({"error": "没有选择文件"}), 400
            
            # 确保临时目录存在
            temp_dir = "temp_images"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            # 使用uuid生成唯一的文件名
            temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
            logger.info(f"保存图片到临时文件: {temp_path}")
            
            try:
                file.save(temp_path)
                logger.info("图片保存成功")
                
                # 使用UnifiedInterface分析图片
                logger.info("开始分析图片")
                result = unified.analyze('image', temp_path)
                logger.info(f"图片分析结果: {result}")
                
                # 生成新的会话ID
                session_id = f"session_{uuid.uuid4()}"
                unified.session_id = session_id
                logger.info(f"生成新会话ID: {session_id}")
                
                # 删除临时文件
                try:
                    os.remove(temp_path)
                    logger.info("临时文件删除成功")
                except Exception as e:
                    logger.error(f"删除临时文件失败: {str(e)}")
                
                # 修改这里：直接返回分析结果，不需要包装在初始消息中
                response_data = {
                    "success": True,
                    "message": result,  # 直接返回分析结果
                    "sessionId": session_id
                }
                logger.info(f"返回响应: {response_data}")
                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"处理图片过程中出错: {str(e)}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                        logger.info("错误后清理临时文件成功")
                    except:
                        pass
                return jsonify({"error": f"处理图片失败: {str(e)}"}), 500
            
        # 处理文本消息
        else:
            data = request.json
            logger.info(f"收到文本请求数据: {data}")
            
            question = data.get('question')
            session_id = data.get('sessionId')
            
            if not question:
                logger.error("问题不能为空")
                return jsonify({"error": "问题不能为空"}), 400
                
            logger.info(f"处理问题: {question}, 会话ID: {session_id}")
            
            # 设置会话ID
            unified.session_id = session_id
            
            # 获取回答
            response = ""
            try:
                # 添加错误处理和重试逻辑
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        for chunk in unified.analyze('text', question, stream=True):
                            logger.debug(f"收到响应块: {chunk}")
                            response += chunk
                        # 如果成功获取响应，跳出循环
                        break
                    except TypeError as e:
                        logger.error(f"第 {retry_count + 1} 次尝试失败: {str(e)}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise Exception("多次尝试后仍然失败")
                        time.sleep(1)  # 等待1秒后重试
                
                if not response:
                    logger.error("生成回答为空")
                    return jsonify({
                        "success": False,
                        "message": "抱歉，我现在无法正确理解您的问题，请稍后再试。",
                        "sessionId": session_id
                    })
                
                logger.info(f"完整响应: {response}")
                return jsonify({
                    "success": True,
                    "message": response,
                    "sessionId": session_id
                })
                
            except Exception as e:
                logger.error(f"生成回答时出错: {str(e)}")
                error_message = "抱歉，处理您的问题时出现错误。请稍后重试。"
                if "multiple retries" in str(e).lower():
                    error_message = "服务暂时不可用，请稍后再试。"
                return jsonify({
                    "success": False,
                    "message": error_message,
                    "sessionId": session_id
                })
            
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "服务器处理请求时发生错误，请稍后重试。",
            "error": str(e)
        }), 500

@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    data = request.get_json()
    message = data.get('message', '')
    session_id = data.get('sessionId', '')
    
    def generate():
        for chunk in unified.process_message_stream(message, session_id):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream')

# 添加 WebSocket 事件处理器
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('message')
def handle_message(data):
    try:
        if isinstance(data, str):
            data = json.loads(data)
        
        message = data.get('message', '')
        session_id = data.get('sessionId', '')
        
        if not message:
            emit('error', {'error': '消息不能为空'})
            return
            
        logger.info(f"处理WebSocket消息: {message}, 会话ID: {session_id}")
        
        # 设置会话ID
        unified.session_id = session_id
        
        try:
            # 使用流式处理发送响应
            for chunk in unified.analyze('text', message, stream=True):
                emit('response', {
                    'chunk': chunk,
                    'sessionId': session_id,
                    'timestamp': time.time()
                })
                time.sleep(0.05)  # 添加小延迟使输出更自然
            
            # 发送完成信号
            emit('complete', {
                'sessionId': session_id,
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"生成响应时出错: {str(e)}")
            emit('error', {
                'error': '生成响应时出错，请稍后重试',
                'details': str(e)
            })
            
    except Exception as e:
        logger.error(f"处理WebSocket消息时出错: {str(e)}")
        emit('error', {'error': '服务器处理消息时出错'})

if __name__ == '__main__':
    logger.info("启动服务器...")
    # 使用 socketio.run 替代 app.run
    socketio.run(
        app,
        host='127.0.0.1',
        port=5000,
        debug=True  # 开发环境可以启用debug模式
    )