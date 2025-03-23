from face_analyzer import FaceAnalyzer
from his_simi_str import HistoryAwareRAG_simi
import logging

logger = logging.getLogger(__name__)

class UnifiedInterface:
    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.rag_analyzer = HistoryAwareRAG_simi()
        self.session_id = "default_session"  # 可以在需要时修改session_id

    def analyze(self, type, content, stream=False):
        """
        统一的分析接口
        :param type: 输入类型，'text' 或 'image'
        :param content: 如果是text则为提示词文本，如果是image则为图片路径
        :param stream: 是否使用流式输出（仅对text类型有效）
        :return: 分析结果
        """
        try:
            if type == 'text':
                # 使用HistoryAwareRAG_simi处理文本
                try:
                    if stream:
                        # 返回生成器，支持流式输出
                        return self.rag_analyzer.query(content, session_id=self.session_id)
                    else:
                        # 如果不需要流式输出，使用get_response方法
                        return self.rag_analyzer.get_response(content)
                except Exception as e:
                    logger.error(f"处理文本时出错: {str(e)}")
                    return "抱歉，处理您的问题时出现错误。"
                
            elif type == 'image':
                # 使用FaceAnalyzer处理图片
                try:
                    return self.face_analyzer.analyze(image_path=content)
                except Exception as e:
                    logger.error(f"图片分析失败: {str(e)}")
                    return f"图片分析失败: {str(e)}"
            else:
                return "不支持的输入类型，请使用 'text' 或 'image'"
            
        except Exception as e:
            logger.error(f"分析过程中出错: {str(e)}")
            return "服务器处理请求时发生错误，请稍后重试。"

if __name__ == "__main__":
    # 使用示例
    analyzer = UnifiedInterface()
    
    # 分析文本示例（流式输出）
    print("文本分析结果:")
    for chunk in analyzer.analyze('text', '请问如何护理油性肌肤？', stream=True):
        print(chunk, end="", flush=True)
    print("\n")
