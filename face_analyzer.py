import base64
from openai import OpenAI
from zhipuai import ZhipuAI

class FaceAnalyzer:
    def __init__(self):
        # API密钥配置
        self.api_keys = {
            'kimi': "sk-kOlqtRLgFMBOZtjqyCNMECsuElayWrPm2NzzhWLqesxIRmaK",
            'zhipu': "7123d59518246f15a00fd8d1c36a2447.PlYA3dT3cfw6xsVc",
            'qwen': "sk-dd6a3bf1168944609a4ee55170fad149"
        }
        
        # API基础URL配置
        self.base_urls = {
            'kimi': "https://api.moonshot.cn/v1",
            'qwen': "https://dashscope.aliyuncs.com/compatible-mode/v1"
        }
        
        # 模型名称配置
        self.model_names = {
            'kimi': "moonshot-v1-8k-vision-preview",
            'zhipu': "glm-4v-plus",
            'qwen': "qwen-vl-plus"
        }

        self.prompt = '''假设你是一个肤质分析专家，请你帮我分析这张图片里人物的肤质，并给出肤质分析报告。最终以markdown格式返回。
        #需要包含的维度有：黑眼圈、肤质情况（油性、干性、混合性）、出油、毛孔、黑头、痘痘、敏感度、黑色素、水分、粗糙度、肤龄、皱纹分布（抬头纹、鱼尾纹、眼部细纹、眉间纹、法令纹）
        #在最终的皮肤报告里，每个部分都需要用完整的文字来描述情况。'''

    def _encode_image(self, image_path):
        """将图片转换为base64编码"""
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def analyze(self, model_type='qwen', image_path=None):
        """
        分析人脸图片
        :param model_type: 模型类型 ('kimi', 'zhipu', 或 'qwen')，默认为'qwen'
        :param image_path: 图片路径
        :return: 分析结果
        """
        if image_path is None:
            raise ValueError("必须提供图片路径")
        
        img_base = self._encode_image(image_path)
        
        if model_type not in ['kimi', 'zhipu', 'qwen']:
            raise ValueError("不支持的模型类型。请使用 'kimi', 'zhipu' 或 'qwen'")

        if model_type == 'zhipu':
            return self._analyze_zhipu(img_base, self.prompt)
        else:  # kimi 和 qwen 使用相同的 OpenAI 客户端格式
            return self._analyze_openai(model_type, img_base, self.prompt)

    def _analyze_zhipu(self, img_base,prompt):
        """智谱AI分析方法"""
        client = ZhipuAI(api_key=self.api_keys['zhipu'])
        response = client.chat.completions.create(
            model=self.model_names['zhipu'],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_base
                        }
                    },
                    {
                        "type": "text",
                        "text": self.prompt
                    }
                ]
            }]
        )
        return response.choices[0].message.content

    def _analyze_openai(self, model_type, img_base, prompt):
        """OpenAI格式的API分析方法（适用于Kimi和Qwen）"""
        client = OpenAI(
            api_key=self.api_keys[model_type],
            base_url=self.base_urls[model_type]
        )
        
        response = client.chat.completions.create(
            model=self.model_names[model_type],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base}"
                        }
                    },
                    {
                        "type": "text",
                        "text": self.prompt
                    }
                ]
            }]
        )
        return response.choices[0].message.content

    def save_result(self, content, model_type):
        """保存分析结果到文件"""
        filename = f'{model_type}_analysis_result.md'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content) 

if __name__ == "__main__":

# 创建分析器实例
    analyzer = FaceAnalyzer()

# 设置提示词

    # 图片路径
    image_path = "./face_analysis/照片1.jpg"

    # 使用不同模型进行分析
    model = 'kimi'

    result = analyzer.analyze(model_type='qwen', image_path=image_path)
    print(result)
