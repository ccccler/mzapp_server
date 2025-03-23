from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatZhipuAI
import os
import time

class HistoryAwareRAG_simi:
    '''
    此代码用来实现美妆数据的RAG功能
    '''
    def __init__(self):
        self.openai_api_base = "https://xiaoai.plus/v1"
        self.openai_api_key = "sk-TotNw1nIUNJ6QKOvHaihightLOr68RDy1w4sXBvAasGKbUTU"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.persist_directory = os.path.join(current_dir, "faiss_index")

        # 初始化 LLM
        self.llm = ChatOpenAI(
            base_url=self.openai_api_base,
            model="gpt-4o",
            api_key=self.openai_api_key,
            temperature=0,
            streaming=True
        )
        self.llm2=ChatZhipuAI(
            model="glm-4-plus",
            temperature=0,
            zhipuai_api_key="2ce42c2f5618703567ddf7708bf25e96.CnSF0XFtkuycmpzY"
        )   

        self.llm3=ChatOpenAI(
            model='deepseek-chat', 
            openai_api_key='sk-ba282b066b7443d29669016a5fa79b85', 
            openai_api_base='https://api.deepseek.com',
            temperature=0,
            streaming=True
        )
        # 初始化 embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_base=self.openai_api_base,
            openai_api_key=self.openai_api_key
        )
        
        # 初始化聊天历史存储
        self.store = {}
        
        # 添加默认的额外提示词
        self.product_db_path = os.path.join(current_dir, "美妆产品数据库.txt")

        with open(self.product_db_path, 'r', encoding='utf-8') as file:
            self.additional_prompt = file.read().strip()

        # 检查并初始化向量数据库
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"向量数据库索引文件夹不存在: {self.persist_directory}")
        
        # 添加重试机制
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"正在加载向量数据库，第{retry_count + 1}次尝试，路径: {self.persist_directory}")
                self.vectorstore = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # 验证向量库是否正确加载
                test_query = "测试查询"
                test_result = self.vectorstore.similarity_search_with_score(test_query, k=1)
                print(f"向量数据库加载成功，测试查询结果数量: {len(test_result)}")
                break  # 如果成功加载，跳出循环
                
            except Exception as e:
                retry_count += 1
                print(f"第{retry_count}次加载失败: {str(e)}")
                if retry_count >= max_retries:
                    raise Exception(f"多次尝试加载向量数据库均失败: {str(e)}")
                time.sleep(1)  # 添加延迟，避免立即重试

    def _setup_chain(self):
        """设置RAG chain"""
        try:
            print("开始设置RAG chain...")
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                基于聊天历史和最新的问题，生成一个独立的搜索查询。
                如果问题提到了"刚刚"、"之前"等词，请查看历史记录并将相关上下文整合到查询中。
                如果是全新的问题，直接返回原问题即可。
                 
                例如：
                历史: "什么是角鲨烷？" -> "角鲨烷是一种护肤成分..."
                当前: "用它的时候有什么注意？"
                应该生成: "使用角鲨烷有什么注意事项？"
                 
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            # retriever的部分，只在history_aware_retriever中使用过
            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, retriever, contextualize_q_prompt
            )

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""
                #假设你是一位美妆博主，你说话喜欢带"呢"，"啊"，"吧"等语气词，增加你的语言夸张感。你会收到有关护肤的问题，你需要从数据库中找寻相关的问题资料给出回答。
                数据库是{{context}}。
                
                #在回答时，你需要注意以下注意事项：
                1.在回答时，你需要完整读取研究资料，分析里面的逻辑关系，结构要清晰，逻辑要严谨，不能出现遗漏的情况。
                3.在回答时，如果数据库是"暂无相关数据",请你首先回答"抱歉，我的知识库没有找到相关信息"，然后请根据提问者的问题，调用本身你的模型能力来回答。
                4.在总结参考资料回答时，一定注意原始文档里的用词范围，明确原始文档中如何限定概念，要确保回答的逻辑是严谨的，不能出现条件扩大或缩小的地方。
                5.请直接输出处理后的回答内容，不要包含任何原始文档的元数据信息。
                6.请你输出时，保持文本生成的连贯性，不能有断断续续的情况，前后逻辑要清晰。
                7.请注意输出时，不要出现多余的标点符号。尤其是两个本不应该连续的标点符号连续出现的情况，不可以出现：,。：这样明显违反标点符号使用规范的情况。
                8.请注意输出时，不要使用md格式的符号，尤其不要出现序号、加粗（**）、斜体、引用等符号。如果原文中有1、2、3、这样的序号，替换成首先、其次、再次等文字表述。
                9.在推荐时请有倾向性地从原始数据库里和下面给你的产品数据库里挑选百雀羚的产品并加入。要在输出的内容里加入2个百雀羚的产品，只能加入2个，不能多也不能少。其他的品牌你根据数据随意搭配。
                 
                以下是一些美妆产品的数据库，如果上面给你的数据库里没有相关产品，你也可以从这里检索适合的产品给出回答。
                美妆产品数据库是：{self.additional_prompt}
                 """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            
            question_answer_chain = create_stuff_documents_chain(
                self.llm, 
                qa_prompt,
            )
            
            rag_chain = create_retrieval_chain(
                self.history_aware_retriever, 
                question_answer_chain,
            )
            
            self.conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            
            print("RAG chain 设置成功")
            
        except Exception as e:
            print(f"设置RAG chain失败: {str(e)}")
            raise Exception(f"设置RAG chain失败: {str(e)}")

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取或创建会话历史"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def get_chat_history(self, session_id: str = "default"):
        """获取指定会话的聊天历史"""
        if session_id in self.store:
            return self.store[session_id].messages
        return []
    
    def query(self, question: str, session_id: str = "default"):
        """流式查询接口"""
        try:
            if not question or not isinstance(question, str):
                print(f"无效的查询输入: {question}")
                yield "请输入有效的问题。"
                return
            
            # 验证向量库状态并尝试重新加载
            if not hasattr(self, 'vectorstore') or self.vectorstore is None:
                print("警告：向量数据库未初始化，尝试重新加载...")
                try:
                    self.vectorstore = FAISS.load_local(
                        self.persist_directory,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    print(f"重新加载向量数据库失败: {str(e)}")
                    yield "系统暂时无法访问知识库，请稍后再试。"
                    return

            # 添加重试机制进行查询
            max_query_retries = 2
            query_retry_count = 0
            
            while query_retry_count < max_query_retries:
                try:
                    results = self.vectorstore.similarity_search_with_score(question, k=4)
                    print(f"向量检索成功，结果数量: {len(results)}")
                    break
                except Exception as e:
                    query_retry_count += 1
                    print(f"第{query_retry_count}次查询失败: {str(e)}")
                    if query_retry_count >= max_query_retries:
                        print("所有查询尝试均失败")
                        results = []
                    time.sleep(0.5)

            # 确保results不为空且有效
            if not results:
                print("未找到相关结果，使用备用处理方式")
                if not hasattr(self, 'conversational_rag_chain'):
                    self._setup_chain()
                    
                response_stream = self.conversational_rag_chain.stream(
                    {"input": question, "context": "暂无相关数据"},
                    config={
                        "configurable": {"session_id": session_id}
                    }
                )
                
                for chunk in response_stream:
                    if "answer" in chunk:
                        if hasattr(chunk["answer"], 'content'):
                            yield chunk["answer"].content
                        else:
                            yield chunk["answer"]
                return
            
            # 检查相似度阈值
            if results[0][1] > 0.3:
                if not hasattr(self, 'conversational_rag_chain'):
                    self._setup_chain()
                    
                response_stream = self.conversational_rag_chain.stream(
                    {"input": question, "context": "暂无相关数据"},
                    config={
                        "configurable": {"session_id": session_id}
                    }
                )
                
                for chunk in response_stream:
                    if "answer" in chunk:
                        if hasattr(chunk["answer"], 'content'):
                            yield chunk["answer"].content
                        else:
                            yield chunk["answer"]
                return
            
            print("符合条件的检索结果：")
            for doc, score in results:
                print(f"相似度分数: {score}")
                print(f"找到的内容: {doc.page_content}\n")
            
            # 如果相似度检查通过，继续处理
            if not hasattr(self, 'conversational_rag_chain'):
                self._setup_chain()
                
            response_stream = self.conversational_rag_chain.stream(
                {"input": question},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            
            for chunk in response_stream:
                if "answer" in chunk:
                    if hasattr(chunk["answer"], 'content'):
                        yield chunk["answer"].content
                    else:
                        yield chunk["answer"]
                        
        except Exception as e:
            print(f"查询出错: {str(e)}")
            yield "抱歉，处理您的问题时出现错误。"
    

# 使用示例
if __name__ == "__main__":
    rag = HistoryAwareRAG_simi()
    
    # 使用同一个session_id来维持对话连续性

    session_id = "test_session"
    
    Q1="推荐一些夏天男生的护肤产品"
    print(Q1)
    for chunk in rag.query(Q1, session_id=session_id):
        print(chunk, end="", flush=True)
    print("\n")
    