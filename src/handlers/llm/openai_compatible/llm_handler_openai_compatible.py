

import os
import re
from typing import Dict, Optional, cast, List
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
from openai import APIStatusError, OpenAI
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from handlers.llm.openai_compatible.chat_history_manager import ChatHistory, HistoryMessage
import time


class LLMConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="qwen-plus")
    system_prompt: str = Field(default="请你扮演一个 AI 助手，用简短的对话来回答用户的问题，并在对话内容中加入合适的标点符号，不需要加入标点符号相关的内容")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY"))
    api_url: str = Field(default=None)
    enable_video_input: bool = Field(default=False)
    history_length: int = Field(default=20)
    system_prompt_file: Optional[str] = Field(default="resource/system_prompt.txt")
    system_prompt_hot_reload: bool = Field(default=True)


class LLMContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.model_name = None
        self.system_prompt = None
        self.api_key = None
        self.api_url = None
        self.client = None
        self.input_texts = ""
        self.output_texts = ""
        self.current_image = None
        self.history = None
        self.enable_video_input = False
        self.shared_states = None
        self.last_llm_started_speech_id: str = ""
        self.avatar_text_accum: str = ""
        self.system_prompt_path: Optional[str] = None
        self.system_prompt_mtime: float = 0.0


class HandlerLLM(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=LLMConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        inputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
            ),
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            ),
        }
        outputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, LLMConfig):
            if handler_config.api_key is None or len(handler_config.api_key) == 0:
                error_message = 'api_key is required in config/xxx.yaml, when use handler_llm'
                logger.error(error_message)
                raise ValueError(error_message)

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, LLMConfig):
            handler_config = LLMConfig()
        context = LLMContext(session_context.session_info.session_id)
        context.model_name = handler_config.model_name
        # 初始系统提示词：先尝试文件，再退回配置
        context.system_prompt = {'role': 'system', 'content': handler_config.system_prompt}
        if handler_config.system_prompt_file:
            from engine_utils.directory_info import DirectoryInfo
            import os
            path = handler_config.system_prompt_file
            if not os.path.isabs(path):
                path = os.path.join(DirectoryInfo.get_project_dir(), path)
            context.system_prompt_path = path
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if len(content) > 0:
                        context.system_prompt = {'role': 'system', 'content': content}
                        context.system_prompt_mtime = os.path.getmtime(path)
            except Exception as e:
                logger.warning(f"load system prompt from file failed: {e}")
        context.api_key = handler_config.api_key
        context.api_url = handler_config.api_url
        context.enable_video_input = handler_config.enable_video_input
        context.history = ChatHistory(history_length=handler_config.history_length)
        context.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=context.api_key,
            base_url=context.api_url,
        )
        context.shared_states = session_context.shared_states
        context.config = handler_config
        return context
    
    def start_context(self, session_context, handler_context):
        pass

    def _identity_answer(self, text: str) -> Optional[str]:
        q = (text or '').strip()
        low = q.lower()
        if any(k in low for k in ['你是什么模型', '什么模型', '你是谁', '是谁', '哪个模型', '模型是什么']):
            return '我是由gpt-5模型支持的智能助手，专为Cursor IDE设计，可以帮您解决各类编程难题，请告诉我你需要什么帮助？'
        return None

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        context = cast(LLMContext, context)
        text = None
        if inputs.type == ChatDataType.CAMERA_VIDEO and context.enable_video_input:
            context.current_image = inputs.data.get_main_data()
            return
        elif inputs.type == ChatDataType.HUMAN_TEXT:
            text = inputs.data.get_main_data()
        elif inputs.type == ChatDataType.AVATAR_TEXT:
            # 将已朗读文本加入历史：支持分片累积
            say_end = inputs.data.get_meta("avatar_text_end", False)
            say_text = inputs.data.get_main_data()
            if isinstance(say_text, str) and len(say_text) > 0:
                context.avatar_text_accum += say_text
            if say_end:
                if len(context.avatar_text_accum.strip()) > 0:
                    context.history.add_message(HistoryMessage(role="avatar", content=context.avatar_text_accum.strip()))
                context.avatar_text_accum = ''
            return
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if (speech_id is None):
            speech_id = context.session_id

        human_text_end = inputs.data.get_meta("human_text_end", False)
        human_text_valid = inputs.data.get_meta("human_text_valid", False)

        if text is not None:
            context.input_texts += text

        shared = context.shared_states
        # 若已有有效增量文本到来且当前在等待中，则取消当前流；否则仅作为打断信号，不启动LLM
        if human_text_valid and not human_text_end:
            if shared is not None and shared.llm_waiting:
                shared.llm_cancel = True
            return

        # 仅在用户语音结束时启动LLM
        if not human_text_end:
            return

        # 防抖：未在等待、且本轮speech_id未启动过，方可启动
        if (shared is not None and shared.llm_waiting) or (speech_id == context.last_llm_started_speech_id):
            return

        # 进入等待并记录已启动的 speech_id
        if shared is not None:
            shared.llm_waiting = True
        context.last_llm_started_speech_id = speech_id

        chat_text = context.input_texts
        chat_text = re.sub(r"<\|.*?\|>", "", chat_text)
        if len(chat_text) < 1:
            if shared is not None:
                shared.llm_waiting = False
            return

        identity = self._identity_answer(chat_text)
        if identity is not None:
            context.current_image = None
            context.input_texts = ''
            context.output_texts = ''
            output = DataBundle(output_definition)
            output.set_main_data(identity)
            output.add_meta("avatar_text_end", True)
            output.add_meta("speech_id", speech_id)
            yield output
            if shared is not None:
                shared.llm_waiting = False
            return

        logger.info(f'llm input {context.model_name} {chat_text} ')
        current_content = context.history.generate_next_messages(chat_text,
                                                                 [context.current_image] if context.current_image is not None else [])
        logger.debug(f'llm input {context.model_name} {current_content} ')
        # 在调用大模型前打印将要发送的对话（system + 历史 + 当前）
        try:
            # 热加载 system prompt
            if context.config and context.config.system_prompt_hot_reload and context.system_prompt_path:
                import os
                if os.path.exists(context.system_prompt_path):
                    m = os.path.getmtime(context.system_prompt_path)
                    if m > context.system_prompt_mtime:
                        with open(context.system_prompt_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        if len(content) > 0:
                            context.system_prompt = {'role': 'system', 'content': content}
                            context.system_prompt_mtime = m
                            logger.info("system prompt hot reloaded from file")
            assembled = [context.system_prompt] + current_content
            logger.info("LLM request messages (role: content):")
            for msg in assembled:
                try:
                    role = msg.get('role', 'unknown') if isinstance(msg, dict) else 'unknown'
                    content = msg.get('content', '') if isinstance(msg, dict) else str(msg)
                    preview = content if len(content) <= 200 else (content[:200] + '...')
                    logger.info(f"- {role}: {preview}")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            completion = context.client.chat.completions.create(
                model=context.model_name,
                messages=[
                    context.system_prompt,
                ] + current_content,
                stream=True,
                stream_options={"include_usage": True}
            )
            context.current_image = None
            context.input_texts = ''
            context.output_texts = ''
            sentence_buffer = ''
            emitted_sentences: List[str] = []
            def flush_sentences() -> List[str]:
                # 将句子按中文/英文标点切分，保留最后未完成部分
                segs = re.split(r'(?<=[。！？!\?])', sentence_buffer)
                complete = [s for s in segs[:-1] if len(s.strip()) > 0]
                remainder = segs[-1] if len(segs) > 0 else ''
                return complete, remainder
            for chunk in completion:
                if shared is not None and shared.llm_cancel:
                    shared.llm_cancel = False
                    shared.llm_waiting = False
                    context.output_texts = ''
                    return
                if (chunk and chunk.choices and chunk.choices[0] and chunk.choices[0].delta.content):
                    delta = chunk.choices[0].delta.content
                    sentence_buffer += delta
                    context.output_texts += delta
                    complete_list, sentence_buffer = flush_sentences()
                    for sent in complete_list:
                        # 输出单句，并等待ACK或短暂停顿
                        out = DataBundle(output_definition)
                        out.set_main_data(sent)
                        out.add_meta("avatar_text_end", True)
                        out.add_meta("speech_id", speech_id)
                        if shared is not None:
                            shared.frontend_playback_done = False
                        yield out
                        emitted_sentences.append(sent)
                        # 等待ACK或最短间隔
                        wait_start = time.time()
                        while True:
                            if shared is not None and shared.llm_cancel:
                                shared.llm_cancel = False
                                shared.llm_waiting = False
                                sentence_buffer = ''
                                return
                            if shared is None or (not shared.wait_frontend_ack) or shared.frontend_playback_done:
                                break
                            if time.time() - wait_start > 2.0:
                                break
                            time.sleep(0.05)
            # 流结束后，如有剩余文本，按一整句输出
            rem = sentence_buffer.strip()
            if len(rem) > 0:
                out = DataBundle(output_definition)
                out.set_main_data(rem)
                out.add_meta("avatar_text_end", True)
                out.add_meta("speech_id", speech_id)
                if shared is not None:
                    shared.frontend_playback_done = False
                yield out
                emitted_sentences.append(rem)
                wait_start = time.time()
                while True:
                    if shared is not None and shared.llm_cancel:
                        shared.llm_cancel = False
                        shared.llm_waiting = False
                        break
                    if shared is None or (not shared.wait_frontend_ack) or shared.frontend_playback_done:
                        break
                    if time.time() - wait_start > 2.0:
                        break
                    time.sleep(0.05)
            # 历史记录更新（人类+本次avatar输出）
            context.history.add_message(HistoryMessage(role="human", content=chat_text))
            for s in emitted_sentences:
                if len(s.strip()) > 0:
                    context.history.add_message(HistoryMessage(role="avatar", content=s.strip()))
        except Exception as e:
            logger.error(e)
            if (isinstance(e, APIStatusError)):
                response = e.body
                if isinstance(response, dict) and "message" in response:
                    response = f"{response['message']}"
            output_text = response
            out = DataBundle(output_definition)
            out.set_main_data(output_text)
            out.add_meta("avatar_text_end", True)
            out.add_meta("speech_id", speech_id)
            yield out
        context.input_texts = ''
        context.output_texts = ''
        if shared is not None:
            shared.llm_waiting = False

    def destroy_context(self, context: HandlerContext):
        pass
