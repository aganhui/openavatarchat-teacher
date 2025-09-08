import os
from typing import Dict, Optional, cast
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC

from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry


class EchoConfig(HandlerBaseConfigModel, BaseModel):
    """简单的回声配置，不需要特殊参数"""
    pass


class EchoContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.input_texts = ""


class HandlerEcho(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="Echo",
            config_model=EchoConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        inputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
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
        # 不需要加载任何模型
        pass

    def create_context(self, session_context, handler_config=None):
        context = EchoContext(session_context.session_info.session_id)
        return context

    def start_context(self, session_context, handler_context):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        context = cast(EchoContext, context)
        text = None

        if inputs.type == ChatDataType.HUMAN_TEXT:
            text = inputs.data.get_main_data()
        else:
            return

        speech_id = inputs.data.get_meta("speech_id")
        if speech_id is None:
            speech_id = context.session_id

        if text is not None:
            context.input_texts += text

        text_end = inputs.data.get_meta("human_text_end", False)
        if not text_end:
            return

        # 直接使用用户输入的文本作为输出
        echo_text = context.input_texts
        if len(echo_text) < 1:
            return

        logger.info(f'Echo input: {echo_text}')

        # 输出文本
        output = DataBundle(output_definition)
        output.set_main_data(echo_text)
        output.add_meta("avatar_text_end", False)
        output.add_meta("speech_id", speech_id)
        yield output

        # 重置输入文本
        context.input_texts = ''

        # 标记文本结束
        logger.info('avatar text end')
        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("avatar_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        pass
