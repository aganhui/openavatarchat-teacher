import os
import re
import threading
import time
import json
from typing import Dict, Optional, List, cast, Any

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
from engine_utils.directory_info import DirectoryInfo


class ScriptReaderConfig(HandlerBaseConfigModel, BaseModel):
    # 兼容旧字段，但优先使用 JSON
    script_file: Optional[str] = Field(default="resource/script.txt")
    script_json_file: Optional[str] = Field(default="resource/script.json")
    sentence_delay: float = Field(default=0.0)
    loop: bool = Field(default=False)
    hot_reload: bool = Field(default=True)
    hot_reload_interval: float = Field(default=0.5)
    frontend_ready_timeout_ms: int = Field(default=5000)


class ScriptReaderContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.shared_states = None
        self._loop_thread: Optional[threading.Thread] = None
        self._running: bool = False
        # JSON 稿件结构：list[ { image: str, lines: list[str] } ]
        self._pages: List[Dict[str, Any]] = []
        self._page_idx: int = 0
        self._line_idx: int = 0
        self._config: Optional[ScriptReaderConfig] = None
        self._last_mtime: float = 0.0
        self._last_path: Optional[str] = None


class HandlerScriptReader(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self._config: Optional[ScriptReaderConfig] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=ScriptReaderConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        inputs: Dict[ChatDataType, HandlerDataInfo] = {}
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
        if isinstance(handler_config, ScriptReaderConfig):
            self._config = handler_config
        else:
            self._config = ScriptReaderConfig()

    def _abs_path(self, path: str) -> str:
        if not os.path.isabs(path):
            return os.path.join(DirectoryInfo.get_project_dir(), path)
        return path

    def _try_parse_json(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                pages: List[Dict[str, Any]] = []
                for item in data:
                    image = item.get("image") if isinstance(item, dict) else None
                    lines = item.get("lines") if isinstance(item, dict) else None
                    if not isinstance(lines, list):
                        continue
                    # 规整每句话
                    norm_lines = []
                    for s in lines:
                        if not isinstance(s, str):
                            continue
                        s2 = s.strip()
                        if s2:
                            norm_lines.append(s2)
                    if len(norm_lines) == 0:
                        continue
                    pages.append({"image": image, "lines": norm_lines})
                if len(pages) > 0:
                    return pages
        except Exception as e:
            logger.warning(f"parse script json failed: {e}")
        return None

    def _fallback_from_text(self, file_path: str) -> List[Dict[str, Any]]:
        content = "大家好，我是智能数字人。现在为您开始播报内容。"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                pass
        segments = re.split(r"(?<=[。！？!\?])|\n+", content)
        lines = [seg.strip() for seg in segments if seg is not None and len(seg.strip()) > 0]
        return [{"image": None, "lines": lines}]

    def _load_pages(self, context: ScriptReaderContext) -> None:
        cfg = cast(ScriptReaderConfig, context._config)
        json_path = self._abs_path(cfg.script_json_file or "")
        txt_path = self._abs_path(cfg.script_file or "")
        chosen_path = None
        pages = None
        if cfg.script_json_file and os.path.exists(json_path):
            pages = self._try_parse_json(json_path)
            chosen_path = json_path
        if pages is None:
            pages = self._fallback_from_text(txt_path)
            chosen_path = txt_path
        context._pages = pages
        context._last_path = chosen_path
        try:
            context._last_mtime = os.path.getmtime(chosen_path)
        except Exception:
            context._last_mtime = 0.0
        # 若越界重置
        if context._page_idx >= len(context._pages):
            context._page_idx = 0
            context._line_idx = 0

    def _hot_reload_if_needed(self, context: ScriptReaderContext):
        cfg = cast(ScriptReaderConfig, context._config)
        if not cfg.hot_reload:
            return
        now = time.time()
        # 用 sleep 周期控制频率，真实检查由 mtime 判断
        try:
            if context._last_path and os.path.exists(context._last_path):
                m = os.path.getmtime(context._last_path)
                if m > context._last_mtime:
                    logger.info("script json hot reloaded")
                    cur_page, cur_line = context._page_idx, context._line_idx
                    self._load_pages(context)
                    # 尝试保留当前位置
                    context._page_idx = min(cur_page, len(context._pages) - 1)
                    context._line_idx = min(cur_line, len(context._pages[context._page_idx]["lines"]) - 1)
        except Exception as e:
            logger.warning(f"hot reload check failed: {e}")

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, ScriptReaderConfig):
            handler_config = self._config if self._config is not None else ScriptReaderConfig()
        context = ScriptReaderContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        context._config = handler_config
        self._load_pages(context)
        return context

    def start_context(self, session_context, handler_context):
        context = cast(ScriptReaderContext, handler_context)
        detail = self.get_handler_detail(session_context, context)
        output_definition = detail.outputs[ChatDataType.AVATAR_TEXT].definition
        # 日志确认
        try:
            total_pages = len(context._pages)
            first_line = None
            if total_pages > 0 and len(context._pages[0].get("lines", []) or []) > 0:
                first_line = context._pages[0]["lines"][0]
            logger.info(f"script initialized: pages={total_pages}, first_line={first_line}")
        except Exception:
            pass
        context._running = True
        context._loop_thread = threading.Thread(target=self._loop, args=(context, output_definition))
        context._loop_thread.start()


    def _emit_text(self, context: ScriptReaderContext, output_definition: DataBundleDefinition, text: str):
        output = DataBundle(output_definition)
        output.set_main_data(text)
        output.add_meta("avatar_text_end", True)
        output.add_meta("speech_id", context.session_id)
        if context.shared_states is not None:
            context.shared_states.frontend_playback_done = False
        context.submit_data(output)

    def _loop(self, context: ScriptReaderContext, output_definition: DataBundleDefinition):
        cfg = cast(ScriptReaderConfig, context._config)
        # 等待前端就绪（支持超时兜底）
        start_wait = time.time()
        while context._running:
            if context.shared_states is None or context.shared_states.frontend_ready:
                break
            if (time.time() - start_wait) * 1000 >= cfg.frontend_ready_timeout_ms:
                logger.warning("frontend_ready not received in time, fallback to start script")
                break
            time.sleep(0.05)
        while context._running:
            if cfg.hot_reload:
                self._hot_reload_if_needed(context)
            # 暂停条件
            if context.shared_states is not None and (context.shared_states.user_speaking or context.shared_states.avatar_speaking):
                time.sleep(0.05)
                continue
            if context.shared_states is not None and context.shared_states.llm_waiting:
                time.sleep(0.05)
                continue
            if context.shared_states is not None and context.shared_states.wait_frontend_ack and not context.shared_states.frontend_playback_done:
                time.sleep(0.05)
                continue
            # 边界检查
            if len(context._pages) == 0:
                time.sleep(0.2)
                continue
            page = context._pages[context._page_idx]
            lines = page.get("lines", []) or []
            if context._line_idx < len(lines):
                text = lines[context._line_idx]
                context._line_idx += 1
                try:
                    self._emit_text(context, output_definition, text)
                    if cfg.sentence_delay > 0:
                        time.sleep(cfg.sentence_delay)
                except Exception as e:
                    logger.warning(f"script submit failed: {e}")
                    time.sleep(0.1)
                continue
            # 当前页结束，插入“请翻页”并翻页
            try:
                self._emit_text(context, output_definition, "请翻页")
            except Exception:
                pass
            context._page_idx += 1
            context._line_idx = 0
            if context._page_idx >= len(context._pages):
                if cfg.loop:
                    context._page_idx = 0
                    context._line_idx = 0
                else:
                    time.sleep(0.2)
                    continue

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        # 不消费输入
        return

    def destroy_context(self, context: HandlerContext):
        context = cast(ScriptReaderContext, context)
        context._running = False
        if context._loop_thread is not None:
            try:
                context._loop_thread.join(timeout=2)
            except Exception:
                pass
