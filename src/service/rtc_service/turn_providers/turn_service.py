from typing import Type, Optional

from pydantic import BaseModel

from service.rtc_service.base_turn_provider import (
    BaseRtcTurnProvider,
    BaseRtcTurnEntity,
)

import os
import re
import subprocess
from urllib.request import urlopen
from urllib.error import URLError


class TurnServerConfigData(BaseModel):
    urls: Optional[list[str]] = None
    username: str
    credential: str


class TurnServerProvider(BaseRtcTurnProvider):

    def get_config_model(self) -> Type[BaseModel]:
        return TurnServerConfigData

    def _is_valid_ip(self, value: str) -> bool:
        if not value:
            return False
        pattern = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
        if not pattern.match(value):
            return False
        parts = value.split(".")
        return all(0 <= int(p) <= 255 for p in parts)

    def _detect_public_ip(self) -> Optional[str]:
        env_ip = os.getenv("TURN_PUBLIC_IP")
        if env_ip and self._is_valid_ip(env_ip):
            return env_ip
        for url in ("https://ipinfo.io/ip", "https://ifconfig.me"):
            try:
                with urlopen(url, timeout=2) as resp:
                    data = resp.read().decode("utf-8").strip()
                    if self._is_valid_ip(data):
                        return data
            except URLError:
                continue
            except Exception:
                continue
        return None

    def _detect_private_ip(self) -> Optional[str]:
        env_ip = os.getenv("TURN_PRIVATE_IP")
        if env_ip and self._is_valid_ip(env_ip):
            return env_ip
        cmds = [
            ["bash", "-lc", "ip -4 addr | grep 'inet ' | awk '{print $2}' | cut -d/ -f1"],
            ["bash", "-lc", "ifconfig | grep 'inet ' | awk '{print $2}'"],
        ]
        for cmd in cmds:
            try:
                output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=2).decode("utf-8")
                for line in output.splitlines():
                    ip = line.strip()
                    if not self._is_valid_ip(ip):
                        continue
                    # 跳过回环与常见容器网卡
                    if ip.startswith("127."):
                        continue
                    return ip
            except Exception:
                continue
        return None

    def _apply_placeholders(self, url: str, public_ip: Optional[str], private_ip: Optional[str]) -> str:
        if public_ip:
            url = url.replace("{PUBLIC_IP}", public_ip)
        if private_ip:
            url = url.replace("{PRIVATE_IP}", private_ip)
        return url

    def prepare_rtc_configuration(self, config: BaseModel):
        public_ip = self._detect_public_ip()
        private_ip = self._detect_private_ip()

        urls: list[str]
        if getattr(config, "urls", None):
            urls = [self._apply_placeholders(u, public_ip, private_ip) for u in config.urls]  # type: ignore[attr-defined]
        else:
            # 若未提供 urls，则基于探测到的公网/内网 IP 自动生成缺省配置
            base_ip = public_ip or private_ip or "127.0.0.1"
            urls = [
                f"turn:{base_ip}:3478",
                f"turns:{base_ip}:5349",
            ]

        result = BaseRtcTurnEntity()
        result.rtc_configuration = {
            "iceServers": [
                {
                    "urls": urls,
                    "username": getattr(config, "username"),
                    "credential": getattr(config, "credential"),
                }
            ]
        }
        return result
