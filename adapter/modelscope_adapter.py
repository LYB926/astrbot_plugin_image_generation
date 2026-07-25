"""ModelScope API-Inference asynchronous image generation adapter."""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

from astrbot.api import logger

from ..core.adapters.async_polling import (
    ASYNC_AMBIGUOUS_SUBMIT_ERROR,
    AsyncAttemptContext,
    AsyncPollResult,
    AsyncPollingImageAdapter,
    AsyncRemoteStatus,
    AsyncSubmitResult,
)
from ..core.shared.constants import (
    RESOLUTION_1K_MAP,
    RESOLUTION_2K_MAP,
    UNSPECIFIED_OPTION,
)
from ..core.shared.logging import safe_log_error_body, safe_log_mapping
from ..core.shared.types import GenerationRequest, ImageCapability, ImageData

SIZE_PATTERN = re.compile(r"^\d{2,5}x\d{2,5}$")


class ModelScopeAdapter(AsyncPollingImageAdapter):
    """ModelScope API-Inference adapter using submit, poll, and download."""

    DEFAULT_BASE_URL = "https://api-inference.modelscope.cn"
    DEFAULT_MODEL = "Qwen/Qwen-Image"

    def get_capabilities(self) -> ImageCapability:
        """Return provider capabilities selected in the configuration."""
        return self._get_configured_capabilities()

    async def _submit_task(
        self,
        request: GenerationRequest,
        attempt: AsyncAttemptContext,
    ) -> AsyncSubmitResult | tuple[None, str]:
        """Submit an asynchronous ModelScope image generation task.

        Args:
            request: Image generation request to submit.
            attempt: Immutable key snapshot for this remote task.

        Returns:
            A remote task id or an error message.
        """
        start_time = time.monotonic()
        url = self._generation_url()
        payload = self._build_payload(request)
        headers = {
            "Authorization": f"Bearer {attempt.api_key}",
            "Content-Type": "application/json",
            "X-ModelScope-Async-Mode": "true",
        }
        self._log_request_overview(request, url, payload=payload)
        self._log_debug_json("请求", payload, request.task_id)

        try:
            async with self._get_session().post(
                url,
                headers=headers,
                json=payload,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as response:
                duration = time.monotonic() - start_time
                self._log_response_status(request, response.status, duration)
                if not 200 <= response.status < 300:
                    error_text = await response.text()
                    self._log_debug_json_text("响应", error_text, request.task_id)
                    self._log_api_error(request, response.status, duration, error_text)
                    error = self._format_api_error_message(response.status, error_text)
                    if response.status >= 500:
                        return None, f"{ASYNC_AMBIGUOUS_SUBMIT_ERROR}: {error}"
                    return None, error

                data = await self._read_response_json(response, request.task_id)
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start_time
            self._log_request_exception(request, duration, exc)
            return None, (f"{ASYNC_AMBIGUOUS_SUBMIT_ERROR}: {safe_log_error_body(exc)}")

        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not isinstance(task_id, str) or not task_id.strip():
            return None, (f"{ASYNC_AMBIGUOUS_SUBMIT_ERROR}: 响应中未找到 task_id")
        return AsyncSubmitResult(remote_task_id=task_id.strip())

    async def _poll_task(
        self,
        request: GenerationRequest,
        attempt: AsyncAttemptContext,
        remote_task_id: str,
    ) -> AsyncPollResult | tuple[None, str]:
        """Fetch and normalize one ModelScope task status response.

        Args:
            request: Image generation request associated with the task.
            attempt: Immutable key snapshot for this remote task.
            remote_task_id: ModelScope task id.

        Returns:
            A normalized task status or a retry-classifiable request error.
        """
        start_time = time.monotonic()
        url = self._task_url(remote_task_id)
        headers = {
            "Authorization": f"Bearer {attempt.api_key}",
            "X-ModelScope-Task-Type": "image_generation",
        }
        self._log_request_overview(request, url, method="GET")

        try:
            async with self._get_session().get(
                url,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as response:
                duration = time.monotonic() - start_time
                self._log_response_status(request, response.status, duration)
                if not 200 <= response.status < 300:
                    error_text = await response.text()
                    self._log_debug_json_text("响应", error_text, request.task_id)
                    self._log_api_error(request, response.status, duration, error_text)
                    return None, self._format_api_error_message(
                        response.status,
                        error_text,
                    )
                data = await self._read_response_json(response, request.task_id)
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start_time
            self._log_request_exception(request, duration, exc)
            return None, safe_log_error_body(exc)

        if not isinstance(data, dict):
            return AsyncPollResult(
                status=AsyncRemoteStatus.UNKNOWN,
                error_message="响应不是 JSON 对象",
            )

        status = str(data.get("task_status") or "").strip().upper()
        if status == "PENDING":
            return AsyncPollResult(status=AsyncRemoteStatus.PENDING)
        if status in {"PROCESSING", "RUNNING"}:
            return AsyncPollResult(status=AsyncRemoteStatus.RUNNING)
        if status == "FAILED":
            return AsyncPollResult(
                status=AsyncRemoteStatus.FAILED,
                error_message=self._extract_task_error(data),
            )
        if status != "SUCCEED":
            return AsyncPollResult(
                status=AsyncRemoteStatus.UNKNOWN,
                error_message=f"未识别的 task_status: {status or '缺失'}",
            )

        output_images = data.get("output_images")
        if not isinstance(output_images, list):
            return AsyncPollResult(
                status=AsyncRemoteStatus.SUCCEEDED,
                error_message="成功响应中 output_images 不是列表",
            )
        image_urls = tuple(
            item.strip()
            for item in output_images
            if isinstance(item, str) and item.strip()
        )
        if len(image_urls) != len(output_images):
            return AsyncPollResult(
                status=AsyncRemoteStatus.SUCCEEDED,
                error_message="成功响应中包含无效图片地址",
            )
        return AsyncPollResult(
            status=AsyncRemoteStatus.SUCCEEDED,
            image_urls=image_urls,
        )

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """Build the documented ModelScope image generation payload.

        Args:
            request: Image generation request to serialize.

        Returns:
            JSON-ready ModelScope request data.
        """
        payload: dict[str, Any] = {
            "model": self.model or self.DEFAULT_MODEL,
            "prompt": request.prompt,
        }
        if negative_prompt := str(
            self.config.extra.get("negative_prompt") or ""
        ).strip():
            payload["negative_prompt"] = negative_prompt
        if size := self._resolve_size(request):
            payload["size"] = size
        if request.images and self.get_capabilities() & ImageCapability.IMAGE_TO_IMAGE:
            payload["image_url"] = [
                self._image_to_data_url(image) for image in request.images
            ]
        return payload

    def _generation_url(self) -> str:
        """Build the ModelScope asynchronous image submission URL."""
        return f"{self._service_root()}/v1/images/generations"

    def _task_url(self, remote_task_id: str) -> str:
        """Build the ModelScope task status URL.

        Args:
            remote_task_id: ModelScope task id returned by submission.

        Returns:
            Fully qualified task status URL.
        """
        return f"{self._service_root()}/v1/tasks/{remote_task_id}"

    def _service_root(self) -> str:
        """Normalize the configured ModelScope service root URL."""
        base = (self.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        for suffix in ("/v1/images/generations", "/v1/tasks", "/v1"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    def _resolve_size(self, request: GenerationRequest) -> str | None:
        """Resolve a documented explicit width-by-height ModelScope size.

        Args:
            request: Image generation request with optional ratio and resolution.

        Returns:
            A validated ``WxH`` value or ``None`` when no size is requested.
        """
        resolution = self._option_value(request.resolution)
        aspect_ratio = self._option_value(request.aspect_ratio)
        for value in (
            resolution,
            self._size_map_value(resolution, aspect_ratio),
            self.config.extra.get("default_size"),
        ):
            if isinstance(value, str) and SIZE_PATTERN.fullmatch(value.strip()):
                return value.strip()

        if not resolution and not aspect_ratio:
            return None

        size_map = (
            RESOLUTION_2K_MAP if resolution in {"2K", "4K"} else RESOLUTION_1K_MAP
        )
        return size_map.get(aspect_ratio or "1:1", size_map["1:1"])

    def _size_map_value(
        self,
        resolution: str | None,
        aspect_ratio: str | None,
    ) -> Any:
        """Read a configured resolution/aspect-ratio size map entry.

        Args:
            resolution: Requested resolution tier or explicit size.
            aspect_ratio: Requested aspect ratio.

        Returns:
            The first matching configured size-map value, if valid JSON exists.
        """
        raw_map = self.config.extra.get("size_map_json")
        if isinstance(raw_map, str):
            try:
                size_map = json.loads(raw_map)
            except json.JSONDecodeError:
                logger.warning(
                    f"{self._get_log_prefix()} size_map_json JSON 解析失败，已忽略"
                )
                return None
        else:
            size_map = raw_map
        if not isinstance(size_map, dict):
            return None

        keys = [
            f"{resolution}:{aspect_ratio}" if resolution and aspect_ratio else None,
            resolution,
            aspect_ratio,
            "default",
        ]
        for key in keys:
            if key and key in size_map:
                return size_map[key]
        return None

    def _option_value(self, value: str | None) -> str | None:
        """Return an option unless it means that the request omits it."""
        if not value or value == UNSPECIFIED_OPTION:
            return None
        return value

    def _image_to_data_url(self, image: ImageData) -> str:
        """Encode one reference image as a ModelScope-compatible data URL."""
        mime_type = (image.mime_type or "image/png").lower()
        encoded = base64.b64encode(image.data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _extract_task_error(self, data: dict[str, Any]) -> str:
        """Extract a compact provider failure message from a task response."""
        for key in ("error", "message", "error_message", "code"):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (dict, list)):
                return safe_log_mapping(value)
            return str(value)
        return "远端任务返回 FAILED"


__all__ = ("ModelScopeAdapter",)
