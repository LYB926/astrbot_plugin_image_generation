"""Generic submit-poll-download support for asynchronous image providers."""

from __future__ import annotations

import abc
import asyncio
import enum
import time
from dataclasses import dataclass

from astrbot.api import logger

from ..shared.logging import (
    format_cn_log_fields,
    format_seconds,
    safe_log_error_body,
    safe_log_text,
    safe_log_url,
    safe_user_error_detail,
)
from ..shared.types import GenerationRequest
from .base import BaseImageAdapter

DEFAULT_ASYNC_POLL_INTERVAL = 5
DEFAULT_ASYNC_POLL_TIMEOUT = 600

ASYNC_AMBIGUOUS_SUBMIT_ERROR = "异步提交状态未知"
ASYNC_REMOTE_FAILED_ERROR = "异步任务失败"
ASYNC_POLL_ERROR = "异步轮询失败"
ASYNC_PROTOCOL_ERROR = "异步协议错误"
ASYNC_TIMEOUT_ERROR = "异步等待超时"
ASYNC_DOWNLOAD_ERROR = "异步结果下载失败"
ASYNC_TERMINAL_ERROR_PREFIXES = (
    ASYNC_AMBIGUOUS_SUBMIT_ERROR,
    ASYNC_REMOTE_FAILED_ERROR,
    ASYNC_POLL_ERROR,
    ASYNC_PROTOCOL_ERROR,
    ASYNC_TIMEOUT_ERROR,
    ASYNC_DOWNLOAD_ERROR,
)


class AsyncRemoteStatus(str, enum.Enum):
    """Normalized status of a provider-side asynchronous task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AsyncAttemptContext:
    """Immutable credentials and metadata for one remote task attempt."""

    api_key: str


@dataclass(frozen=True)
class AsyncSubmitResult:
    """Result of successfully submitting a provider-side task."""

    remote_task_id: str


@dataclass(frozen=True)
class AsyncPollResult:
    """Normalized response from one provider-side task status request."""

    status: AsyncRemoteStatus
    image_urls: tuple[str, ...] = ()
    error_message: str | None = None


class AsyncPollingImageAdapter(BaseImageAdapter):
    """Base adapter for providers that submit, poll, and download images."""

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """Submit and resolve one provider-side asynchronous generation task.

        Args:
            request: Image generation request to submit.

        Returns:
            Generated image bytes or a user-facing error message.
        """
        attempt = AsyncAttemptContext(api_key=self._get_current_api_key())
        submit_result = await self._submit_task(request, attempt)
        if not isinstance(submit_result, AsyncSubmitResult):
            _, error = submit_result
            return None, error or "异步任务提交失败"

        remote_task_id = submit_result.remote_task_id.strip()
        if not remote_task_id:
            return None, ASYNC_AMBIGUOUS_SUBMIT_ERROR

        logger.debug(
            f"{self._get_log_prefix(request.task_id)} 异步任务已提交: "
            + format_cn_log_fields(远端任务=safe_log_text(remote_task_id, 120))
        )
        deadline = time.monotonic() + self._get_async_poll_timeout()

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, self._format_async_timeout_error(remote_task_id)

            try:
                poll_result = await asyncio.wait_for(
                    self._poll_task(request, attempt, remote_task_id),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                return None, self._format_async_timeout_error(remote_task_id)

            if not isinstance(poll_result, AsyncPollResult):
                _, error = poll_result
                error = error or "轮询请求失败"
                if not super()._is_retryable_error(error):
                    return None, self._post_submit_error(ASYNC_POLL_ERROR, error)
                logger.debug(
                    f"{self._get_log_prefix(request.task_id)} 异步轮询暂时失败: "
                    + format_cn_log_fields(
                        远端任务=safe_log_text(remote_task_id, 120),
                        错误=safe_log_error_body(error, 200),
                    )
                )
                if not await self._wait_for_next_poll(deadline):
                    return None, self._format_async_timeout_error(remote_task_id)
                continue

            logger.debug(
                f"{self._get_log_prefix(request.task_id)} 异步任务状态: "
                + format_cn_log_fields(
                    远端任务=safe_log_text(remote_task_id, 120),
                    状态=poll_result.status.value,
                )
            )
            if poll_result.status in {
                AsyncRemoteStatus.PENDING,
                AsyncRemoteStatus.RUNNING,
            }:
                if not await self._wait_for_next_poll(deadline):
                    return None, self._format_async_timeout_error(remote_task_id)
                continue

            if poll_result.status == AsyncRemoteStatus.FAILED:
                return None, self._format_remote_failed_error(poll_result.error_message)

            if poll_result.status == AsyncRemoteStatus.UNKNOWN:
                return None, self._post_submit_error(
                    ASYNC_PROTOCOL_ERROR,
                    poll_result.error_message or "响应中未识别任务状态",
                )

            if poll_result.status != AsyncRemoteStatus.SUCCEEDED:
                return None, self._post_submit_error(
                    ASYNC_PROTOCOL_ERROR,
                    "响应中未识别任务状态",
                )

            if not poll_result.image_urls:
                return None, self._post_submit_error(
                    ASYNC_PROTOCOL_ERROR,
                    "成功响应中未找到图片地址",
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, self._format_async_timeout_error(remote_task_id)
            try:
                images, error = await asyncio.wait_for(
                    self._fetch_result_images(request, attempt, poll_result, deadline),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                return None, self._format_async_timeout_error(remote_task_id)
            if images is not None:
                logger.debug(
                    f"{self._get_log_prefix(request.task_id)} 异步任务完成: "
                    + format_cn_log_fields(
                        远端任务=safe_log_text(remote_task_id, 120),
                        图片=f"{len(images)}张",
                    )
                )
                return images, None
            return None, self._post_submit_error(
                ASYNC_DOWNLOAD_ERROR,
                error or "未能下载生成图片",
            )

    @abc.abstractmethod
    async def _submit_task(
        self,
        request: GenerationRequest,
        attempt: AsyncAttemptContext,
    ) -> AsyncSubmitResult | tuple[None, str]:
        """Submit one provider-side generation task.

        Args:
            request: Image generation request to submit.
            attempt: Immutable context for this remote attempt.

        Returns:
            A remote task id or an error message. Ambiguous submission failures
            must start with ``ASYNC_AMBIGUOUS_SUBMIT_ERROR``.
        """

    @abc.abstractmethod
    async def _poll_task(
        self,
        request: GenerationRequest,
        attempt: AsyncAttemptContext,
        remote_task_id: str,
    ) -> AsyncPollResult | tuple[None, str]:
        """Fetch and normalize one provider-side task status response.

        Args:
            request: Image generation request associated with the task.
            attempt: Immutable context for this remote attempt.
            remote_task_id: Provider-side task identifier.

        Returns:
            A normalized poll result or a retry-classifiable request error.
        """

    async def _fetch_result_images(
        self,
        request: GenerationRequest,
        _attempt: AsyncAttemptContext,
        poll: AsyncPollResult,
        deadline: float,
    ) -> tuple[list[bytes] | None, str | None]:
        """Download successful result URLs before the asynchronous deadline.

        Args:
            request: Image generation request associated with the task.
            _attempt: Immutable context for this remote attempt.
            poll: Successful normalized poll response.
            deadline: Monotonic deadline for all polling and downloading.

        Returns:
            Downloaded images or an error message.
        """
        images: list[bytes] = []
        for url in poll.image_urls:
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                return None, "成功响应包含无效图片地址"

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, "下载结果超过等待预算"
                try:
                    image, error, retryable = await asyncio.wait_for(
                        self._download_result_image(request, url),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    return None, "下载结果超过等待预算"

                if image is not None:
                    images.append(image)
                    break
                if not retryable:
                    return None, error or "下载图片失败"
                logger.debug(
                    f"{self._get_log_prefix(request.task_id)} 异步结果下载暂时失败: "
                    + format_cn_log_fields(
                        地址=safe_log_url(url),
                        错误=safe_log_error_body(error or "下载图片失败", 200),
                    )
                )
                if not await self._wait_for_next_poll(deadline):
                    return None, "下载结果超过等待预算"

        return images or None, None if images else "成功响应中未下载到图片"

    async def _download_result_image(
        self,
        request: GenerationRequest,
        url: str,
    ) -> tuple[bytes | None, str | None, bool]:
        """Download one provider result image without forwarding provider credentials.

        Args:
            request: Image generation request associated with the result.
            url: Signed or public result image URL.

        Returns:
            Image bytes, error text, and whether the download may be retried.
        """
        start_time = time.monotonic()
        try:
            async with self._get_session().get(
                url,
                proxy=self.proxy,
                timeout=self._get_download_timeout(),
            ) as response:
                duration = time.monotonic() - start_time
                if response.status == 200:
                    return await response.read(), None, False

                error_text = await response.text()
                retryable = response.status >= 500 or response.status in {
                    408,
                    409,
                    425,
                    429,
                }
                logger.debug(
                    f"{self._get_log_prefix(request.task_id)} 异步结果下载状态: "
                    + format_cn_log_fields(
                        地址=safe_log_url(url),
                        状态码=response.status,
                        耗时=format_seconds(duration),
                    )
                )
                return (
                    None,
                    f"下载图像错误 ({response.status}): {error_text}",
                    retryable,
                )
        except Exception as exc:  # noqa: BLE001
            return None, safe_log_error_body(exc), True

    def _get_async_poll_interval(self) -> float:
        """Return a validated interval between provider status requests."""
        return float(
            self._coerce_async_positive_int(
                "async_poll_interval",
                DEFAULT_ASYNC_POLL_INTERVAL,
                minimum=1,
            )
        )

    def _get_async_poll_timeout(self) -> float:
        """Return a validated total wait budget for one remote task."""
        return float(
            self._coerce_async_positive_int(
                "async_poll_timeout",
                DEFAULT_ASYNC_POLL_TIMEOUT,
                minimum=1,
            )
        )

    def _coerce_async_positive_int(
        self,
        key: str,
        default: int,
        *,
        minimum: int,
    ) -> int:
        """Read a positive integer adapter setting with a safe fallback.

        Args:
            key: Adapter-specific configuration key.
            default: Fallback value when the setting is invalid.
            minimum: Smallest permitted value.

        Returns:
            Parsed configuration value or the fallback.
        """
        value = self.config.extra.get(key, default)
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= minimum else default

    async def _wait_for_next_poll(self, deadline: float) -> bool:
        """Sleep until the next poll without exceeding the total wait deadline.

        Args:
            deadline: Monotonic deadline for the current remote task.

        Returns:
            Whether time remains after sleeping.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(self._get_async_poll_interval(), remaining))
        return deadline - time.monotonic() > 0

    def _format_async_timeout_error(self, remote_task_id: str) -> str:
        """Format a non-retryable timeout message for an accepted task."""
        return self._post_submit_error(
            ASYNC_TIMEOUT_ERROR,
            f"远端任务 {safe_log_text(remote_task_id, 120)} 未在等待时间内完成",
        )

    def _format_remote_failed_error(self, detail: str | None) -> str:
        """Format a non-retryable provider-side failure message.

        Args:
            detail: Optional provider error detail.

        Returns:
            A sanitized error message for the caller.
        """
        if self.show_user_error_details and detail:
            safe_detail = safe_user_error_detail(detail, 600)
            if safe_detail:
                return f"{ASYNC_REMOTE_FAILED_ERROR}: {safe_detail}"
        return ASYNC_REMOTE_FAILED_ERROR

    def _post_submit_error(self, category: str, detail: str) -> str:
        """Mark an error as non-retryable after a remote task was accepted."""
        if not self.show_user_error_details:
            return category
        safe_detail = safe_user_error_detail(detail, 600)
        return f"{category}: {safe_detail}" if safe_detail else category

    def _is_retryable_error(self, error: str) -> bool:
        """Prevent the outer retry loop from resubmitting accepted remote tasks."""
        if error.startswith(ASYNC_TERMINAL_ERROR_PREFIXES):
            return False
        return super()._is_retryable_error(error)


__all__ = (
    "ASYNC_AMBIGUOUS_SUBMIT_ERROR",
    "AsyncAttemptContext",
    "AsyncPollResult",
    "AsyncPollingImageAdapter",
    "AsyncRemoteStatus",
    "AsyncSubmitResult",
)
