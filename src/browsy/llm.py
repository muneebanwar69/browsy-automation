"""
Custom OpenAI LLM subclass that handles screenshots from Playwright.

The stock mcp-agent OpenAIAugmentedLLM converts ImageContent from tool results
into ChatCompletionContentPartImageParam (type="image_url"), which OpenAI rejects
in 'tool' role messages with HTTP 400. This subclass intercepts image content,
stores it separately for display, and replaces it with a text placeholder in
the tool message so the LLM loop can continue.
"""

from __future__ import annotations

import json
import base64
from typing import List, Optional, Tuple

from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
)
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ImageContent,
    EmbeddedResource,
    TextContent,
    TextResourceContents,
)

from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.tracing.telemetry import get_tracer
from mcp_agent.tracing.semconv import GEN_AI_TOOL_CALL_ID, GEN_AI_TOOL_NAME

from opentelemetry import trace


class CapturedScreenshot:
    """Holds a captured screenshot from a Playwright tool call."""

    def __init__(self, data: str, mime_type: str, tool_name: str):
        self.data = data  # base64-encoded image data
        self.mime_type = mime_type  # e.g. "image/png"
        self.tool_name = tool_name  # e.g. "playwright_screenshot"


class BrowsyOpenAILLM(OpenAIAugmentedLLM):
    """
    Extended OpenAI LLM that captures screenshots and strips image data
    from tool-role messages to avoid OpenAI API 400 errors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_screenshots: List[CapturedScreenshot] = []

    async def execute_tool_call(
        self,
        tool_call: ChatCompletionMessageToolCall,
    ) -> ChatCompletionToolMessageParam:
        """
        Execute a tool call, intercept any image content from the result,
        store it for later display, and replace it with a text placeholder
        in the tool message returned to the LLM.
        """
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.{self.name}.execute_tool_call"
        ) as span:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            tool_call_id = tool_call.id
            tool_args = {}

            if self.context.tracing_enabled:
                span.set_attribute(GEN_AI_TOOL_CALL_ID, tool_call_id)
                span.set_attribute(GEN_AI_TOOL_NAME, tool_name)
                span.set_attribute("tool_args", tool_args_str)

            try:
                if tool_args_str:
                    tool_args = json.loads(tool_args_str)
            except json.JSONDecodeError as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                return ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call_id,
                    content=f"Invalid JSON provided in tool call arguments for "
                    f"'{tool_name}'. Failed to load JSON: {str(e)}",
                )

            tool_call_request = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(name=tool_name, arguments=tool_args),
            )

            result = await self.call_tool(
                request=tool_call_request, tool_call_id=tool_call_id
            )

            self._annotate_span_for_call_tool_result(span, result)

            # Process content: separate images from text
            content_parts = []
            for c in result.content:
                if isinstance(c, ImageContent):
                    # Capture the screenshot
                    self.captured_screenshots.append(
                        CapturedScreenshot(
                            data=c.data,
                            mime_type=c.mimeType,
                            tool_name=tool_name,
                        )
                    )
                    # Replace with text placeholder for the LLM
                    content_parts.append(
                        ChatCompletionContentPartTextParam(
                            type="text",
                            text=f"[Screenshot captured successfully - {c.mimeType}]",
                        )
                    )
                elif isinstance(c, EmbeddedResource):
                    if isinstance(c.resource, TextResourceContents):
                        content_parts.append(
                            ChatCompletionContentPartTextParam(
                                type="text", text=c.resource.text
                            )
                        )
                    elif (
                        c.resource.mimeType
                        and c.resource.mimeType.startswith("image/")
                    ):
                        # Capture embedded image resource
                        self.captured_screenshots.append(
                            CapturedScreenshot(
                                data=c.resource.blob,
                                mime_type=c.resource.mimeType,
                                tool_name=tool_name,
                            )
                        )
                        content_parts.append(
                            ChatCompletionContentPartTextParam(
                                type="text",
                                text=f"[Screenshot captured successfully - {c.resource.mimeType}]",
                            )
                        )
                    else:
                        content_parts.append(
                            ChatCompletionContentPartTextParam(
                                type="text",
                                text=f"{c.resource.mimeType}:{c.resource.blob}",
                            )
                        )
                elif isinstance(c, TextContent):
                    content_parts.append(
                        ChatCompletionContentPartTextParam(
                            type="text", text=c.text
                        )
                    )
                else:
                    content_parts.append(
                        ChatCompletionContentPartTextParam(
                            type="text", text=str(c)
                        )
                    )

            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call_id,
                content=content_parts,
            )

    def clear_screenshots(self):
        """Clear captured screenshots (call before each new query)."""
        self.captured_screenshots.clear()

    def get_screenshots_base64(self) -> List[dict]:
        """
        Get all captured screenshots as a list of dicts with base64 data.
        
        Returns:
            List of {"data": str, "mime_type": str, "tool_name": str}
        """
        return [
            {
                "data": s.data,
                "mime_type": s.mime_type,
                "tool_name": s.tool_name,
            }
            for s in self.captured_screenshots
        ]
