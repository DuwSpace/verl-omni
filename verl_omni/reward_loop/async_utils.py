# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Async helpers for reward work that must finish before cleanup."""

import asyncio


async def gather_complete(awaitables, *, return_exceptions: bool = False):
    """Drain every awaitable before propagating cancellation or failure."""
    aggregate = asyncio.gather(*awaitables, return_exceptions=True)
    cancellation = None
    while True:
        try:
            results = await asyncio.shield(aggregate)
            break
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
    if cancellation is not None:
        raise cancellation
    if not return_exceptions:
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise errors[0]
    return results


__all__ = ["gather_complete"]
