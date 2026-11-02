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

"""One replica lease per recaption, retained until all sibling videos finish."""

import asyncio
from collections import defaultdict


class RecaptionLeases:
    """Coalesce concurrent sibling acquisition and release each owner once."""

    def __init__(self, client):
        self.client = client
        self.owners = {}
        self.released = set()
        self.request_groups = {}
        self.remaining = defaultdict(int)

    async def acquire(self, request_id, payload):
        """Resolve all leaves of a recaption to one shared server acquisition."""
        group = payload.get("sampling_params", {}).get("recaption_group_id", request_id)
        self.request_groups[request_id] = group
        self.remaining[group] += 1
        if group not in self.owners:
            self.owners[group] = asyncio.create_task(self.client._acquire_server(group))
        return await self.owners[group]

    def complete(self, request_id):
        """Release the replica after the last sibling completes."""
        group = self.request_groups[request_id]
        self.remaining[group] -= 1
        if self.remaining[group] == 0:
            self._release(group)

    def release_expired(self, request_id):
        """Release an acquisition that completed after the round timed out."""
        self._release(self.request_groups[request_id])

    def release_all(self):
        """Release acquired owners after preparation failure."""
        for group in self.owners:
            self._release(group)

    def _release(self, group):
        owner = self.owners[group]
        if group not in self.released and owner.done() and not owner.cancelled() and owner.exception() is None:
            self.client._release_server(owner.result()[0])
            self.released.add(group)

    def __len__(self):
        return len(self.owners)
