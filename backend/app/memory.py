# Handling memory for consecutive queries to LLM

from typing import Dict, List, Literal


Role = Literal["user", "assistant", "system"]


class MemoryStore:


    def __init__(self, max_turns: int = 10, max_chars: int = 4000):
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self.max_turns = max_turns
        self.max_chars = max_chars

    def get(self, session_id: str) -> List[Dict[str, str]]:
        return self._store.get(session_id, [])

    def append(self, session_id: str, role: Role, content: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = []

        self._store[session_id].append({"role": role, "content": content})

        if len(self._store[session_id]) > self.max_turns:
            self._store[session_id] = self._store[session_id][-self.max_turns :]

        while sum(len(m["content"]) for m in self._store[session_id]) > self.max_chars:
            if self._store[session_id]:
                self._store[session_id].pop(0)
            else:
                break
