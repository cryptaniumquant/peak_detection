import json
import os
from datetime import datetime, timezone
from typing import Optional
import config


class StateStore:
    def __init__(self, filename: str = 'bot_state.json'):
        self.path = os.path.join(config.STATE_DIR, filename)
        self._state = {
            'strategies': {},  # strategy -> { last_notified_ts: iso }
            'simulation_now': None,
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self._state.update(json.load(f))
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    def get_last_notified(self, strategy: str) -> Optional[datetime]:
        s = self._state.get('strategies', {}).get(strategy)
        if s and s.get('last_notified_ts'):
            try:
                return datetime.fromisoformat(s['last_notified_ts'])
            except Exception:
                return None
        return None

    def set_last_notified(self, strategy: str, ts: datetime):
        self._state.setdefault('strategies', {})
        self._state['strategies'].setdefault(strategy, {})
        # ensure timezone-aware ISO
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self._state['strategies'][strategy]['last_notified_ts'] = ts.isoformat()
        self._save()

    def get_simulation_now(self) -> Optional[datetime]:
        v = self._state.get('simulation_now')
        if v:
            try:
                return datetime.fromisoformat(v)
            except Exception:
                return None
        return None

    def set_simulation_now(self, ts: datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self._state['simulation_now'] = ts.isoformat()
        self._save()
