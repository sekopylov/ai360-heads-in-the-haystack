"""Runtime-local shutdown watchdog: actual deadline, not an activity keep-alive."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import threading
from urllib.request import Request, urlopen


def stop_reason(status, now, deadline):
    if now >= deadline:
        return 'two_hour_deadline'
    if status.get('stage') == 'complete' and status.get('archive') and status.get('archive_sha256'):
        return 'complete_archive_saved'
    if status.get('stage') == 'failed':
        return 'experiment_failed'
    return None


def unassign_runtime():
    # Same runtime-management endpoint used by google.colab.runtime.unassign().
    # Direct POST avoids requiring notebook JS from this detached watchdog process.
    address = os.environ.get('TBE_RUNTIME_ADDR')
    if not address:
        raise RuntimeError('Colab runtime management service is unavailable')
    with urlopen(Request(f'http://{address}/unassign', method='POST'), timeout=10) as response:
        if response.status != 200:
            raise RuntimeError('Runtime unassignment failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--seconds', type=int, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or not os.environ.get('TBE_RUNTIME_ADDR'):
        raise RuntimeError('A positive deadline and Colab runtime service are required')
    start = time.time()
    deadline = start + args.seconds
    payload = dict(started_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(),
                   deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                   max_seconds=args.seconds, armed=True)
    # Drive I/O may hang. Neither status reads nor journal writes may block the deadline.
    def record_snapshot(snapshot):
        try:
            args.record.write_text(json.dumps(snapshot, indent=2))
        except OSError:
            pass
    threading.Thread(target=record_snapshot, args=(dict(payload),), daemon=True).start()
    latest = {}
    def read_status():
        while True:
            try:
                data = json.loads(args.status.read_text()) if args.status.is_file() else {}
                latest['status'] = data
            except (OSError, ValueError):
                pass
            time.sleep(5)
    threading.Thread(target=read_status, daemon=True).start()
    monotonic_deadline = time.monotonic() + args.seconds
    while True:
        reason = stop_reason(latest.get('status', {}), time.monotonic(), monotonic_deadline)
        if reason:
            payload.update(stop_reason=reason, requested_stop_utc=datetime.now(timezone.utc).isoformat())
            threading.Thread(target=record_snapshot, args=(dict(payload),), daemon=True).start()
            try:
                unassign_runtime()
                return
            except Exception as error:
                print(f'Runtime stop failed, retrying: {type(error).__name__}', flush=True)
                time.sleep(5)
                continue
        time.sleep(min(5, max(.1, monotonic_deadline-time.monotonic())))


if __name__ == '__main__':
    main()
