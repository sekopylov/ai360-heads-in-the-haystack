"""Recover local CLI metadata for an explicitly selected existing runtime.

Run with the google-colab-cli tool's Python. Never allocates or deletes VMs
and never prints tokens. Refuses to overwrite an existing local session name.
"""
import argparse

from colab_cli.common import state
from colab_cli.state import SessionState

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--name', required=True)
parser.add_argument('--endpoint', required=True)
args = parser.parse_args()
if state.store.get(args.name) is not None:
    raise RuntimeError('Session name already registered; use colab status instead')
matches = [a for a in state.client.list_assignments() if a.endpoint == args.endpoint]
if len(matches) != 1:
    raise RuntimeError('Selected runtime is not present exactly once on the server')
assignment = matches[0]
info = assignment.runtime_proxy_info
state.store.add(SessionState(
    name=args.name, endpoint=assignment.endpoint,
    token=info.token, url=info.url, token_expires_at=info.expires_at(),
    variant=assignment.variant.name, accelerator=assignment.accelerator.value,
    machine_shape=assignment.machine_shape.name,
))
print(f'Reconnected {args.name}: {assignment.accelerator.value}')
