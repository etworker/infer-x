"""Signal-tracing wrapper for vllm api_server startup debugging."""
import os
import signal
import sys
import time


def _handler(signum, frame):
    sig_name = signal.Signals(signum).name
    import traceback
    with open("/tmp/vllm_signal_trace.log", "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] PID={os.getpid()} received {sig_name}({signum})\n")
        traceback.print_stack(frame, file=f)
        f.write("---\n")
        f.flush()
    # Forward to default handler
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2):
    signal.signal(s, _handler)

with open("/tmp/vllm_signal_trace.log", "a") as f:
    f.write(f"[{time.strftime('%H:%M:%S')}] PID={os.getpid()} wrapper loaded, handlers installed\n")
    f.flush()

# Now exec into the real vllm api_server
sys.argv = sys.argv[1:]  # remove wrapper from argv
from vllm.entrypoints.openai.api_server import *
