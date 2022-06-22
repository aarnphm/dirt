import threading

from watchfiles._rust_notify import RustNotify

notifier = RustNotify(["./"], True, True, 50)

stop_event = threading.Event()

changes = notifier.watch(1_600, 0, 0, stop_event)
if changes != set():
    print("hello")
    print(f"Changes: {changes}")
    stop_event.set()
