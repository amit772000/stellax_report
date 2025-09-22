import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys

class RunOnSaveHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("generate_report.py"):
            print("🔄 File changed! Running script...")
            subprocess.run([sys.executable, "generate_report.py"])

if __name__ == "__main__":
    event_handler = RunOnSaveHandler()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    print("👀 Watching for changes in generate_report.py...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
