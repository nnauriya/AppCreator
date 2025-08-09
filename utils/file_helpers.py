import datetime

def write_text_log(text, filename="app.log"):
    """
    Append a text log entry to a specified log file with timestamp.

    Args:
        text (str): Text to log.
        filename (str): Log file name (path relative to project root).
    """
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {text}\n\n")
