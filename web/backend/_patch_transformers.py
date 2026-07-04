path = r"C:\Users\kongze\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\transformers\modeling_utils.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

old = (
    '            with safe_open(resolved_archive_file, framework="pt") as f:\n'
    "                metadata = f.metadata()\n"
    '\n'
    '            if metadata.get("format") == "pt":'
)
new = (
    '            with safe_open(resolved_archive_file, framework="pt") as f:\n'
    "                metadata = f.metadata() or {}\n"
    '\n'
    '            if metadata.get("format") == "pt":'
)

count = src.count(old)
print("matches:", count)
if count == 1:
    src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("patched OK")
else:
    print("NO CHANGE - already patched or pattern mismatch")