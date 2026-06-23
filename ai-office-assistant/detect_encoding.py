import ftfy

with open("server.py", "r", encoding="utf-8") as f:
    content = f.read()

fixed = ftfy.fix_text(content)

with open("server_fixed.py", "w", encoding="utf-8") as f:
    f.write(fixed)

print("修复完成，请打开 server_fixed.py 查看。")