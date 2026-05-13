import sys
filepath = r"D:\TanHuangWork\Menopausal_women_sleep\Paper\els-cas-templates\main.tex"
outpath = r"D:\TanHuangWork\Menopausal_women_sleep\cloudStudio\_tex_sections.txt"
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(outpath, 'w', encoding='utf-8') as out:
    sections = [
        ("3.1 Algorithm Performance", 206, 290),
        ("3.2 Sleep Architecture", 289, 360),
        ("3.3 Ablation Study", 359, 445),
    ]
    for name, start, end in sections:
        out.write(f"\n{'='*60}\n=== {name} (lines {start+1}-{end}) ===\n{'='*60}\n")
        for i in range(start, min(end, len(lines))):
            out.write(f"{i+1}: {lines[i]}")

print("Done - wrote _tex_sections.txt")
