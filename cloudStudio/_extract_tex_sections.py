import os
path = r"D:\TanHuangWork\Menopausal_women_sleep\Paper\els-cas-templates\main.tex"
outdir = r"D:\TanHuangWork\Menopausal_women_sleep\cloudStudio"

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract multiple sections and write them
sections = {
    'tex_l370_l395.txt': (369, 395),    # SOTA table
    'tex_l393_l405.txt': (392, 405),    # SOTA analysis paragraph
    'tex_l408_l420.txt': (407, 420),    # Ablation table
    'tex_l425_l430.txt': (424, 430),    # Ablation analysis paragraph
}

for fname, (start, end) in sections.items():
    with open(os.path.join(outdir, fname), 'w', encoding='utf-8') as f:
        for i in range(start, min(end, len(lines))):
            f.write(f"{i+1}: {lines[i]}")
    print(f"Wrote {fname} ({end-start} lines)")

print("Done")
