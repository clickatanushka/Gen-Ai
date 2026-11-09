#!/usr/bin/env bash
# Render one module deck to a contact sheet: preview.sh NN OUTDIR
set -e
NN=$1; OUT=$2
DECK=$(ls "$(dirname "$0")"/../03_Modules/Module_${NN}_*/4_Slides_Module${NN}.pptx)
mkdir -p "$OUT/m$NN"
soffice -env:UserInstallation=file://$OUT/lo_$NN --headless --convert-to pdf --outdir "$OUT/m$NN" "$DECK" >/dev/null 2>&1
pdftoppm -r 40 -png "$OUT/m$NN/4_Slides_Module${NN}.pdf" "$OUT/m$NN/s"
python3 - "$OUT/m$NN" <<'E'
import glob, sys
from PIL import Image
d = sys.argv[1]
ims = [Image.open(f) for f in sorted(glob.glob(d + "/s-*.png"))]
w, h = ims[0].size; cols = 4; rows = (len(ims) + cols - 1) // cols
c = Image.new("RGB", (w * cols, h * rows), "white")
for i, im in enumerate(ims):
    c.paste(im, ((i % cols) * w, (i // cols) * h))
c.save(d + "/sheet.png")
print(d + "/sheet.png", len(ims), "slides")
E
