import sys
import re

infer_file = sys.argv[1]
output_file = sys.argv[2]

with open(infer_file) as f:
    text = f.read()

try:
    forward = float(re.search(r'\+\+,--":\s*([0-9.]+)', text).group(1))
    reverse = float(re.search(r'\+\-,\-\+":\s*([0-9.]+)', text).group(1))
except:
    forward = reverse = 0.0

if forward > 0.8:
    stranded = "1"
elif reverse > 0.8:
    stranded = "2"
else:
    stranded = "0"

with open(output_file, "w") as out:
    out.write(stranded + "\n")
