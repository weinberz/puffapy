import re
import sys

b2 = 0
mor = 0
tfr = 0

content = open(sys.argv[1],'r').read()

for arg in sys.argv[2:]:
    p = re.compile("(tfr|b2|mor).*"+arg+".*?\n")
    res = p.search(content)
    if res is not None:
        if res.group(1) == 'b2':
            b2 += 1
        elif res.group(1) == 'mor':
            mor +=1
        elif res.group(1) == 'tfr':
            tfr +=1

print("{} B2 cells, {} MOR cells, {} TfR Cells".format(b2, mor, tfr))
