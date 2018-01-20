# An instruction:
# 1) Start this script
# 2) Open test.bmp in an image editor
# 3) Modify the image and see results of the classification immediately

(while [[ 1 == 1 ]]; do echo test.bmp ; sleep 2; done) | ./cclass -cpu run net
