[ ] move to escience center template https://github.com/NLeSC/python-template

[ ] images are in JPG format. Are there compressions issues/artifacts?
 bash $ identify -verbose 0_Edinburgh_Nat_Gallery.jpg
  ...
  Compression: JPEG
  Quality: 75
  ...

[ ] get uncompressed (raw) images?

[ ] SIFT is often done in grayscale, to prevent mismatch due to hue differences, there are extensions for using color.
 https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7514695


[ ] Canny edge dection has:
 * 2 times a double; threshold parameters. determined emperically. https://en.wikipedia.org/wiki/Canny_edge_detector
 * integer apertureSize=3
 * boolean flag L2=gradient=false
