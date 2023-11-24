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

 What are good values?
   https://stackoverflow.com/questions/25125670/best-value-for-threshold-in-canny
   Looks like good values for image '0_Edinburg..' are around 50 and 300 => values are not bytes

[ ] Slide 11: Dendogram is all the time 1 vs. the rest. Is that 'family tree'
    realistic? What would the LA draw?
