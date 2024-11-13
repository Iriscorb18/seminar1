1) For the first exercise, the image is attached as a "Captura_ffmpeg_exercise1.JPG".

2) We have tried to use the conversion from the lecture notes but for some reason the conversion from 
YUV to RGB does not return the same value as the input from RGB to YUV. Then we have used the 
conversion computation shown in wikipedia.

3) We have used the command ffmpeg -y -i "input_image" -vf scale={width}:{height} "output_image" which we have seen from internet
We use the command "-y" to overwrite and to avoid generating new images for every output.

4) We have searched for a code that could read in a "serpentine" mode any given image. 
In this case we use a color image, but it could be possible to use a black and white.

5) We use a command similar to the one used in exercise 3 but this case instead of changing the scale we change the format to grey.

6) For this exercise we have also searched through the internet a code that does a run-length encoding and we have tried it with
a random array of numbers.

7) To use the DCT we take advantatge of the library scipy.fftpack where we can import both the DCT function and IDCT function
We can see that the IDCT conversion from the DCT image does not return exactly the same color, we suppose its due the image being also
jpeg and not a raw image, where some information could be lost.

8) For this exercise we also take advantage of another library but in this case pywt which has wavelet coding, we need to
understand the plots of "Approximation" (cA), "Horizontal Detail" (cH), "Vertical Detail" (cV) and "Diagonal Detail" (cD).













PD: "Y recuerda, nunca dejes de soñar, Jorge". Att: Maria Patiño

