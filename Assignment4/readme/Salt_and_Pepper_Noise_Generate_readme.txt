- Purpose of this code
 This code was witten to randomly scatter salt and pepper noise on input images.

- How to run this code
 Load 'Salt_and_Pepper_Noise_Generate.cpp' in Visual Studio and run this with 'lena.jpg'

- How to adjust parameters (if any)

 Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp);
 const Mat input: Give input image
 float ps: density of salt noise. Give value between 0-1.
 float pp: density of pepper noise. Give value between 0-1.
 
- How to define default parameters
 Mat Add_salt_pepper_Noise(const Mat input, float ps, float pp);
 const Mat input: Give input image.
 float ps: density of salt noise. Give value between 0-1. In this code, set 0.1.
 float pp: density of pepper noise. Give value between 0-1. In this code, set 0.1.
