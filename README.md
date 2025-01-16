# HIPI

*Read this in other languages: [한국어](docs/README_ko.md)*

HIPI - Hidden Picture Game Generator by Shape Matching and Style Transfer.

## Project Summary
Traditional hidden picture puzzles are typically based on black and white or limited color line drawings. These are commonly consumed in the form of picture books, where lines naturally create shapes to hide target objects. This can be called a **shape-based hidden picture puzzle**. An automated method for creating these can be found in [Yoon et al.][1].

**Texture-based hidden picture puzzles** are created in a photorealistic style, usually made by artists. Artists may hide objects within natural elements or conceal multiple small objects. The hiding methods vary depending on artistic techniques and presentation.

Several papers describe automated methods for creating these, including [Chu et al.][2] and [Du et al.][3]. These papers selected hiding locations based on the image's contrast for texture-based hidden picture puzzles. However, hiding multiple objects takes considerable time.

Therefore, **I proposed creating photorealistic hidden picture puzzles where, once appropriate hiding locations are found, the texture-based object concealment could be handled using a Style Transfer model.** Specifically, **I propose a framework and method that finds object locations using shape-based matching and applies style transfer to that state.**


## Framework
Figure 1. Framework Diagram
![hipi_service_diagram](https://github.com/user-attachments/assets/7171179e-e623-4594-aef5-7fc69a3958e8)


## Main Content
### 1. Preprocess
For image composition, **the target photo's background must be removed.** I used the RMBG-2.0 model[Briaai][4]. After that, **various filters are applied** to both background and target photos to enhance feature point extraction.

### 2. Feature Point Extraction
While referencing the method used in [1], **internal parameter modifications, feature point extraction methods, and NMS processing** have been added and enhanced. Feature point extraction utilizes ORB[Rublee et al.][5], and for target images, the method was improved to **extract N points by separately combining contour feature points and internal feature points.**
**For background images, M feature points (more than the target image) are extracted. Additionally, N-Nearest Neighbor is used for similarity calculations.**
Commonly, **PCA is used to normalize rotation and scale changes for shape context.**
When creating shape context vectors, referring to [Belongie et al.][6], **a 60-dimensional vector is created by extracting 5 distance-based zones and 12 angle-based zones.**
Subsequently, **the target image's shape context vectors are stored in Milvus DB.**
    
### 3. Shape Matching:
When searching for target image shape contexts using N-nearest neighbors from the background image, **Bulk Index Search can be utilized using the Vector DB's Index.** This process is called primary search.
For more precise searching, **the average similarity score is calculated from the 12 groups closest to the Nearest Neighbor group, referencing the 12 angle-based zones.** This process is called **secondary search.**

Figure 2. Formula for secondary search

![image](https://github.com/user-attachments/assets/46e38031-bf87-4192-aaaa-9f5e8cb24fd1)

Consequently, using TPS interpolation, the similarity heatmap is as follows.

Figure 3. Heatmap

![image](https://github.com/user-attachments/assets/333bc947-16cc-41de-bee8-2c4f37c39867)

### 4. Overlay Problem:
Since similar target images are matched based on the background image's N-nearest neighbors, **Grid NMS** was used to solve the overlap problem. The grid is considered as twice the radius of feature points to prevent overlapping.

### 5. Combine Image:
When compositing images, the rotation angle and scale changes from PCA must be considered.

### 6. Style Transfer:
Using the StyleID model[Chung et al.][7], style transfer is performed with the background as Style and the composited image as Content. Gamma and T values are appropriately adjusted during the style transfer process.

## Code Instruction
...
## Demo

Figure 4. background image(left) and hidden picture puzzle with apple and zebra(right)
![image](https://github.com/user-attachments/assets/f9dec08f-6630-481d-a437-267e283e0b10)

Figure 5. background image(left) and hidden picture puzzle with apple and notebook(right)
![image](https://github.com/user-attachments/assets/c0ad9798-647a-47ef-8d0f-de1d7dc01ad9)


Figure 6. Hidden Picture puzzle with apple and notebook on a dessert background
<br/>
![image](https://github.com/user-attachments/assets/35fed767-0233-46c8-905b-a04a64578c99)

## Conclusion and Future Work
I presented a method for creating photorealistic hidden picture puzzles by combining shape-based hidden picture creation methods with style transfer models. I also proposed a framework incorporating vector DB for implementation.

Additional considerations include various hyperparameters. Background images provided by users may lack suitable details for hiding objects or have too many. There's also a possibility that feature points may not be well extracted. **As seen in Figure 6, when the contrast distribution is relatively uniform, the generation can become too difficult, suggesting the need to consider contrast distribution.** 

The Style Transfer model also **appears to need modifications in lower latent space dimensions to be more sensitive to local changes.** Furthermore, **there is insufficient verification whether the Shape Matching process is truly necessary for better hidden picture puzzles.** 

Future research will address the lack of performance evaluation metrics and quantitative analysis such as processing speed, accuracy and success rates, and satisfaction levels.


## References

[1] Jong-Chul Yoon, In-Kwon Lee and Henry Kang, "A Hidden-picture Puzzles Generator," Computer Graphics Forum, vol. 27, no. 7, pp. 1869-1877, 2008.

[2] H.-K. Chu, W.-H. Hsu, N. J. Mitra, D. Cohen-Or, T.-T. Wong, and T.-Y. Lee, "Camouflage images," ACM Transactions on Graphics, vol. 29, no. 4, Article 51, pp. 1-8, Jul. 2010

[3] H. Du and L. Shu, "Structure Importance-Aware Hidden Images," in 2019 IEEE 11th International Conference on Advanced Infocomm Technology (ICAIT), 2019, pp. 36-42.

[4] RMBG-2.0, "Background removal model," Hugging Face, 2024. [Online]. Available: https://huggingface.co/briaai/RMBG-2.0

[5] E. Rublee, V. Rabaud, K. Konolige and G. Bradski, "ORB: An efficient alternative to SIFT or SURF," in 2011 International Conference on Computer Vision, 2011, pp. 2564-2571.

[6] S. Belongie, J. Malik and J. Puzicha, "Shape matching and object recognition using shape contexts," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2002, vol. 24, no. 4, pp. 509-522.

[7] Jiwoo Chung, Sangeek Hyun, Jae-Pil Heo, "Style injection in diffusion: A training-free approach for adapting large-scale text-to-image models," Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 8795-8805
