# HIPI
HIPI - Hidden Picture Game Generator by Shape Matching and Style Transfer.

HIPI - 실사 숨은그림찾기를 Shape matching과 Style Transfer 모델을 활용하여 제작하는 서비스입니다.

## Project Summary
기존 숨은그림찾기는 일반적으로 흑백 형태, 혹은 몇 가지 색상이 입혀져있는 형태 기반의 숨은그림찾기입니다. 
이는 사람이 그리는 그림책 형태로 많이 소비되며, 선이 자연스럽게 숨기는 대상의 형태를 만들기에 **형태 기반 숨은그림찾기**라고 명명할 수 있습니다. 
이를 자동으로 만드는 방법은 [Yoon et al.][1]에서 볼 수 있습니다.

**질감 기반 숨은그림찾기**는 실사와 같은 형태로 연출하며, 보통 예술가가 제작합니다. 
예술가는 자연물에 대상을 숨기기도 하고, 작게 여러 대상을 숨기기도 합니다. 
숨기는 방식은 그림 기법과 연출에 따라 다양합니다.

이를 자동으로 만드는 논문은 [Chu et al.][2]과 [Du et al.][3] 등 다양합니다.
이들은 질감 기반 숨은그림찾기를 위하여 사진의 명암을 기반으로 숨길 물체의 장소를 선정하였습니다.
또한, 여러 물체를 숨기기에 시간이 오래 걸립니다.

따라서 이에 착안하여 **실사 기반의 숨은그림찾기를 만들되, 적절한 숨길 위치를 찾아준다면 질감 기반으로 물체를 숨기는 방법을 Style Transfer 모델로 처리하면 좋을 것이라고 생각했습니다.**
따라서 **형태 기반으로 숨길 이미지의 위치를 찾아내고, 그 상태로 스타일 전이를 하는 방법과 프레임워크를 제안합니다.**


## Framework
그림 1. 프레임워크 다이어그램
![hipi_service_diagram](https://github.com/user-attachments/assets/7171179e-e623-4594-aef5-7fc69a3958e8)


## Main Content
### 1. Preprocess
이미지 합성을 위하여 **대상 사진에 배경을 제거해줘야 합니다.** 저는 RMBG-2.0 모델[Briaai][4]을 사용했습니다.
그 이후, 배경 사진과 대상 사진에 **다양한 필터를 처리**하여 특징점을 더욱 잘 추출할 수 있도록 만들어야 합니다.

### 2. Feature Point Extraction
[1]에서 사용된 방법을 참고하되, **내부 파라미터의 변화와 특징점 추출 방법, NMS 처리** 등이 추가 및 보완되었습니다.
특징점 추출은 ORB[Rublee et al.][5]를 활용하였으며, 대상 사진의 경우 **윤곽 특징점과 내부 특징점을 따로 합쳐 N개를 추출하는 방식**으로 개선하였습니다.
**배경 사진은 대상 사진 이상의 M개의 특징점을 추출합니다. 또한, 유사도 계산을 위하여 N-Nearest Neighbor를 사용합니다.**
공통적으로 **PCA를 활용하여 shape context에 대해 회전과 크기 변화를 정규화해줍니다.** 
이후 shape context vector를 만드는데, [Belongie et al.][6]을 참고하여 **5개의 거리별 구역과 12개의 각도별 구역을 추출하여 60차원의 벡터**를 만듭니다. 
이후, **대상 사진의 shape context vector는 Milvus DB**에 저장합니다.
    
### 3. Shape Matching:
배경 사진의 N-nearest Neighbor를 대상 사진의 shape context를 검색할 때, **Vector DB의 Index를 활용하여 Bulk Index Search를 사용할 수 있습니다.** 이 과정을 **1차 검색**이라 합니다.
이후, 더욱 정교한 검색을 위하여 **12개의 각도별 구역을 참고하여 Nearest Neighbor 그룹과 가장 가까운 12개의 그룹에서의 유사도 점수의 평균**을 냅니다. 이 과정을 **2차 검색**이라 합니다.

그림 2. 2차 검색 식
![image](https://github.com/user-attachments/assets/46e38031-bf87-4192-aaaa-9f5e8cb24fd1)

결과적으로 TPS 보간법을 활용하여 나온 유사도 Heatmap은 아래와 같습니다.

그림 3. Heatmap
![image](https://github.com/user-attachments/assets/333bc947-16cc-41de-bee8-2c4f37c39867)

### 4. Overlay Problem:
배경 사진의 N-nearest neighbor 기준으로 유사한 대상 사진이 대응되어 있기에 겹침 문제를 해결하고자 **Grid NMS**를 사용했습니다. 특징점 반지름의 2배를 그리드로 간주하여 겹치지 않도록 합니다.

### 5. Combine Image:
이미지 합성을 할 때, PCA에서 회전한 각도와 크기 변화를 고려하여 합성해야 합니다.

### 6. Style Transfer:
StyleID라는 모델[Chung et al.][7]을 활용하여 배경을 Style, 합성된 이미지를 Content로 스타일 전이를 진행합니다. 감마값과 T값을 적절히 조절하여 스타일 전이를 진행합니다.

## Code Instruction
...
## Demo
최종적으로 나온 사진은 아래와 같습니다.

그림 4. 배경 사진(왼쪽)과 사과와 조랑말이 숨겨진 숨은그림찾기(오른쪽).
![image](https://github.com/user-attachments/assets/f9dec08f-6630-481d-a437-267e283e0b10)

그림 5. 배경 사진(왼쪽)과 사과와 노트북이 숨겨진 숨은그림찾기(오른쪽)
![image](https://github.com/user-attachments/assets/c0ad9798-647a-47ef-8d0f-de1d7dc01ad9)


그림 6. 그림 사막에서 사과와 공책이 숨겨져 있는 숨은그림찾기
<br/>
![image](https://github.com/user-attachments/assets/35fed767-0233-46c8-905b-a04a64578c99)

## Conclusion and Future Work
실사 기반 숨은그림찾기 제작방법을 형태 기반 숨은그림찾기를 제작하는 방법과 스타일 전이 모델을 활용하여 제시했습니다. 
또한 이를 구현하기 위한 프레임워크를 벡터 DB를 포함하여 제안하였습니다.

추가로 고려해야 하는 부분은 다양한 종류의 하이퍼파라메터입니다. 
사용자가 입력하는 배경 사진은 숨길만 한 디테일이 존재하지 않을 수도 있고, 너무 많을 수도 있습니다. 
특징점이 잘 나오지 않을 가능성 또한 존재합니다. 
**그림 6과 같이 명암 분포가 균일한 편이라면 너무나도 어렵게 생성될 수 있다고 생각하기에 명암 분포를 고려할 필요성 또한 있어 보입니다.**
Style Transfer 모델 또한 **조금 더 지역적인 변화에 민감하도록 더 낮은 latent space 차원에서의 변경도 필요할 것으로 보입니다.**
또한, **Shape Matching 과정이 실질적으로 더 좋은 숨은그림찾기에 필요한 과정인지에 대한 검증 또한 미흡합니다.** 
처리 속도, 정확도 및 성공률, 만족도 등과 같은 성능 평가 지표나 정량적 분석이 부족하기에 추후 연구에 있어 보완할 것입니다.


## References

[1] Jong-Chul Yoon, In-Kwon Lee and Henry Kang, "A Hidden-picture Puzzles Generator," Computer Graphics Forum, vol. 27, no. 7, pp. 1869-1877, 2008.

[2] H.-K. Chu, W.-H. Hsu, N. J. Mitra, D. Cohen-Or, T.-T. Wong, and T.-Y. Lee, "Camouflage images," ACM Transactions on Graphics, vol. 29, no. 4, Article 51, pp. 1-8, Jul. 2010

[3] H. Du and L. Shu, "Structure Importance-Aware Hidden Images," in 2019 IEEE 11th International Conference on Advanced Infocomm Technology (ICAIT), 2019, pp. 36-42.

[4] RMBG-2.0, "Background removal model," Hugging Face, 2024. [Online]. Available: https://huggingface.co/briaai/RMBG-2.0

[5] E. Rublee, V. Rabaud, K. Konolige and G. Bradski, "ORB: An efficient alternative to SIFT or SURF," in 2011 International Conference on Computer Vision, 2011, pp. 2564-2571.

[6] S. Belongie, J. Malik and J. Puzicha, "Shape matching and object recognition using shape contexts," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2002, vol. 24, no. 4, pp. 509-522.

[7] Jiwoo Chung, Sangeek Hyun, Jae-Pil Heo, "Style injection in diffusion: A training-free approach for adapting large-scale text-to-image models," Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 8795-8805
