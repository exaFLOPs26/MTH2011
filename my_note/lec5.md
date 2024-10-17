## Kernel methods
> For non-linear hypothesis by mapping(feature map) attributes to features

- But the features φ(x) is high-dimensional ($$ d^(3)$$)

### Kernel trick (case for OLS)
<img width="157" alt="image" src="https://github.com/user-attachments/assets/45e26d32-2f33-414a-9894-5e5418c8fc23">

> θ can be represented as a linear combination of the vectors φ(x(1)),...,φ(x(n)).

<img width="386" alt="image" src="https://github.com/user-attachments/assets/e2da4728-db7f-4e2d-908c-a1db40fefe0b">  

<img width="575" alt="image" src="https://github.com/user-attachments/assets/13002e2b-1241-415a-9a0e-7224cfc79a80">  

<img width="575" alt="image" src="https://github.com/user-attachments/assets/ff875817-0579-49b8-96cf-452ae63cb2b7">  

> Like we said feature φ(x) is high-dimensional, but kernel is only doing kind of inner product which is way cheaper.  

<img width="541" alt="image" src="https://github.com/user-attachments/assets/7d0b9b14-b0de-4c38-ba7e-8499953eae24">  

> We only need euclidean inner product of <x,z> with O(d) times!

Kernel method의 핵심은 주어진 attribute x를 directly하게 사용하는 것이 아니라 feature map을 통해 feature 즉 변형해서 사용하자
결국 nonlinear한 hypothesis을 만들기 위함
nonlinear한 data를 linear로 변환해서 tractable하게
high dimension data를 kernel을 통해 압축한다-> feature를 deep learning이 다 해주고 있음
사실 feature를 만드는 것은 우리가 자연스럽게 하는 것이다. 무의식적으로도...!
Kernel is high-dimension d^(3)
그래서 최대한 Φ를 직접 구하는 식들은 피하고 싶어! 차라리 부분적인 원소 i와 j 내적 계산이 나을 수도! 왜냐하면 high degree를 바로 계산하는 것이 아니라 euclidian inner product를 하고 제곱 세제곱~
회계계수 theta 다시 말해 parameter들은 new feature들의 linear combination으로 표현가능하다

