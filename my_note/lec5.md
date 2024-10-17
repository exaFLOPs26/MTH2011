## Kernel methods
> For non-linear hypothesis by mapping(feature map) attributes to features

- But the features φ(x) is high-dimensional ($$ d^(3)$$)

### Kernel trick (case for OLS)
<img width="157" alt="image" src="https://github.com/user-attachments/assets/45e26d32-2f33-414a-9894-5e5418c8fc23">

> θ can be represented as a linear combination of the vectors φ(x(1)),...,φ(x(n)).

<img width="386" alt="image" src="https://github.com/user-attachments/assets/e2da4728-db7f-4e2d-908c-a1db40fefe0b">  

<img width="575" alt="image" src="https://github.com/user-attachments/assets/13002e2b-1241-415a-9a0e-7224cfc79a80">  

<img width="575" alt="image" src="https://github.com/user-attachments/assets/ff875817-0579-49b8-96cf-452ae63cb2b7">  

> Like we said feature φ(x) is high-dimensional, but kernel is only doing sorts of inner product which is way cheaper.  

<img width="541" alt="image" src="https://github.com/user-attachments/assets/7d0b9b14-b0de-4c38-ba7e-8499953eae24">  

> We only need euclidean inner product of <x,z> with just O(d) times!

<img width="467" alt="image" src="https://github.com/user-attachments/assets/ba4c8372-aa5c-47ca-8084-6867c837a57c">

> Also in calculating y for new x, only kernel is needed!

### Okay, we got that kernel is great. Mapping of φ(x) to K is great. But if there a mapping of K to φ(x)? Meaning that only knowing K without what φ(x) is!

<img width="569" alt="image" src="https://github.com/user-attachments/assets/62c55a67-8d94-4f36-814b-58ec47bfcfdb">

<img width="577" alt="image" src="https://github.com/user-attachments/assets/945a0cf0-4032-40f9-9a3d-c16eba9a4721">

> If K is a valid kernel (i.e., if it corresponds to
some feature mapping φ), then the corresponding kernel matrix K ∈ R (n×n)
is symmetric positive semidefinite  

<img width="577" alt="image" src="https://github.com/user-attachments/assets/29588794-2155-4f87-b121-0ff93d31e319">

> iff relation satisfies!  

- Deep learning learns the kernel  
- nonlinear한 data를 linear로 변환해서 tractable하게
- high dimension data를 kernel을 통해 압축한다

