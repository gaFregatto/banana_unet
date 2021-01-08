## SETUP
- Instalação do pyenv https://github.com/pyenv/pyenv
- Instalação do pyenv-virtualenv https://github.com/pyenv/pyenv-virtualenv
- pyenv install 3.8.6
- pyenv virtualenv 3.8.6 {envname}
- pyenv activate {envname}
- pip install -r requirements.txt

É importante garantir que as versões do CUDA e cudNN sejam compatíveis com o tensorflow-gpu (2.4) e com o driver da placa de vídeo. As versões utilizadas foram:
- CUDA 11.0
- cudNN 8.0.5

### Para executar:
Execute o arquivo main.py presente na pasta src com o ambiente virtual ativado.
```./banana_unet/src$ python main.py```
