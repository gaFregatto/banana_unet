#SETUP

- Instalação do pyenv https://github.com/pyenv/pyenv
- Instalação do pyenv-virtualenv https://github.com/pyenv/pyenv-virtualenv
- pyenv install 3.8.6
- pyenv virtualenv 3.8.6 {envname}
- pyenv activate {envname}
- pip install -r requirements.txt

É importante garantir que as versões do CUDA e cudNN sejam compatíveis com a versão do tensorflow-gpu instalada (2.4), e o driver da placa de vídeo. Eu utilizei as seguintes versões:
- CUDA 11.0
- cudNN 8.0.5