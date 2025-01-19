

# Criação de música interactiva através da deteção de movimento
  

Este projeto permite que os utilizadores criem música de forma interactiva, utilizando tecnologia de deteção de movimento, reconhecimento facial e reconhecimento de gestos das mãos. Os utilizadores podem executar gestos em linguagem gestual americana (ASL) para produzir notas musicais e controlar vários aspectos do processo de criação musical.

  

## Pré-requisitos

  

### Requisitos de Hardware

- Câmara Web (integrada ou externa)

- Computador com capacidade de processamento suficiente para análise de vídeo em tempo real

  

### Requisitos de  Software

- Python 3.8.10

- Visual Studio Code (IDE recomendado)

- Blender

- Git - Para clonagem de repositórios

  
## Instalação do projeto passo-a-passo
Nesta secção, vamos explorar as etapas necessárias para colocar o projeto em funcionamento. Antes de executar o código, é necessário seguir as etapas mencionadas neste capítulo, de modo a obter todos os requisitos operacionais para executar todo o código no projeto. 


### 1º Passo - Instalação do VSCode (Opcional, mas recomendado)
Para ter a melhor experiência possível com este projeto, é altamente recomendável a instalação do IDE VSCode. É também mais fácil clonar o repositório do projeto através do VSCode.

Para o fazer, recomendamos que siga o tutorial abaixo indicado:

<a href="https://www.youtube.com/watch?v=D2cwvpJSBX4" target="_blank"><img src="https://i.ytimg.com/vi/D2cwvpJSBX4/maxresdefault.jpg" 
alt="IMAGE ALT TEXT HERE" width="490" height="276" border="10" /></a>


### 2º Passo - Criação do Ambiente Virtual
Depois de ter o VSCode instalado, é necessário criar um ambiente virtual para o projeto e recomendamos a utilização da versão 3.8.10 do *Python*; o projeto foi construído utilizando a versão mencionada e por essa razão, temos a certeza que o projeto corre nessa versão.
Para mais informações sobre ambientes virtuais, consulte o link abaixo.
 - (https://code.visualstudio.com/docs/python/environments)


### 3º Passo - Clonagem do Repositório do Projeto e Extensões
Depois de criar um ambiente virtual, pode agora clonar o repositório do projeto. Para isso, é necessário ter o *"Git ”* instalado. Para informações sobre a instalação do *"Git ”* [consulte este link](https://git-scm.com/).

Também recomendamos a instalação das seguintes extensões do VSCode:

 - Code Runner - Permite-lhe executar todo o tipo de código numa grande variedade de linguagens de programação.
 - Python

Agora, para clonar o repositório, abra o VSCode, pressione **Ctrl + Shift + “P ‘** no seu teclado e insira o comando ’**Git: Clone**”; Pressione **Enter** e cole o link do repositório do projeto: “**https://github.com/Quywed/tp_final_CV.git**”

Agora já tem o repositório do projeto clonado! 
**NOTA: Para ter a certeza que está no diretório correto, insira no terminal o comando “cd TPFINAL/tp_final ”**.


### 4º Passo - Instalar as Bibliotecas Python
Está quase pronto para colocar o projeto a funcionar! Mas antes disso, é preciso instalar as seguintes bibliotecas, usando o comando **pip install nome_do_pacote** no terminal:

- threading - Utilizar threads de computação no *Python*

- logging - Desativar os logs de terminal do **Ultralytics Object Detection Model, YOLOv11** 

- pickle - Computar o ficheiro pickle de ASL

- cv2 - Acesso à webcam e manipulação de imagens

- mediapipe - Solução visual da Google de deteção de mãos e faces, entre outros tipos de deteção

- numpy - Biblioteca Python para uma fácil manipulação de arrays. Muito utilizada em projectos do campo de STEM

- pygame - Biblioteca open-source para o desenvolvimento de multimédia. Também utilizada para manipulação de áudio

- ultralytics - Empresa que criou o YOLOv11

- time - Manipulação do tempo, como a capacidade de criar atrasos na lógica do código

- socket - Comunicação com o Blender

- json - Comunicação com o Blender

- queue - Comunicação com o Blender

- subprocess - Comunicação com o Blender

- os - Manipulação de diretórios, utilizada para aceder aos ficheiros de som.

- random - Utilizado para a seleção aleatória de uma música.

- tkinter - Biblioteca nativa do Pyrhon para desenvolvimento de GUI's
- 
 ### 5º Passo - Instalação do Blender
Para que o código interaja com o Blender, primeiro é necessário ter o Blender instalado. Recomendamos seguir o tutorial abaixo:

<a href="https://youtu.be/43K1mgIBvOI" target="_blank"><img src="https://i.ytimg.com/vi/43K1mgIBvOI/maxresdefault.jpg" 
alt="IMAGE ALT TEXT HERE" width="490" height="276" border="10" /></a>


## Como executar o programa

Em primeiro lugar, deve começar por certificar que tem o ficheiro blender a correr. Infelizmente, não conseguimos criar um ficheiro executável, por isso tem de seguir estes passos:

1. Abrir o ficheiro .blend “audio-visualizer”.
2. Espere que o Blender abra
3. Clique no separador “Scripting” localizado na parte superior da interface do Blender.
4. Execute o Script, premindo **ALT + P** no teclado.
5. Reproduzir a Animação, premindo **Barra de Espaços** no teclado.

Por esta altura, já deve ter a parte do Blender do projeto a funcionar. Se quiser renderizar as animações de formas diferentes, pode fazê-lo alterando o **Viewport Shading**, premindo Z no teclado e selecionando o modo que deseja com o rato. 

6. Iniciar o programa através do VSCode, ou escrevendo o seguinte comando no terminal:

```bash

python  main.py

```
Para fechar o programa Python, pressione **CTRL + C** no teclado.

 ## Controlos Básicos:

- Mostrar gestos ASL para as letras A, B, C, D e I para tocar diferentes notas musicais

- Mostrar a letra U para fazer parar a reprodução de música de fundo

- Inclinar a face para cima/baixo para mudar um octavo do piano

- Mostrar duas mãos em simultâneo para iniciar/parar o metrónomo

- Passe qualquer um dos seus dedos indicadores sobre o ponto de interrogação localizado no canto superior direito para abrir o menu de ajuda. Deve aparecer uma imagem com mais informações sobre as funcionalidades do projeto.
  

## Seleção de instrumentos e selecções de música

Por defeito, o piano é o instrumento selecionado. Para mudar de instrumento, escreva **'obj'** no terminal e prima **Enter** para ativar a deteção de objectos YOLO. Apresentar os seguintes objectos para mudar de instrumento:

| Objeto a mostrar| Instrumento ou funcionalidade|
| ----------- | ----------- |
| Garrafa| Bongo|
| Telefone celular | Piano|
| Vaso de plantas | Bateria|
| Copo | Música aleatória|
| Mochila | Música de fundo (Menu)|


Para parar a deteção de objectos, escreva **'stop'** e prima **Enter**.

  

### Notas de piano predefinidas
| Letra ASL| Nota Musical |
| ----------- | ----------- |
| A | Do |
| B | Re |
| C| Mi|
| D | Fa |
| I | Sol|  

## Funcionalidades

- Reconhecimento de gestos ASL para criação de notas musicais

- Reconhecimento de inclinação de face para controlo de octavos

- Deteção de objectos para seleção de instrumentos

- Seleção de música de fundo

- Metrónomo incorporado (100 BPM)

- Menu de ajuda interativo

- Reprodução de áudio em tempo real

## Análise do projeto

Para obter uma análise aprofundada do projeto, recomendamos que consulte o relatório "**relatorioCV.pdf**" deste mesmo projeto, localizado no mesmo diretorio que o programa principal.

## Estrutra do Projeto

  

```

TPFINAL/

	tp_final/

	├── bongo/ # Ficheiro de sons do Bongo

	├── drums/ # Ficheiro de sons da Bateria

	├── metronome/ # Ficheiro de som do metrónomo
	
	├── piano/ # Ficheiros de sons do piano

	├── custom_music/ # Ficheiros das músicas de Fundo

	├── main.py # Programa Principal

	├── audio_visualizer.blend # Ficheiro Blender

	├── model.p # Ficheiro pickle de reconhecimento de ASL

	├── img_utils / # Imagens de apoio ao projeto

	├── object_models /  # YOLOv11

	├── relatorioCV.docx # Ficheiro WORD do relatorio

	├── relatorioCV.pdf # Ficheiro PDF do relatorio

``` 


## Limitações

- O programa suporta apenas instrumentos tocados com uma mão

- Transições rápidas de gestos podem ocasionalmente acionar notas não intencionais

- Projeto limitado a 5 letras ASL para garantir a precisão do reconhecimento de gestos

- Possível processo tedioso para fazer com que o ficheiro .blend funcione com o código principal
  

## Créditos

- Deteção de ASL adaptada do repositório [sign-language-detetor-flask-python](https://github.com/SohamPrajapati/sign-language-detector-flask-python) do user SohamPrajapati's.

- Amostras de som de Freesound.org

- Deteção de objectos com o YOLOv11 da Ultralytics.

- Reconhecimento de face e mão com o MediaPipe do Google.
  

## Licenças

Livre para ser utilizado por qualquer pessoa, independentemente do contexto. Realizado para a disciplina de COMPUTAÇÃO VISUAL, do curso de ENGENHARIA DE SISTEMAS E TECNOLOGIAS INFORMÁTICAS do INSTITUTO DE ENGENHARIA da UALG.

## Desenvolvedores

PROJETO DESENVOLVIDO POR PEDRO GONÇALVES e LUCAS MARTINS:

- Pedro Gonçalves (a79297@ualg.pt): https://github.com/PedroDanielG
- Lucas Martins (a79294@ualg.pt): https://github.com/Quywed
