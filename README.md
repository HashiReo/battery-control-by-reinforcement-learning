## これはなに？
強化学習による蓄電池の最適な充放電を実現するためのコードです。  

## Gettering Started
### Installation / 導入
VSCodeのremote connectionを使ってdocker fileを読み込んでください。  

### Usage / 実行
[main.py](/Battery-Control-By-Reinforcement-Learning/main.py)を実行してください。  



## Others / その他
### Directroy / ディレクトリ構成
ディレクトリ構成は以下の２つを参考にしています：
1. The Hitchhiker's Guide to Python    
https://python-guideja.readthedocs.io/ja/latest/writing/structure.html    
2. Cookiecutter: Better Project Templates  
https://cookiecutter.readthedocs.io/en/latest/index.html  


Project Organization / プロジェクトの構成  
------------  
    ├── .devcontainer      <- docker関連ファイル
    │
    ├── .gihub             <- gothub関連ファイル
    │      └── workflows   <- pullreqされると走るもの
    │
    ├── Battery-Control-By-Reinforcement-Learning   <- Source code for use in this project.
    │       ├── __init__.py    <- Makes source codes a Python module
    │       │── main.py
    │       └── paramaters.py   <- 機械学習の調整パラメータを記述
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── tests              <- テストコード
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    

### Information for the code / コードの図解

プログラム構成
![image](https://github.com/Takuya510634/Battery-Control-by-Reinforcement-Learning-1/assets/105347514/d9158e4d-da82-469f-afc9-2c56ad89a311)

ファイル名対応表
![image](https://github.com/Takuya510634/Battery-Control-by-Reinforcement-Learning-1/assets/105347514/973445c6-0a90-44ee-b8ce-6ee51c32daae)
