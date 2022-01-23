### 创建环境

    conda create -n slu python=3.6
    source activate slu
    pip install torch==1.7.1
	pip install transformers

### 运行
    
在根目录下运行`python3 slu_bert.py`进行训练

在根目录下运行`test.py`将对`data/test_unlabelled.json`进行预测，并将结果保存在根目录下`test_result.json`

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成
        
        python scripts/slu_baseline.py --<arg> <value>
    其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU
+ `utils/vocab.py`:构建编码输入输出的词表
+ `utils/example.py`:读取训练数据
+ `utils/test_example.py`:读取待预测的无标签数据
+ `utils/batch.py`:输入BERT前对数据预处理，并以批为单位转化
+ `model/slu_bert_tagging.py`:改进后模型
+ `scripts/slu_bert.py`:主程序脚本
+ `test.py`:对无标签数据集进行预测

