# PSD

Source code for our TOIS paper: Document-Level Relation Extraction with Progressive Self-Distillation

### 1.Description

We provide codes of PSD  instantiation with ATLOP. Folder named **psd_ikd_docred** includes code related to **PSD**  instantiation with **ATLOP** using **ikd loss** (refer to Section 4.1) runing on **DocRED** dataset. **psd_ikd_redocred**, **psd_ikd_cdr** ,**psd_ikd_gda** are the same. You can find their optimal configurations of tunable hyperparameters in Table 2 and experiment results in Table 3 and Table 4.

Besides, we also provide code of extension PSD described in section 4.2 in folder **psd_ikd_rkd_docred**, which includes code related to **PSD**  instantiation with **ATLOP** using **ikd loss** (refer to Section 4.1) and **rkd loss** (refer to Section 4.2) runing on **DocRED** dataset. 

### 2.Usage 

You should first install Python dependency packages.

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Then you can find a script in every folder named train_[dataset].sh, run it to train and eval the model. You need to make adjustments to some parameters of the script, like dataset path and some tunable hyperparameters.

```
sh train_[dataset].sh
```

### 3.License

This project is licensed under the MIT License.

### 4.Citation

If you use this work or code, please kindly cite the following paper:

```
@article{wang2024document,
  title={Document-Level Relation Extraction with Progressive Self-Distillation},
  author={Wang, Quan and Mao, Zhendong and Gao, Jie and Zhang, Yongdong},
  journal={ACM Transactions on Information Systems},
  year={2024},
  publisher={ACM New York, NY}
}
```

### 5.Concats

If you have any questions for this code, please feel free to contact [Gao Jie](mailto:879356763@qq.com), we will reply as soon as possible.