# Triton Ascend Kernels

## 🔥最新消息

- [2025/11] Triton Ascend Kernels项目首次创建。

## 🚀简介

该仓库提供了基于[Triton-Ascend](https://gitcode.com/Ascend/triton-ascend)的高性能triton算子实现。

## 🔍目录结构

```
triton-ascend-kernels/
│
├── benchmark/                     # 性能测试用例目录
├── docs/                          # 文档目录
├── src/
│   └── triton_ascend_kernels/     # 主体源代码目录
├── tests/                         # 功能测试用例目录
├── build.sh                       # 构建脚本
├── CONTRBUTING.md                 # 贡献指南
├── LICENSE                        # 许可证
├── README.md                      # 文档入口
├── SECURITY.md                    # 安全声明
└── pyproject.toml                 # python包配置文件
```

## 💡版本说明

请参考发行版页面获取版本相关信息。

## ⚙️环境部署

```bash
# Install from source
git clone https://gitcode.com/Ascend/triton-ascend-kernels.git
cd triton-ascend-kernels
pip install -e .
```

## 📝功能介绍

本项目提供亲和昇腾的高性能Triton算子实现，详情请参阅 [算子列表](docs/kernel_list.md)

## 👍贡献代码

欢迎参与Triton Ascend Kernels的开发及代码贡献，详情请参阅 [贡献指南](CONTRIBUTING.md)

## 🪧安全声明

请参考[安全声明](SECURITY.md)

## 🚩免责声明

### 致Triton Ascend Kernels使用者

1. Triton Ascend Kernels提供的所有内容仅供您用于非商业目的。
2. 对于Triton Ascend Kernels测试用例以及示例文件中所涉及的数据集，平台仅用于功能测试，华为不提供任何模型权重和数据集，如您使用这些数据进行训练，请您特别注意应遵守对应模型和数据集的License，如您因使用这些模型和数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用Triton Ascend Kernels过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitcode提交issue，我们将及时审视并解决。
4. Triton Ascend Kernels功能依赖的第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，Triton Ascend Kernels仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。

### 致数据所有者

如果您不希望您的算子或数据集在Triton Ascend Kernels中被提及，或希望更新Triton Ascend Kernels中有关的描述，请在Gitcode提交issue，我们将根据您的issue要求删除或更新您相关描述。衷心感谢您对Triton Ascend Kernels的理解和贡献。

## 📃证书

本项目基于 [MIT](LICENSE)

## 🤝建议与交流

欢迎大家为社区贡献。如有任何疑问或建议，请在Gitcode提交[issue](https://gitcode.com/Ascend/triton-ascend-kernels/issues)，我们将及时审视并解决。感谢您的支持。
