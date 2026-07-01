# InferX

多后端推理引擎管理工具，支持 llama.cpp、vLLM、SGLang、TGI、Ollama、TensorRT-LLM、LMDeploy、OpenVINO 八种推理后端。

## 功能特性

- **多后端支持** - 统一管理 8 种推理引擎
- **Web UI** - 浏览器端仪表盘
- **多实例管理** - 同时运行多个模型
- **模型下载** - 支持 HuggingFace、ModelScope、URL 下载
- **实时监控** - GPU/CPU/内存监控
- **告警系统** - 资源阈值告警
- **使用统计** - 请求计数、延迟统计
- **审计日志** - 操作历史记录

## 快速开始

### 安装

```bash
pip install -e .
```

### 运行

```bash
python -m inferx.main --port 8999
```

访问 http://localhost:8999

### Docker

```bash
docker build -t inferx .
docker run -p 8999:8999 inferx
```

## 项目结构

```
inferx/
├── src/inferx/
│   ├── __init__.py
│   ├── main.py           # 入口
│   ├── config.py          # 配置管理
│   ├── models.py          # 数据模型
│   ├── manager.py         # 实例管理
│   ├── backends/          # 后端实现
│   │   ├── base.py
│   │   ├── llamacpp.py
│   │   ├── vllm.py
│   │   └── ...
│   ├── monitoring.py      # 监控告警
│   ├── downloader.py      # 模型下载
│   ├── monitor.py         # 资源监控
│   └── router.py          # API 路由
├── static/
│   └── index.html
├── tests/
│   ├── test_models.py
│   ├── test_backends.py
│   └── test_api.py
├── config.yaml
├── requirements.txt
└── README.md
```

## API 文档

启动服务后访问 http://localhost:8999/docs 查看 Swagger 文档。

### 主要端点

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/system/info` | 系统信息 |
| GET | `/api/system/backends` | 后端列表 |
| GET | `/api/instances` | 实例列表 |
| POST | `/api/instances` | 启动实例 |
| GET | `/api/alerts` | 告警列表 |
| GET | `/api/stats/overview` | 使用统计 |

## 测试

```bash
pip install -r requirements-dev.txt
pytest tests/
```

## 配置

编辑 `config.yaml` 或通过 Web UI 配置页面修改：

```yaml
config:
  model_dir: ~/models
  default_backend: llamacpp
  port_range_start: 8080
  port_range_end: 8180
```

## 许可证

MIT License - 详见 [LICENSE](LICENSE)
