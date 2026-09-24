# hf-remote-downloader

一个面向 Hugging Face 模型、数据集和 Space 仓库的下载器，支持：

- HTTP Range 断点续传、重试和 SHA-256/Git blob 校验
- 文件锁和 JSON 状态，避免重复下载
- 按路径分片，以及多主机进度汇总
- Linux `systemd --user`、macOS `launchd` 后台任务
- 远程部署时由目标主机直接下载，当前 SSH 会话不承载文件流量

## 安装

```bash
python -m venv .venv
.venv/bin/python -m pip install .
```

Windows 可将上面的 `python` 和 `.venv/bin/...` 替换为 `py` 和
`.venv\Scripts\...`。如果需要 SOCKS 代理：

```bash
python -m pip install '.[socks]'
```

## 下载

直接下载不需要预先填写任何代理地址：

```bash
hf-download-repo https://huggingface.co/owner/repo ./data/repo
```

程序默认遵循 `requests` 的标准网络环境变量：
`HTTPS_PROXY`、`HTTP_PROXY`、`ALL_PROXY` 和 `NO_PROXY`。也可以显式指定
代理；重复 `--proxy` 可提供故障转移列表，显式配置会优先于环境变量：

```bash
hf-download-repo https://huggingface.co/owner/repo ./data/repo \
  --proxy https://proxy.example:8443 \
  --proxy https://backup-proxy.example:8443
```

需要完全绕过代理时使用 `--no-proxy`。旧版本使用的 `HF_PROXY` 和
`HF_PROXIES` 仍可作为兼容配置，但新部署建议使用标准环境变量。

私有仓库可通过 `HF_TOKEN` 提供 token，不要把 token 写进命令行或提交到
仓库：

```bash
HF_TOKEN=... hf-download-repo https://huggingface.co/owner/private-repo ./data
```

也可以使用兼容的镜像或 API 地址：

```bash
hf-download-repo https://huggingface.co/owner/repo ./data \
  --endpoint https://hf.example.org
```

## 远程主机

```bash
hf-download-repo https://huggingface.co/owner/repo /data/repo \
  --host gpu-server
```

代理配置以**实际执行下载的主机**为准：控制机的代理环境不会隐式复制到
远程主机。远程主机可以自行设置 `HTTPS_PROXY` 等标准变量，或者显式传入
`--proxy`。这样不会把控制机上的 `127.0.0.1` 之类地址误带到另一台机器。

首次使用远程后台任务前，Linux 通常需要：

```bash
loginctl enable-linger "$USER"
```

`--background` 会安装并启动后台任务；`--install-service` 只安装不启动。
Linux 使用 `systemd --user`，macOS 使用 `launchd`。任务配置和日志位于
`~/.local/share/hf-remote-downloader/`。

## 历史版本、分片和进度

下载历史提交中的 `.pth` 文件：

```bash
hf-download-repo https://huggingface.co/owner/repo ./data \
  --history --shard 0/2
```

多台机器可使用不同的 `--shard N/M`；也可设置 `HF_SHARD_INDEX` 和
`HF_SHARD_COUNT`。进度查询：

```bash
hf-download-progress https://huggingface.co/owner/repo ./data --interval 10
hf-download-progress https://huggingface.co/owner/repo /data/repo \
  --host gpu-server --once
```

当只提供目标目录时，进度工具会读取目录中的 catalog/state：

```bash
hf-download-repo --status ./data
```

## 重新分配双机任务

先停止所有 worker，再用实际测得的速度生成不可变计划。速度参数单位是
MiB/s，没有针对某台机器的默认值：

```bash
python -m hf_downloader.rebalance \
  --manifest ./manifest.json \
  --local-dest ./local-data \
  --remote-host gpu-server \
  --remote-dest /data/repo \
  --speeds 20 15 \
  --output ./rebalance-plan
```

计划会复用已校验文件、保留最大的断点前缀，并只为未开始的文件重新分配
worker。执行计划中的 `local.json`、`remote.json` 时请使用对应的 manifest
和 shard，避免重复下载。

## 校验与恢复

- LFS 文件按 SHA-256 `oid` 校验，普通 Git 文件按 Git blob SHA-1 校验；没有
  可信 hash 时不会仅凭文件大小判定成功。
- 临时文件使用 `.uploading`，校验失败的内容会保留为 `.corrupt` 便于诊断。
- state 文件记录进度、失败次数和最近错误；进度工具支持多目标汇总、速度
  平滑和 ETA。
- 下载目录中的 state、catalog、锁和临时文件不应提交到 Git。

## 开发检查

```bash
.venv/bin/python -m compileall hf_downloader
.venv/bin/python -m unittest discover
```
