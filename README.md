# ollama-cliproxyapi

CLIProxyAPI（CPA）原生 Go 插件：把 **Ollama**（云端 ollama.com 或本地 `localhost:11434`）接入 CPA，
统一暴露三种客户端协议（OpenAI Chat Completions / Anthropic Messages / OpenAI Responses），
并提供基于 Ollama 官方账户接口（`/api/balance`、`/api/usage`）的余额与用量查询页面。

- 插件 ID：`ollama-cliproxyapi`（由 `.so` 文件名决定，请勿改名）
- Provider：`ollama`
- 语言/构建：Go 1.26+，`-buildmode=c-shared`，产物为 `plugins/linux/amd64/ollama-cliproxyapi.so`

> 本插件以 [jerryhu1234/commandcode-go-cliproxyapi](https://github.com/jerryhu1234/commandcode-go-cliproxyapi)
> （MIT）为架构模板移植改造；协议适配内核（三种格式的请求/流式转换）与其保持同源，
> Ollama 化的差异集中在：模型目录（`/v1/models`）、默认路由、认证头、额度/用量 API。

## 功能

- **模型代理**：客户端可用三种格式调用同一个 Ollama 模型：
  - `POST /v1/chat/completions`（OpenAI Chat）
  - `POST /v1/messages`（Anthropic Messages）
  - `POST /v1/responses`(OpenAI Responses）
  客户端格式与上游端点自动映射，同一模型也能用其它格式调用（见下文"路由规则"）。
- **模型目录**：定时从 `GET {base-url}/v1/models` 拉取模型清单，公开名默认加前缀 `ollama/`
  （如 `ollama/gpt-oss:20b`）。云端目录的 `<model>:cloud` 模型照常透传。
- **余额/用量页**：管理后台 `/v0/resource/plugins/ollama-cliproxyapi/quota` 资源页 +
  `POST /v0/management/plugins/ollama-cliproxyapi/quota-usage` API：
  - `GET https://ollama.com/api/balance`（Bearer key）→ 新版 USD 余额
    （`included.balance_usd / allowance_usd / period`、`purchased.balance_usd`），
    兼容旧版计划（`included.session / weekly` 的 `remaining_percent / resets_at`）。
  - `GET https://ollama.com/api/usage?range=24h&scope=self` → 请求数、用量 USD、
    输入/缓存输入/输出 token 合计。
  - 注意上游限制：账户接口 **10 req/min/user**（429 + Retry-After），页面已做缓存与节流；
    均衡卡片在本地模式（localhost）自动禁用。
- **不支持的调用**：`count_tokens`、插件级 `http_request` 返回明确的 unsupported 错误。

## 路由规则

1. 所有模型**默认路由 `chat/completions`**（不按模型名猜测协议）。
2. `route-overrides` 可把模型（按上游 ID，如 `claude-sonnet-4`）钉到 `messages` 或 `responses`。
3. 客户端来源格式转换：Claude 格式请求打到默认模型时，会被翻译成 OpenAI Chat 格式发给 Ollama；
   只有被钉到 `messages` 的模型/原生公告 `supported_endpoints` 含 `/v1/responses` 的模型
   才会按对应上游端点直发。

## 构建

```bash
# 需要 Go 1.26+ 与 gcc（CGO）
go build -trimpath -buildmode=c-shared -o plugins/linux/amd64/ollama-cliproxyapi.so .
```

产物：`plugins/linux/amd64/ollama-cliproxyapi.so`（与生成的 `.h` 一同被 `.gitignore` 忽略）。

## 安装

把 `ollama-cliproxyapi.so` 与 CPA 的其他插件放在一起（CPA 插件目录），在 CPA `config.yml` 中：

1. 插件注册块（`plugins` 列表项或 store 形状，按你的 CPA 版本）：

```yaml
plugins:
  - id: ollama-cliproxyapi
    name: Ollama
    author: jerryhu1234
    tags: [Provider, ollama, usage, Streaming]
```

2. 插件配置块（`plugins.configs.ollama-cliproxyapi`，YAML 按 base64 直接挂到
   `plugins.configs` 的 store 形状时由 CPA 注入）：

### 云端（ollama.com）

```yaml
base-url: https://ollama.com
model-prefix:
  enabled: true
  value: ollama
api-keys:
  - value: ${OLLAMA_API_KEY}   # 支持环境变量展开
catalog:
  refresh-interval: 15m
  stale-while-unavailable: true
protocols:
  chat-completions: true
  messages: true
  responses: true
request-timeout: 5m
stream-first-data-timeout: 60s
stream-idle-timeout: 3m
stream-total-timeout: 30m
stream-finish-grace: 15s
max-response-bytes: 67108864   # 64 MiB
allow-http: false
route-overrides:
  claude-sonnet-4:
    protocol: messages
    endpoint: /v1/messages
```

### 本地（localhost:11434）

```yaml
base-url: http://localhost:11434
allow-http: true               # 本地必须开
api-keys:
  - value: ollama              # 本地服务器不校验，但字段必填（占位符即可）
catalog:
  refresh-interval: 15m
  stale-while-unavailable: true
protocols:
  chat-completions: true
  messages: true
  responses: true
```

本地模式下：
- 余额/用量接口自动禁用（本地服务器没有 `/api/balance`、`/api/usage`，卡片会说明原因）。
- 云模型 ID 带 `:cloud` 后缀（如 `qwen3-coder:480b-cloud`），由 Ollama 自行路由到云端并计费。

## 配置项说明

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `base-url` | `https://ollama.com` | 上游权威地址；本地为 `http://localhost:11434` |
| `catalog-url` | `{base-url}/v1/models` | 模型目录地址（OpenAI 兼容格式） |
| `model-prefix.enabled/value` | `true` / `ollama` | 公开模型名前缀，`ollama/xxx` |
| `api-keys[].value` | — | 至少一个；支持 `${ENV_VAR}` 展开；本地可用占位符 `ollama` |
| `catalog.refresh-interval` | `15m` | 目录刷新周期（最小 1m） |
| `catalog.stale-while-unavailable` | `true` | 刷新失败时继续用旧目录 |
| `protocols.*` | 三项均 `true` | 三种上游端点的总开关（钉到禁用协议的模型会被排除） |
| `route-overrides.<id>.protocol/endpoint` | — | 把某个上游模型钉到指定协议与端点 |
| `request-timeout` | `5m` | 非流式整请求超时 |
| `stream-first-data-timeout` | `60s` | 流式首字节超时 |
| `stream-idle-timeout` | `3m` | 流式空闲超时 |
| `stream-total-timeout` | `30m` | 流式总时长超时 |
| `stream-finish-grace` | `15s` | 终止帧（finish/done）宽限 |
| `max-response-bytes` | 64 MiB | 响应体上限（目录抓取下限 16 MiB） |
| `allow-http` | `false` | 允许 http 上游（本地模式必须） |

## 错误分类

上游错误按 CPA 分类返回（便于宿主做故障转移/冷却）：
`auth_failure`、`invalid_model`、`unsupported_protocol_or_parameter`、`rate_limit`（含 429 启发式，
额度上游对 Retry-After 透传）、`quota_exhaustion`、`billing`、`upstream_server_failure`、
`timeout_or_network_failure`、`translation_failure`。

## 与模板（commandcode 插件）的差异

- 目录：`GET /v1/models`（OpenAI 兼容形状 `{"data":[...]}`，容忍裸数组），
  默认路由 `chat/completions`。
- Messages 认证：仅 `Authorization: Bearer` + `anthropic-version: 2023-06-01`
  （**不移交** `x-api-key`——Ollama 云端不接受）。
- 移除了模板的会话派生/`x-commandcode-session` 头逻辑（Ollama 无此概念）。
- 额度：`/api/balance` + `/api/usage`（模板是 `/alpha/billing/*` + `/alpha/whoami`）；
  无邮箱同步（Ollama 没有 whoami 概念），卡片标签用 key 哈希前 12 位。
- QuotaProvider：`SupportsReset=false`（Ollama 不支持重置）。

## License

MIT（继承模板许可证）。