# Usage Prediction(用量预测)设计方案

> 状态:**提案(待评审)**。基于现有 ollama-cliproxyapi quota 插件扩展,
> 对标 CPA Manager 凭证页(codex 双窗口:5h 主窗口 / weekly 次窗口)的展示形态,
> 但补充其缺少的能力:**基于历史趋势的前瞻预测**。

## 0. 需求映射

| # | 需求 | 本方案落点 |
|---|------|-----------|
| 1 | 每个账号,每个窗口的额度**预计何时用完** | 账号卡:窗口级 ETA + 倒计时 + 置信度 + 48h 迷你趋势图 |
| 2 | 每个大类型(ollama / commandcode / …)的**总额度使用预测** | 类型级聚合卡:总剩余、总速率、车队 ETA、首个账号耗尽点、48h 容量阶梯曲线 |

## 1. 与 CPA Manager 凭证页的差异 → 为什么必须加"采样"

Manager 凭证页是**当下快照**(剩余 %、重置时间),没有时间维度。要预测
"什么时候用完",必须有**剩余量随时间变化的序列**。可用数据源两条:

- **A. 上游账户 API 巡检采样(Phase 1,权威)**
  - `GET /api/balance` → included.balance_usd / allowance_usd / period.until,
    legacy included.session / weekly(remaining_percent + resets_at)。
  - `GET /api/usage?range=24h|7d` → totals + **分桶序列(官方文档已证实,
    见 D5)**:24h = 25 个每小时桶,7d/30d = 每日桶;每桶含 request_count /
    usage_usd / token 计数,当前桶 `partial: true`。文档明示
    "New usage may take time to appear" ⇒ 最近桶系统性欠计:分桶只取
    日内形状,速率定标一律以余额增量为权威(见 §4)。
  - legacy 计划(session/weekly)分桶**只给请求数,不含 USD/token**
    (官方文档)⇒ percent 窗口只能靠余额采样拟合。
  - 硬约束:账户接口 **10 req/min/user**(429 + Retry-After)。
- **B. executor 本地遥测(Phase 2,精修)**
  - executorexecute / execute_stream 完成时本地记账 (ts, key 摘要, model,
    in/out tokens) —— 只记 token/计时元数据,不落 prompt/响应内容。
  - 用两次 usage 采样的 ΔUSD/Δtokens 推出隐含单价,把 token 速率换算成 $/h,
    弥补采样间隔造成的粒度损失,并度量突发性(burstiness)。

**原则:预测查询只读本地 store,绝不实时打上游**(除显式"立即刷新"按钮,
其同样走节流)。这保证预测页随时可打开,不怕 10 req/min 限制。

## 2. 架构(本仓库新增 `internal/predictor`)

```
internal/predictor/
  sample.go     — Sample 数据模型 + 采样器(后台循环、节流、退避)
  store.go      — 内存环形序列(默认保留 7d)+ 可选 NDJSON 落盘
  model.go      — 烧速率估计(短/长窗双斜率) + ETA + 置信度
  aggregate.go  — 类型级聚合(fleet ETA / first-depletion / 容量曲线)
  management.go — 管理 API 路由 + 资源页装配
  *_test.go
```

- 生命周期:复用 catalog refresh loop 的既有模式(`stop/done` + `lifeMu`,
  reconfigure 重建 goroutine),**不引入新的并发原语**;shutdown 时落盘一次。
- 注册:`registerManagement` 增加路由 `POST /plugins/<plugin>/prediction`
  与资源页 `/prediction`(Menu: Ollama Usage Prediction)。
  capabilities 无需变更(management_api 已开启)。

## 3. 采样设计(约束下的节流)

- 身份:沿用 `quotaIdentity`(key sha256 摘要 → authRecordID),与 executor、
  credential 标签保持同一稳定 ID;**store 中绝不落 key 明文**。
- 节奏:每账号默认每 `sample-interval`(10m)采 1 次 balance;usage 合并为
  同次 best-effort;相邻账号错峰(相位分散),间隔 ±20% 抖动。
- 全局令牌桶:默认预算 8 req/min(留余量给手动刷新),429 → 按 Retry-After
  退避,并把该账号样本标记 `stale`。上游限额为**每用户 10 req/min 且同账号
  多 key 共享**(官方文档;上游甚至推荐 1/min 轮询)⇒ 全局预算同时防住
  "同一账号多个 key 各自采样"的叠加,10m 采样仅用上限的 ~2%。
- **临耗尽加速采样(D3 的 P1 手段)**:正常 10m;任一窗口剩余 <20% 或短期
  烧速 >2× 长期烧速时,该账号间隔缩至 5m(预算内:10 账号全加速 ≈4 req/min)。
  越接近耗尽精度越要紧,此手段不碰 executor 热路径。
- reset 分段:采样携带 ResetsAt;窗口重置处 remaining 跳回 cap / 100%,
  拟合时段必须按 reset 分段,防止斜率被跳变污染。

## 4. 预测模型(Phase 1:双斜率线性)

每窗口样本序列 `(t_i, remaining_i)`(percent 窗口在 percent 空间同构处理):

- `r_short`:最近 τ=4~6h 的 EWMA 线性拟合斜率(响应快);
- `r_long`:自分段起点(或最近 24h)的斜率(防抖);
- `r = max(r_short, r_long)`(保守 = 更快烧完)。**主口径取保守**:
  预测用于排任务决策,按"最坏近期速率"不会超卖;`r_long` 单独保留为
  乐观参考口径。
- `ETA_conservative = remaining / r`,`ETA_optimistic = remaining / max(r_long, ε)`;
  UI 主数字 = 保守 ETA,副注一行"按近 24h 均速 ≈ X"(给区间,不给假精度)。
- `r <= ε` → 细分两态:**`stable`**(有过消耗记录、当前零烧速)/
  **`no-data`**(从未见消耗,只报"暂无消耗数据",不宣称稳定);
- 与重置比较:
  - `reset_at < now + ETA` → 状态 `resets-first`,附"到重置前预计再消耗
    r × (reset_at − now)";
  - 否则 → `exhausts-in <ETA>`(倒计时展示);
  - `remaining <= 0` → `exhausted`,等重置。
- 置信度:分段内样本 <3 或拟合 R² < 0.5 → low;burstiness 高 → 提示
  "速率波动大"。**展示精度按置信度降级(D6)**:`high` → "≈6.5 小时";
  `medium` → "≈4~9 小时"(保守/乐观区间);`low` → "≈2~4 天"量级 + 波动提示。
  不给伪精确分钟。
- **Purchased 余额**:无上限、无重置的衰减余额,作为附加可预测行
  (resets_at 为空,纯倒计时,状态只有 exhausts-in / exhausted / stable)。
- **Shape-aware(D5 已定档,P1 生效)**:usage 24h 分桶给出日内烧速曲线,
  模型取 `rate(t) = profile(hour-of-day) × scale`:`scale` 由余额增量定标
  (余额无时滞警示,是权威信号),`profile` 来自分桶且 partial 桶不入;
  legacy percent 窗口无 USD 桶,退回纯采样拟合。相比恒速外推,
  夜间/白天形态误差显著下降,且不引入新依赖。

### 4.1 误差预算:对"预测准不准"的诚实预期

| 误差源 | 量级 | 方案缓解 |
|--------|------|----------|
| 余量数据本身(included/purchased) | 低 —— 上游权威值,balance 无时滞警示 | — |
| usage 分桶欠计("new usage may take time") | 最近桶系统性偏低 | 分桶仅取形状,定标用余额增量,partial 桶剔除 |
| 突发/时段性负载外推(**主导项**) | 单账号小时尺度 ETA 可差 1.5~2× | shape-aware 日内曲线 + 保守 max 口径 + 区间展示 + 临耗尽 5m 加速 |
| 行为突变(临近时刻启动大任务) | 原理上不可预测 | 保守口径的系统性偏早(宁可提前预警,不误报"还够用") |
| legacy rolling reset 语义近似 | 边界处偏差 | ETA vs resets_at 比较吸收;resets-first 状态优先给结论 |
| 同账号多 key 共享限额 | 采样互相挤占 | 全局 8/min 预算兜底 |

**结论(按场景)**:

- **趋势正确性(会不会耗尽、量级)**:高 —— 余量是上游直接数据;
- **单账号 ETA 单点精度(突发负载)**:小时尺度可能差 1.5~2×;设计目标不是
  精确分钟,而是"足够早的正确预警"+ 把不确定性可见化;
- **车队级/类型级(需求 2)**:多账号独立突发相互平均(σ∝1/√N),N≥4 且
  shape-aware 时,48h 舰队投影误差预期 ±15~20% 量级(经验预期,非保证);
- **稳定负载(夜间/周末/月度 included 窗)**:拟合误差通常 <10~20%;
- **实测验证(P1 验收单测)**:合成时间序列回测(工作日形态/突发/夜间归零)
  证明 shape-aware 相对 flat 拟合的误差下降;D3 触发阈值即主导项的实测裁决。

## 5. 类型级聚合(需求 2)

对类型内全部账号(仅 ollama;跨类型见 §8):

- 总剩余 `Σrem` / 总上限 `Σcap` / 总速率 `Σr`(只计活跃账号);
- **fleet ETA** = `Σrem / Σr`(全部额度耗尽时刻);
- **first-depletion ETA** = `min_i ETA_i`(第一个账号死亡时刻);
- **容量投影曲线**:逐小时 `rem_i(t) = max(0, rem_i − r_i·t)`,
  `t ≥ reset_at` 时跳回 `cap_i`,求和得"可用 USD + 可用账号数"双序列
  阶梯曲线;前端堆叠面积图标注三个点:首个账号耗尽 / 车队容量腰斩 / 全灭。
- **deadline 查询(排任务口径)**:给定未来时点 T(如"明天 20:00"),
  从投影曲线直接读取:届时可用账号数 / 可用 USD / 相对总容量比例,
  回答"这个时点前的大任务排不排得下" —— 比 fleet ETA 更贴近实际决策。
- **风险徽标**:`critical`(已有账号 exhausted 或容量 <10%)/
  `tight`(首个账号 24h 内耗尽,或 horizon 末剩余 <30%)/ `healthy`。
- **stale 聚合规则**:样本滞后超过 3×采样间隔的 stale 账号不计入速率、
  ETA 与投影(旧样本会低估烧速),但计入展示分母,类型卡标 `N/M 新鲜`。

## 6. 管理 API

```text
GET  /v0/resource/plugins/ollama-cliproxyapi/prediction         # 页面
POST /v0/management/plugins/ollama-cliproxyapi/prediction       # 数据
  body {}             → 全量:{accounts:[…], by_type:{ollama:{…}}}
  body {deadline:ts}  → 附加 by_type.ollama.at_deadline 快照
  body {refresh:…}    → 手动刷新指定 key(节流内)
```

响应(账号级,窗口级):

```json
{
  "accounts": [{
    "key_id": "ollama-key-…", "label": "Ollama credential …",
    "updated_at": "…", "stale": false,
    "windows": [{
      "name": "included", "remaining_usd": 12.4, "cap_usd": 30.0,
      "rate_per_hour": 0.9, "eta_seconds": 49600, "resets_at": "…",
      "state": "exhausts-in|resets-first|exhausted|stable",
      "confidence": "low|medium|high",
      "series": [[ts, remaining], …]
    }]
  }],
  "by_type": {
    "ollama": {
      "remaining_usd": 120.5, "cap_usd": 300.0, "rate_per_hour": 8.2,
      "fleet_eta_seconds": 48000, "first_depletion_seconds": 9100,
      "active_accounts": 6, "stale_accounts": 1, "risk": "healthy",
      "projection": [[ts, available_usd, active_accounts], …],
      "at_deadline": {"available_accounts": 4, "available_usd": 88.0,
                      "fraction": 0.29}
    }
  }
}
```

## 7. 配置(`plugins.configs.ollama-cliproxyapi` 新增节)

```yaml
predictor:
  enabled: true
  persistence: true       # 已评审决定:P1 默认落盘,重启不丢历史
  sample-interval: 10m    # 每账号巡检节奏
  budget-per-minute: 8    # 上游账户 API 全局预算
  history: 7d             # 序列保留
  horizon: 48h            # 预测展示窗口
  data-dir: ""            # 空串 = 自动解析 ~/.cli-proxy-api/usage-predictor/;
                          # 显式设 "off" = 仅内存模式
```

落盘说明(已按评审决定调整:默认开启):

- 插件与 CPA 同进程,具备文件系统权限;**不**借 AuthSave 存历史
  (避免污染凭证目录)。
- 目录自动解析失败(如无法定位 home)→ 降级内存模式并 warn 一次,
  注册/请求路径不受影响(预测能力降级,服务不降级)。
- 刷盘策略:每 5 分钟一次追加式 NDJSON + shutdown 时收尾一次;崩溃最多丢
  最近 5 分钟样本,相对 10m 采样间隔可忽略。启动时读取并按 `history` 截断。
- data-dir 只写非敏感序列(账号用摘要 ID),**不落 key 明文**。

## 8. 跨类型(ollama / commandcode / …)总览路径 —— 完全插件内实现

边界约束(2026-10-10 修正):**我们只能修改插件本身** —— Management Center
(React WebUI)与 CPA 主程序都不在可控范围,聚合视图不落任何外部工具:

- **事实依据(已在 quota_page.html 验证)**:插件资源页从 Management Center
  storage 读取 managementKey,并以统一 `Authorization: Bearer` 调用管理 API;
  该 key 对**全部插件路由**通用 ⇒ 插件页面 JS 可用同一凭证调用兄弟插件的
  同形端点,前端 fan-out 收齐各类型数据。
- **本类型(P1)**:页面只渲染 ollama 自己的 /prediction 数据,零外部依赖。
- **跨类型(P3a,规范)**:`/plugins/<plugin>/prediction` 响应统一 `by_type`
  形状,是兄弟插件移植的契约(commandcode-go-cliproxyapi 等由各自插件仓库
  落地 —— 改的仍是插件,符合边界)。
- **跨类型(P3b,聚合页签)**:prediction 页新增"全部类型"视图:
  - 发现兄弟端点:首选管理 API 的插件清单(开工首日确认其存在性);
    备选 `predictor.peer-plugins` 配置列表 + 逐个探测(404 = 未实现约定,跳过);
  - 汇总:各类型 `by_type` 原样并排 + 轻量跨类型合计(总剩余/总速率);
  - 失败隔离:单端点失败/未实现不阻塞其他类型渲染,短超时(≈5s);
  - 兜底:没有任何兄弟插件实现约定时,自动退化为 ollama 单类型视图
    (即 P1 形态),永不劣化。
- 已知限制:页面凭证来自 Management Center(iframe 内嵌),独立打开时
  沿用现有 quota 页的报错与引导,无回归。
- 副产品:`/prediction` 约定公开后,任何第三方工具未来想聚合,直接消费
  各插件端点即可,与本插件无耦合。

## 9. 里程碑

| 阶段 | 内容 | 验收 |
|------|------|------|
| P1 | 首日:usage 分桶探测(30m,结果只影响置信度计算);sampler + store + 双斜率模型 + 账号 ETA 卡 + 类型聚合卡 + `/prediction` API | 单测:模型数学(ETA/reset 分段/R²)、节流与 429 退避、生命周期 join;页面可用 |
| P2 | executor 遥测接入:per-request token 记账 + 隐含单价推断 + 实时速率修正 | 预测在余额滞后场景下偏差下降;token 记账不落内容 |
| P3 | `/prediction` 契约规范(本文档 §6/§8)发布;兄弟插件(commandcode 等)按各自仓库移植;本插件 prediction 页"全部类型"页签:发现 + 并排汇总 + 失败隔离,无兄弟实现时退化单类型 | 多插件端点形状一致;聚合页签在兄弟缺位/失败时不劣化 |

## 10. 风险与边界

- **429 限流**:一切实时路径(页面/预测)不直接打上游;手动刷新走令牌桶。
- **上游滞后/近似语义**:percent 窗口的 reset 语义近似(rolling),预测以
  "能否够用到下一次重置"为主结论,不给伪精确分钟;低置信显式标注。
- **数据安全**:store 不落 key 明文;遥测不落 prompt/响应内容(沿用现有
  debug 元数据边界)。
- **生命周期安全**:与 catalog loop 同款 stop/done 模式,shutdown 落盘
  一次后退出,避免 dlclose 后残留 goroutine。

## 11. 决策记录

| # | 决策 | 状态 |
|---|------|------|
| D1 | P1 即默认落盘持久化(`persistence: true`),重启不丢历史;解析/写失败自动降级内存模式 | **已定(2026-10-10)** |
| D2 | 冻结本方案并完成评审后,才开工 P1 实现 | **已定(2026-10-10)** |
| D3 | executor 遥测**推迟**;P1 先以**临耗尽加速采样**(<20% 剩余或短速>2×长效 → 5m)补精度;`model.go` 抽象 `RateSource`(P1 实现 `SamplingRateSource`,P2 零重构);触发条件:实测 5h 窗 ETA 偏差 >25% 或 included/weekly >10% 才立项 P2 | **已定(2026-10-10)** |
| D4 | 跨类型聚合**全部插件内实现**:prediction 页前端 fan-out(已验证 quota_page.html 的 Bearer 凭证对全管理 API 通用,可调兄弟插件同形端点);兄弟插件(commandcode 等)以各自插件仓库移植达成覆盖 —— 不改 WebUI/主程序 | **已定(2026-10-10,推翻原"manager 侧"落点)** |
| D5 | usage 分桶:**已定(2026-10-10,官方文档证实)** —— 24h=25 个每小时桶(granularity=hour),7d/30d=每日桶,含 USD/token,当前桶 partial;legacy 计划桶仅请求数。⇒ P1 模型升级 **shape-aware**:分桶取日内 profile,余额增量定标,partial 桶不入;"new usage may take time" 已入误差预算(§4.1)。首日探测降级为 30 分钟实测一次:确认响应与文档一致 + 验证 balance 对刚发生请求的时滞(若余额也有滞后期,stale 判定按观察调整) | **已定(2026-10-10)** |
| D6 | ETA 口径:**采纳**:主 = 保守(max 斜率,防排任务超卖),副注 = 乐观(r_long);展示精度随置信度降级(high 单点/medium 区间/low 量级);`stable`(有消耗、零烧速)与 `no-data`(从未消耗)分态 | **已定(2026-10-10)** |
| D7 | 多窗口耦合(用户 2026-10-10 提出的领域事实):ollama Pro 账号的 5h session 与 weekly 窗口**共享同一消耗流**——同一批请求同时燃烧两个计费表,任一 pct 计费表归零即整账号停服,其它窗口冻结到该门重置。实现:① 投影/截止求值从闭式线性外推改为 1h 步进仿真(先检查门、燃烧、后 bump 重整,stay-full 语义保持);② 硬门 = session/weekly 两个 pct 表;included 是服务门(空→dark,shipped 保守语义不变,付费 purchased 兜底不计);③ 账号卡注记:门先空且(无重置或重置晚于该窗自身耗尽)→ 窗口 ETA 替换为冻结时点 + 副注说明,重置早于自身耗尽只算延迟不注记。Max(单月度 USD 窗)无 pct 门,投影路径零变化 | **已定(2026-10-10)** |
| D8 | 账号级"已耗尽等重置"区分(用户 2026-10-10 采纳):纯派生,不碰存储/窗口状态机。`AccountView` 加 `status`(ready/exhausted)+ `next_reset`(最早耗尽窗口重置时刻);判定与 fleet 仿真同口径:任一 pct 门空或 included 空 → exhausted,purchased 是计费渠道不算门,<3 采样的 no-data 读数不拦截;`FleetView` 加 `exhausted_accounts` 计数,fleet ETA 分子分母只用活账号(dark 账号留有历史燃烧估计,会虚拉短总额度 ETA),全 dark 时 ETA 为空;页面账号卡整卡降透明度 + 徽章"已耗尽 · {t} 恢复",fleet 账号副注追加"· {k} 等待重置"。included 空但 purchased 有结余时是否可继续烧付费:保持保守 dark,部署后按实际账号形态再定(D7 同问) | **已定(2026-10-10)** |
| D9 | fleet risk 徽章重构(用户 2026-10-10 报告误报:45 账号 61% 容量却 CRITICAL):原 critical 触发器"任一账号 USD 计费面归零"对 Pro 形态账号(无 included 窗、purchased 恒 $0)永久为真,与 D8"purchased 归零不算 dark"直接冲突;"首耗尽 <24h → tight"在滚动重置池里也恒真。改为**双轴取最差**:容量轴(<10% critical / 投影末端 <30% tight)保留;新增 dark 比例轴——直接复用 D8 的 `exhausted_accounts`,dark/已采样 ≥50% → critical、≥20% → tight(等待重置是滚动周期常态,半数停服才是危机);两个失真触发器删除,首耗尽信息只留在自己的统计卡。同轮修复页面 i18n 回退:t() 改 `key in dict` 判断,en 里空串值(`eta-after` 等)不再被当缺键吐出键名(`in 13deta-after` → `in 13d`)。注意:store 既有不变量 `Included.Cap > 0` 才落窗口,cap-0 额度不进数据模型 | **已定(2026-10-10)** |
| D10 | 持久化根因修复(用户实测 79→83MB 文件重启"丢"历史,2026-10-11 锁定双重根因):① **加载自锁**——`persistLoadMaxBytes=64MiB` 硬顶:文件超限 → load 报 "oversized" → 静默 memory-only → 而压缩只在"启动加载成功后"才跑 → 超限即永久自锁,文件只增不减;② **20× 写放大**——容器证据 679,164 行中 600,087 行是小时桶(88%):`mergeHour` 对自带 `full=true` 的到达桶无条件覆盖+标脏,而 24h usage range 在桶完成后约一整天内每个采样都持续重报同一批完整桶(partial→full 只发生一次,完整桶却被重复追加数十次)。修复:①流式加载(`load(sink)` 按行解码进 store,删硬顶,内存随保留态有界——用户现有 83MB 文件下次启动即流式恢复);②`mergeHour` 无变化检测(同桶同 full 标志同 USD → 不标脏,实测 24 桶/采样 → ~1/采样预期);③`Restore` 保留期对齐(inherit TTL:恢复态按各自最新采样自锚裁剪 + 小时桶按墙钟下限删,恢复后压缩只重排保留态——长期停机的文件也不会带出超期记录);④稳态压缩(flush 时文件 >8MiB 直接从保留态 rewrite,增长自限,`.tmp` 残留清理);⑤失败可见化(引擎 `PersistStatus`:ok/off/load-failed/write-failed,降级不再是静默——页面 meta 下警告横幅中英双语,off 态(无数据目录)同样显示);⑥`Snapshot`/disabled-engine 载荷带 `persistence` 字段(ok 时省略) | **已定(2026-10-11)** |

> **方案已冻结(2026-10-10,D1~D6 全部已定),P1 开工。**