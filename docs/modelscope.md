# ModelScope 异步图像接口配置

`modelscope` 适配器用于 ModelScope API-Inference 的异步图像生成接口。它会提交图像任务，轮询远端状态，并在任务成功后下载 `output_images` 中的结果。上层仍使用插件统一的任务、队列、审核、额度和结果发送流程。

当前版本支持：

- `POST /v1/images/generations` 异步提交
- `GET /v1/tasks/{task_id}` 状态轮询
- `SUCCEED` 成功状态和 `output_images` 下载
- ModelScope Access Token Bearer 认证
- 文生图
- 由已启用模型能力决定的图生图；本地参考图会转为 data URL 数组发送到 `image_url`
- `negative_prompt`
- 通过尺寸映射发送显式 `size`（`WxH`）
- 代理、HTTP 超时、轮询间隔和总等待时间配置

当前不支持：

- SSE、Webhook 或远端取消
- 在远端任务已创建后自动重新提交任务
- 未定义格式的 `seed`、`steps`、`guidance` 或 `loras` 高级参数

## 前置条件

1. 注册并登录 ModelScope。
2. 在账号页面创建 Access Token。
3. 绑定阿里云账号并完成实名认证；ModelScope API-Inference 使用前需要完成这两项。
4. 在 [AIGC 模型列表](https://www.modelscope.cn/aigc/models) 中确认模型已支持 API-Inference，且账号有可用额度。

不要把 Access Token 写入 README、日志、截图或仓库配置。将它仅填入 AstrBot WebUI 的“API 密钥”字段，或由部署环境的私密配置注入。

## 快速配置

1. 在“图像模型供应商”中新增 **ModelScope 异步接口**。
2. 填写供应商名称，例如 `ModelScope`。
3. 接口地址保持默认 `https://api-inference.modelscope.cn`，或填写兼容服务的根地址。
4. 在“API 密钥”填写 ModelScope Access Token。
5. 在“可用模型列表”填写可用模型，例如 `Qwen/Qwen-Image`。
6. 在“生图模型”中选择 `ModelScope/Qwen/Qwen-Image`。

默认模板仅勾选“文生图”。只有选中的模型在 ModelScope AIGC 文档中明确支持图片编辑时，才勾选“图生图”。

示例配置概念如下，Token 仅作占位：

```text
供应商名称: ModelScope
接口地址: https://api-inference.modelscope.cn
API 密钥: <MODELSCOPE_ACCESS_TOKEN>
可用模型: Qwen/Qwen-Image
模型能力: 文生图
异步轮询间隔: 5 秒
异步总等待时间: 600 秒
```

服务根地址、`/v1` 地址和误填的图像生成路径都会被适配器规范化为：

```text
<服务根>/v1/images/generations
<服务根>/v1/tasks/<task_id>
```

## 请求参数

每个插件子请求会提交一次：

```json
{
  "model": "Qwen/Qwen-Image",
  "prompt": "A quiet lakeside cabin at dawn",
  "negative_prompt": "blurry, low quality",
  "size": "1024x1024"
}
```

适配器添加以下请求头：

```http
Authorization: Bearer <MODELSCOPE_ACCESS_TOKEN>
Content-Type: application/json
X-ModelScope-Async-Mode: true
```

模型支持图生图、且模板勾选“图生图”时，插件会将经过既有安全校验和图片转换后的参考图发送为：

```json
{
  "image_url": [
    "data:image/png;base64,<REFERENCE_IMAGE_BASE64>"
  ]
}
```

多张参考图会按原顺序组成数组。具体模型的多图限制、支持的格式和编辑能力以 ModelScope 模型页面为准；不支持图片编辑的模型不要勾选“图生图”。

## 尺寸映射

ModelScope 图像接口使用显式 `WxH` 的 `size`，插件统一请求则是宽高比与分辨率。`默认尺寸` 默认是 `1024x1024`，因此新建 provider 会固定发送该尺寸。清空此配置后，适配器才会使用框架宽高比与分辨率映射。

适配器按下面顺序决定是否发送 `size`：

1. 请求分辨率本身已经是 `WxH`。
2. `尺寸映射 JSON` 中的 `分辨率:宽高比`、分辨率或宽高比键。
3. `尺寸映射 JSON` 的 `default` 键。
4. `默认尺寸`。
5. `默认尺寸` 为空时，使用框架公共映射：`1K` 使用 `RESOLUTION_1K_MAP`，`2K`/`4K` 使用 `RESOLUTION_2K_MAP`；未选择宽高比时按 `1:1`。
6. 宽高比和分辨率均为“不指定”时不发送 `size`。

示例：

```json
{
  "1K:1:1": "1024x1024",
  "1K:16:9": "1280x720",
  "2K:1:1": "1536x1536",
  "default": "1024x1024"
}
```

每个模型允许的尺寸范围不同。请按照模型页面的限制配置映射；例如文档中 Qwen-Image 的尺寸范围与部分 SD、FLUX 模型不同。

## 轮询、重试和取消

- `超时时间` 控制单次提交或状态查询 HTTP 调用；结果图片下载沿用插件的独立下载超时。
- `异步轮询间隔` 默认 5 秒，控制未完成任务的状态查询间隔。
- `异步总等待时间` 默认 600 秒，从收到 `task_id` 后开始计算，覆盖后续轮询和结果图片下载。
- 一次远端任务会固定使用提交时选中的 API Key，即使其他并发请求发生 Key 轮换也不会改变。
- 收到 `task_id` 后，远端 `FAILED`、未知状态、协议异常、超时和最终下载失败都不会重新提交生成任务，避免重复计费。
- 轮询和下载的短暂传输失败会在剩余总等待时间内重试同一远端任务或同一图片 URL。
- 提交连接超时或 5xx 可能表示服务端已经创建任务但客户端未拿到 `task_id`；本版本默认不自动重提。
- 取消插件任务会取消本地等待、HTTP 调用或轮询 sleep，不会调用 ModelScope 远端取消 API。远端任务可能继续运行并消耗额度。

## 排障

### 返回“异步提交状态未知”

请求可能已经到达 ModelScope，但客户端没有拿到可用的 `task_id`。为避免重复生成，插件不会自动重新提交。检查网络、代理和服务端日志后，再由用户决定是否重新发起一次新任务。

### 返回“异步任务失败”

ModelScope 已明确返回 `FAILED`。检查模型名称、额度、提示词、尺寸和模型可用性。管理员可在“向用户显示详细错误信息”开启后获取经过脱敏的远端错误摘要。

### 返回“异步等待超时”

远端任务在总等待预算内未完成，插件停止本地等待而不会重新提交。可适当增加“异步总等待时间”，但不要通过增大外层重试次数来处理已提交的任务。

### 图生图未生效

确认当前模型在 ModelScope 明确支持 `image_url` 图像编辑，并且 provider 的“模型能力”已勾选“图生图”。默认 `Qwen/Qwen-Image` 模板不会声明此能力。