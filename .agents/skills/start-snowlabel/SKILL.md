---
name: start-snowlabel
description: >
  启动 / 重启 SnowLabel 的本地服务与公网 Demo。当用户提到电脑重启后服务停了、
  项目网址 / 公网链接打不开或失效了、"帮我启动服务""起一下服务器""把 demo 跑起来"
  "重新生成公网链接""launch/start the SnowLabel demo",或任何想让 SnowLabel 跑起来
  (9090 标注界面 + MariaDB + MinIO + cloudflared 公网隧道)的场景,都要用本 skill——
  即使用户没明说"skill"或"启动"二字也适用(例如"网址带不开了""关机后项目没了")。
  公网网址已固定为 https://snowlabel.zhuwenmao.com(Cloudflare 命名隧道),重启不变。
  本 skill 知道正确的 env、加载等待、以及固定网址的验证方式。
---

# 启动 SnowLabel 服务

## 这个 skill 干什么

SnowLabel 的公网 Demo 由 **4 个服务**组成,本 skill 把它们一次性起好并报告网址:

| 服务 | 端口 | 作用 |
|---|---|---|
| SnowLabel 服务器 | 9090 | 标注 UI(`/`)、记忆面板(`/inspector`)、API |
| MariaDB | 3306 | mask 元数据缓存 |
| MinIO | 9000 / 9001 | 图片对象存储 |
| cloudflared 命名隧道 `snowlabel` | — | 把 9090 暴露成**固定**公网网址 `https://snowlabel.zhuwenmao.com`(重启不变) |

仓库里已有一条龙脚本 **`start_public_demo.ps1`** 把这些全做了(幂等:已在跑的会跳过,
旧的 9090 会先杀掉重起),本 skill 主要是**正确地调它 + 抓出新网址 + 验证**。

## 前置条件(很重要)

1. **科学上网代理必须开着**(Clash 等,通常端口 7897)。cloudflared 注册隧道的接口在国内被墙,
   不走代理隧道一定建不起来。脚本会自动探测 7897/7890/10809 等常见端口。
   - 没开代理:服务器和三库照样能起,**只是没有公网网址**(本机 `http://localhost:9090/` 仍可用)。
2. 必须在**仓库根目录** `D:\project\codex_workspace\LabelAnything` 下运行。

## 怎么做(标准流程)

用 PowerShell 工具运行下面这一条,**超时设到 300000ms(5 分钟)**——服务器加载 DINOv3-L 约 40 秒,
脚本会等 `/stats` 就绪(最多 180s)再等隧道网址(最多 30s),然后打印公网链接:

```powershell
powershell -ExecutionPolicy Bypass -File "D:\project\codex_workspace\LabelAnything\start_public_demo.ps1"
```

脚本结束时会打印**固定公网网址 `https://snowlabel.zhuwenmao.com`**(每次都一样,不再变)。
转达给用户即可,并附上本机地址 `http://localhost:9090/`(标注界面)和 `http://localhost:9090/inspector`(记忆面板)。

## 验证(报告前确认真起来了)

```powershell
# 本机 9090 就绪(应返回 200)
(Invoke-WebRequest http://localhost:9090/stats -UseBasicParsing -TimeoutSec 5).StatusCode
# 三库端口在听
Get-NetTCPConnection -LocalPort 9090,3306,9000 -State Listen | Select-Object LocalPort
# 公网固定网址可达(走 Cloudflare 边缘,应返回 200)
(Invoke-WebRequest https://snowlabel.zhuwenmao.com/stats -UseBasicParsing -TimeoutSec 15).StatusCode
```

隧道是否连上,看日志里有没有 `Registered tunnel connection`:

```powershell
Select-String -Path "D:\project\codex_workspace\LabelAnything\data\cache\cf.err.log" -Pattern 'Registered tunnel connection' | Select-Object -Last 1
```

## 坑(务必避免)

- **公网网址已固定为 `https://snowlabel.zhuwenmao.com`,重启不变**(Cloudflare 命名隧道 `snowlabel`)。一次性配置已完成:`~/.cloudflared\` 里有 `cert.pem` + `<tunnelID>.json`,`snowlabel.zhuwenmao.com` 已 CNAME 到该隧道。**别再退回随机的 `tunnel --url` 快速隧道。** 若 `~/.cloudflared` 的凭证丢了需重配,见下方"命名隧道重配"。
- **绝不要传历史的 `SNOWLABEL_CONF_AUTO` / `SNOWLABEL_CONF_SUGGEST`(0.61/0.41 等)。** 脚本已显式 `Remove-Item` 清掉它们并用 `config.py` 默认 0.68/0.55。手动起服务器时也别带。
- **`SNOWLABEL_RES` 必须 512**(DINOv3 是 patch16,512÷16 整除)。脚本已设好,别改成 224。
- **公网 Demo 用 numpy 后端 + `SNOWLABEL_MULTIUSER=1` + 独立库 `data\memory_public`**——只有 numpy 后端真正按 user_id 隔离访客,且绝不碰你本地的车辆库。脚本已设好。
- 隧道建不起来 **99% 是代理没开/端口没命中**——让用户打开科学上网工具后重跑;或看 `data\cache\cf.err.log`。
- **`-Fresh` 参数会清空所有访客图片和公网记忆库,慎用**——只有用户明确说"从零清空重来"时才加。

## 只要本机、不要公网网址时

用户若只想本机用、不需要给别人的链接,可只起服务器(仍需先 `start_infra.ps1` 起三库),
跳过 cloudflared。但默认情况(用户说"网址打不开了""起服务")都是要公网的,**用 `start_public_demo.ps1`**。

## 停服务

```powershell
powershell -ExecutionPolicy Bypass -File "D:\project\codex_workspace\LabelAnything\stop_public_demo.ps1"
```

## 命名隧道重配(仅当 `~/.cloudflared` 凭证丢失,正常无需做)

正常情况隧道凭证一直在,不用碰。只有 `~/.cloudflared\cert.pem` 或 `<tunnelID>.json` 丢了才需重建
(全程带 `$env:HTTPS_PROXY="http://127.0.0.1:7897"`,cloudflared 在 `data\cache\cloudflared.exe`):

1. `cloudflared tunnel login` —— 生成授权链接;用户在浏览器选 `zhuwenmao.com` → 授权。
   - 国内网络下自动取证书常 `EOF` 失败,会回退成**浏览器下载 `cert.pem`**;把它从下载目录拷到 `~/.cloudflared\cert.pem`。
   - 账号邮箱必须先验证(163 邮箱会"吃掉"一次性验证链接 → 用 QQ/Gmail/Outlook 换掉再验)。
2. `cloudflared tunnel create snowlabel` —— 生成隧道 + `<tunnelID>.json`。
3. `cloudflared tunnel route dns snowlabel snowlabel.zhuwenmao.com` —— 自动建 CNAME。
4. 之后 `start_public_demo.ps1` 里的 `tunnel run ... snowlabel` 即可跑起来。

## 备注

- 服务器每次请求都从磁盘重读 `snowlabel/static/labeler.html`(`Cache-Control: no-store`),所以改前端后用户刷新即生效,**改前端不需要重启服务器**。
- 服务器、MariaDB、MinIO、cloudflared 都是 `Start-Process` 起的独立(最小化/隐藏)进程,脚本退出后它们继续在后台跑。
