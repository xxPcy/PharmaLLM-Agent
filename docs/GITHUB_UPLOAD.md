# 首次上传 GitHub

上传的是整理后的发布目录。原服务器上的模型和服务继续使用原运行目录，本步骤不需要重新加载模型，也不会产生公开体验网址。

## 1. 创建空仓库

登录 GitHub，打开 [新建仓库](https://github.com/new)：

- Repository name：`PharmaLLM-Agent`。
- Description：`Medical LLM agent and MRI segmentation workspace built on PharmaLLM, with LangGraph, vLLM, and nnU-Net.`。
- Visibility：准备开源时选 Public。
- **不要勾选**自动创建 README、`.gitignore` 或 License，本发布目录已经包含这些文件。

现有模型工程的远程仓库为 `xxPcy/PharmaLLM`。在同一账号下创建新仓库后，HTTPS 地址为 `https://github.com/xxPcy/PharmaLLM-Agent.git`。新仓库专门展示衍生应用，原仓库保留训练、评测与推理工程的定位。

## 2. 检查并提交本地文件

在整理后的 `PharmaLLM_GitHub` 目录执行，不能在包含模型和数据的父目录执行：

```bash
cd /path/to/PharmaLLM_GitHub
python3 tools/check_public_repo.py
git init -b main
```

首次使用 Git 时，为当前仓库设置提交者身份。替换下面的示例值；邮箱可以使用 GitHub Settings → Emails 提供的隐私 noreply 地址：

```bash
git config user.name "xxPcy"
git config user.email "YOUR_GITHUB_NOREPLY_EMAIL"
```

接着执行：

```bash
git add .
python3 tools/check_public_repo.py --staged
git status --short
git diff --cached --stat
git commit -m "Initial release: PharmaLLM-Agent"
```

检查脚本返回 `"ok": true` 后再提交。如果失败，根据返回的文件路径和原因移出对应文件；脚本不输出命中的密钥内容。`--staged` 会检查真正将要提交的 Git 索引内容，避免只检查工作区。

## 3. 关联并上传

```bash
git remote add origin https://github.com/xxPcy/PharmaLLM-Agent.git
git push -u origin main
```

使用 HTTPS 时，命令行可能要求认证。Username 填 GitHub 用户名，Password 使用个人访问令牌 PAT，**不是网页登录密码**。仅在交互式密码输入框输入令牌，不把它放进 URL、脚本或 README。也可以使用已配置好的 GitHub CLI 或 SSH 认证。

如果已有同名远程、远程不是空仓库，或服务器尚未配置认证，先处理相应提示；不要通过强制推送覆盖远程内容。

## 4. 展示设置

上传完成后，在仓库网页补充：

- About：一句话介绍；域名部署成功后再填写 Website。
- Topics：`medical-llm`、`langgraph`、`fastapi`、`vllm`、`medical-imaging`、`nnunet`。
- README：真实体验地址、演示登录方式、约一分钟的视频链接。
- GitHub 个人主页：将此仓库 Pin 到展示区。

新仓库创建并上传成功后，可将[原仓库引导文案](RELATED_PROJECTS.md)中的 Markdown 添加到旧仓库 README。本次只准备文案，未修改或推送旧仓库。

首次上传后会触发仓库附带的 Actions CPU 检查。它使用临时 PostgreSQL / Redis 和 FakeModel，不会连接你的 GPU 服务器；只有线上实际运行通过后才标注 CI 成功。

## 5. 后续更新

发布目录是代码快照，原开发目录的修改不会自动同步。先将新代码重新整理到发布目录，确认未带入运行配置，再提交：

```bash
python3 tools/check_public_repo.py
git add .
python3 tools/check_public_repo.py --staged
git diff --cached --stat
git commit -m "Describe the change"
git push
```

检查仅覆盖常见凭据与文件规则；新截图、文档和数据仍需人工检查。权重留在服务器即可，不需要为了开源代码上传大模型。

参考：[GitHub 官方本地代码上传说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、[命令行认证](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github)、[大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。
