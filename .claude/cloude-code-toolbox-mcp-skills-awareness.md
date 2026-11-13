# Claude Code ToolBox — MCP & Skills awareness

_Generated: 2026-10-28T08:30:27.212Z_

## How to use this report

- **Saved copy:** This file is **`.claude/cloude-code-toolbox-mcp-skills-awareness.md`** — refreshed whenever the toolbox runs an MCP & Skills scan (including on workspace open when auto-scan is enabled). It is meant for **Claude Code workspace context** together with `CLAUDE.md` (which gets a shorter replaceable summary when auto-merge is on).
- **MCP:** Lists **configured** servers from Claude Code config (`~/.claude.json` for user scope, `.mcp.json` for project scope). Use `/mcp` in the Claude Code panel to connect servers for your session.
- **Skills:** **On-disk** folders with `SKILL.md`. Claude Code does not auto-load them; attach `SKILL.md` or paths in chat when useful.
- **Task routing:** When the user’s request matches a server’s purpose (e.g. Confluence → Confluence/Atlassian MCP), prefer that **server id** from the tables below.

---

## MCP — workspace

Workspace `mcp.json` _(folder: knowledge_base)_

- **/home/max/ML/knowledge_base/.mcp.json** — _File missing_

_No active workspace servers in mcp.json._

## MCP — user profile

- **/home/max/.claude.json** — _File exists — no servers defined_

_No active user-scoped servers in mcp.json._

## Skills (local `SKILL.md` folders)

### Project-scoped

_None found (or no workspace open)._

### User-scoped

- **automate** — `/home/max/.agents/skills/automate`
  - Use this skill to create Codex Automations.

- **babysit** — `/home/max/.agents/skills/babysit`
  - >-

- **bdk-create-agent** — `/home/max/.agents/skills/bdk-create-agent`
  - >-

- **bdk-debug** — `/home/max/.agents/skills/bdk-debug`
  - >-

- **bdk-deploy** — `/home/max/.agents/skills/bdk-deploy`
  - >-

- **bdk-evals** — `/home/max/.agents/skills/bdk-evals`
  - >-

- **bdk-framework-map** — `/home/max/.agents/skills/bdk-framework-map`
  - >-

- **bdk-github** — `/home/max/.agents/skills/bdk-github`
  - >-

- **bdk-hillclimb** — `/home/max/.agents/skills/bdk-hillclimb`
  - >-

- **bdk-mcp-auth** — `/home/max/.agents/skills/bdk-mcp-auth`
  - >-

- **bdk-migrate-v2** — `/home/max/.agents/skills/bdk-migrate-v2`
  - >-

- **bdk-otel** — `/home/max/.agents/skills/bdk-otel`
  - >-

- **bdk-setup-slack** — `/home/max/.agents/skills/bdk-setup-slack`
  - >-

- **canvas** — `/home/max/.agents/skills/canvas`
  - >-

- **create-hook** — `/home/max/.agents/skills/create-hook`
  - >-

- **create-rule** — `/home/max/.agents/skills/create-rule`
  - >-

- **create-skill** — `/home/max/.agents/skills/create-skill`
  - >-

- **create-subagent** — `/home/max/.agents/skills/create-subagent`
  - >-

- **env-setup** — `/home/max/.agents/skills/env-setup`
  - >-

- **loop** — `/home/max/.agents/skills/loop`
  - >-

- **migrate-to-builds** — `/home/max/.agents/skills/migrate-to-builds`
  - >-

- **migrate-to-skills** — `/home/max/.agents/skills/migrate-to-skills`
  - >-

- **new-repo** — `/home/max/.agents/skills/new-repo`
  - >-

- **origin** — `/home/max/.agents/skills/origin`
  - >-

- **rename-chat** — `/home/max/.agents/skills/rename-chat`
  - >-

- **review** — `/home/max/.agents/skills/review`
  - Review code changes with the Bugbot or Security Review subagent.

- **review-bugbot** — `/home/max/.agents/skills/review-bugbot`
  - Review code changes with Bugbot subagent.

- **review-security** — `/home/max/.agents/skills/review-security`
  - Review code changes with Security Review subagent.

- **sdk** — `/home/max/.agents/skills/sdk`
  - >-

- **share** — `/home/max/.agents/skills/share`
  - >-

- **shell** — `/home/max/.agents/skills/shell`
  - >-

- **split-to-prs** — `/home/max/.agents/skills/split-to-prs`
  - >-

- **statusline** — `/home/max/.agents/skills/statusline`
  - >-

- **subscribe** — `/home/max/.agents/skills/subscribe`
  - >-

- **update-cli-config** — `/home/max/.agents/skills/update-cli-config`
  - >-

- **update-cursor-settings** — `/home/max/.agents/skills/update-cursor-settings`
  - >-

- **walkthrough-artifacts** — `/home/max/.agents/skills/walkthrough-artifacts`
  - >-

---

## Suggested next steps

- **MCP:** Use this extension’s hub **MCP** tab, or `claude mcp list` in the terminal. In Claude Code, use `/mcp` to connect servers for the session.
- **Edit config:** Open `~/.claude.json` (user MCP) or `<workspace>/.mcp.json` (project MCP) via the extension commands.
- **Refresh this report:** run **Intelligence — scan MCP & Skills awareness** again after changing MCP config or adding skills.

_Report from Claude Code ToolBox extension._
