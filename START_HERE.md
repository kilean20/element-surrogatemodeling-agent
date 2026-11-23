# Start using ESA

Download a version from GitHub **Releases → Source code (zip)** and extract it.
If no release is listed, **Code → Download ZIP** downloads the selected branch.
GitHub supplies the download; you do not need a separately emailed archive.
A repository clone works too. Choose where you want to work:

| Your situation | Open in Codex or Claude Code | Starting prompt |
|---|---|---|
| New project using the extracted ESA folder | The extracted folder containing this file and `SETUP.md` | [New-project prompt](USER_PROMPTS.md#1-starting-in-the-extracted-esa-folder) |
| Existing surrogate-modeling project | Your existing project; keep the ESA software in a separate folder | [Existing-project prompt](USER_PROMPTS.md#2-adding-esa-to-an-existing-project) |

Copy the matching prompt from [USER_PROMPTS.md](USER_PROMPTS.md). It sends the
AI to [SETUP.md](SETUP.md). No previous ESA conversation or global Genesis skills
are needed. The extracted folder may be renamed.

Use an installed, signed-in AI client and Python **3.11 or newer** with `venv`
and `pip`. Python 3.11 is the reference configuration. The AI can help inspect
your installation and choose a private environment. Libraries are installed
separately; a GPU, uv, Docker and an ESA hosting service are not prerequisites.

Allow normal project permissions. If asked, reopen the project and send:

~~~text
I reopened this project. Please confirm ESA is available in this session and finish the setup check.
~~~

The AI should confirm an actual ESA tool call, rather than relying only on the
installer's check. Setup is complete when that live connection is verified.

Then provide your field-file path and modeling goal. Known beam parameters,
coordinate conventions, geometry and performance targets help; unknown facts can
be investigated or left explicitly unknown. The prompt guide includes an example.

The base software ships without facility fields, training data, magnet models or
credentials. You decide ownership and terms for your own work. Read
[test results and limits](QUALIFICATION.md) and the [usage manual](docs/usage_manual.md).
