# Starting prompts for ESA

Download a version of the Element Surrogate Agent (ESA) from GitHub Releases
using **Source code (zip)** and extract it. **Code → Download ZIP** also works
for the selected branch, and a clone is another option. Open the chosen
project in Codex or Claude Code and use the matching prompt below. The AI does
not need a previous ESA conversation or global Genesis installation.

## 1. Starting in the extracted ESA folder

Open the extracted ESA folder as your project and paste:

~~~text
Read SETUP.md and set up the Element Surrogate Agent (ESA) in this project for
this AI client. Choose an appropriate Python environment, verify that the tools
work, and tell me how to provide my field data and begin modeling.
~~~

## 2. Adding ESA to an existing project

Keep the extracted ESA software in a separate folder. Open your existing
surrogate-modeling project in the AI client. Replace `<ESA folder>` below with the
software folder's full path, such as `C:/tools/element-surrogate-agent` on Windows
or `/home/user/tools/element-surrogate-agent` on Linux, then paste:

~~~text
The ESA software is at "<ESA folder>". Read its SETUP.md and review this existing
surrogate-modeling project. Add the ESA tools and guidance that will help,
reusing prior work and preserving the project's organization, working software
and accepted decisions. Explain important conflicts before changing established
choices. Verify the additions and summarize what to do next.
~~~

You can add a specific focus, such as field analysis, model benchmarking or
model/data cards. Existing work should be assessed and reused where appropriate.
Using ESA does not by itself require new tracking data or model retraining.

For a project containing several magnets, the AI should identify the relevant
single-magnet study and keep its ESA records there rather than treating all
magnets as one case. If that study folder becomes the setup target, the AI should
explain which folder to reopen so the client loads its tools.

## After setup

Describe the scientific task and provide the relevant file path. For example:

~~~text
Build a 4D transverse particle-map surrogate from the field at "<field file>".
Analyze the field first, recommend suitable ranges and accuracy/speed goals,
and estimate the computing time. Ask me about physical facts you cannot establish.
~~~

Include known beam parameters, geometry and performance goals when available.
The AI should help resolve unknowns rather than require you to fill out a long
questionnaire before useful work can begin.

Setup may require normal client permissions or a reopen to load its tools.
If asked to reopen, send:

~~~text
I reopened this project. Please confirm ESA is available in this session and finish the setup check.
~~~

The AI should confirm a live ESA tool call before reporting the connection ready.
ESA currently registers five workflow skills; individual capabilities can be
used as needed. Initialization may add standard work/records folders, but it
does not automatically understand or qualify existing results.

See [SETUP.md](SETUP.md) for the AI's procedure,
[the usage manual](docs/usage_manual.md) for scientific examples, and
[QUALIFICATION.md](QUALIFICATION.md) for the tested scope.
