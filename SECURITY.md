# Security policy

## Report a vulnerability

Email the base-software maintainer Kilean Hwang at
[hwang@frib.msu.edu](mailto:hwang@frib.msu.edu). Do not put credentials, private
fields, particle data or sensitive machine details in a public issue.
Include the affected version, reproduction, impact and a proposed fix if known.
Users designate contacts for software and artifacts created in their own cases.

## Supported scope

ESA is supervised research software. Use an identified, qualified release and
its stated platform/dependency scope; current development source and release
candidates are not automatically qualified. Report defects against an exact
version. Old qualification results do not establish a newer version's behavior.

MCP runs with the selected local client's authority. Local Python adapters,
study scripts and incoming project files must be reviewed with the same care as
other code the user runs. Path checks, locks, hashes and approvals are consistency
controls, not an operating-system security boundary or authenticated human
identity. Strict isolation requires an actual OS boundary.

Keep software updated in a separate environment and use the staged project
update procedure. Keep the previous environment and recovery records until
adoption is accepted. Review external feedback for sensitive content and obtain
sharing authority separately; no feedback route is required for local use.
