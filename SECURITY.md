# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's **Report a vulnerability**
feature (Security tab -> Advisories) on this repository once it is published. Do not open a public
issue. Include the version/commit, a minimal reproduction and the impact you expect.

This is a small educational project maintained on a best-effort basis; there is no guaranteed
response time.

## Scope and threat model

In scope: the filesystem sandbox (path traversal, symlink/junction escapes, secret-file
exclusion, size/count limits), tool permission boundaries, handling of malformed model output,
secret handling in config and logs, and the Docker Compose defaults.

The repository under review is **untrusted input**, and so is every model output. The design
assumes a malicious repository may try to read files outside its root, make an agent execute
something, or inject instructions into the model.

Known limits (not vulnerabilities in themselves):

- Prompt injection cannot be prevented, only contained: tools are read-only, so a successful
  injection can at worst yield a wrong finding.
- The sandbox narrows but does not eliminate time-of-check/time-of-use races against a hostile
  local process.
- Secrets in ordinarily named files are not detected.
- In `deepseek` mode, source excerpts are sent to an external provider by design.
- The LiteLLM Proxy runs database-free with one master key; it is published on `127.0.0.1` only
  and must not be exposed publicly as configured.
- Vulnerabilities in dependencies (LangGraph, LiteLLM, Ollama, ...) should be reported upstream.
  This project does not scan the reviewed code for CVEs.
