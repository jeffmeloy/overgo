# Security policy

## Scope and threat model

The overgo server and swap proxy are single-operator tools that bind to
loopback (`127.0.0.1`) by default. They are not hardened for exposure to
untrusted networks; placing them behind a reverse proxy on a shared network
is the operator's decision and the operator's responsibility.

A credential-less request is admitted by one rule the server and the swap
proxy share (`internal/apimanifest.AdmitCredentialless`): on a real
listener the Host must be loopback, since a matching Origin alone cannot
stop DNS rebinding, and the browser's cross-origin protection applies. The
proxy applies the rule before anything of its own changes: a foreign
origin can neither place a provider key in the proxy's environment nor
launch a child, and it never reaches the idle shell. A request that reaches
beyond the server (the proxy's `?swap=` on the health probe, the provider
listing that sends a key to an endpoint) is admitted as a mutation
(`AdmitCredentiallessEffect`): cross-site fetch metadata or a foreign
Origin refuses it, the workbench's own same-origin requests and loopback
clients without fetch metadata pass. A server with a key may bind a
non-loopback address, so this rule is not what admits a request to it: the
key is, and a route that stays public answers whoever reaches it.

A provider's variable goes to one endpoint: a declaration binds the pair until
it retires, and a listing names a declared pair or carries a key of its own,
which goes to that endpoint and is kept nowhere. The server's own secrets
(`OVERGO_API_KEY`, `OVERGO_HF_TOKEN`, `OVERGO_MODELSCOPE_TOKEN`) cannot be
declared. An authenticated client can still bind a variable no declaration
holds yet.

## API keys

- The server accepts an optional bearer key; when configured, every request
  must carry it except the routes that name what is served and show nothing
  of the disk: `/health`, `/healthz`, `/v1/health`, `/metrics`, `/v1/models`
  and `/models`.
- The swap proxy's idle shell takes no key: its `/catalog/models` shows disk
  paths to any client the shared admission passes, which is to say on loopback.
- The web GUI holds the entered key in memory for the page session.
  Persisting it in browser storage is opt-in via the "remember" control
  next to the key field; unchecking it removes the stored key.
- Keys never appear in URLs, logs, or generated artifacts.

## Model output handling

The web GUI renders model output through a sanitizing renderer that builds
DOM nodes from text (`internal/server/webui/md.js`); raw HTML injection
paths are not part of the UI toolkit. Model output is data, never markup.

## Data handling

The server operates only on the local repository store (`overgodb-store`)
and local model files. It sends no telemetry and makes no outbound network
requests on its own.

## Reporting

Report suspected vulnerabilities to the repository owner
(jeffmeloy@gmail.com). This is a private repository; there is no public
disclosure program.
