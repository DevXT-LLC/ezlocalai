# Reverse-tunnel redirect reference

`tunnel_redirect_cases.json` records 59 actual `ClientSession.ws_connect`
connections made by the original Python CUDA image on 2026-10-20:

```
sha256:c2180ffee0b1579dc63b33236777c1bdc70c899c51f979c91969e04d601ff5cb
aiohttp 3.14.5
```

The fixture records the installed request implementation, cookie jar and cookie
helper source hashes. Capture used two loopback HTTP listeners and one HTTPS
listener with a temporary trusted certificate. Credentials are synthetic.
Each record includes the source outcome, outgoing request targets,
Authorization/Cookie fields and WebSocket challenge reuse across redirects.
The native test repeats the network exchanges through
`TunnelWebSocketConnection`, including TLS certificate verification and the
first WebSocket frame arriving with the upgrade response.

Coverage includes all five redirect statuses, relative and absolute references,
query/fragment handling, the initial-only worker_id parameter, Location/URI
fallback, logical ws/wss origin changes, Bearer and URL Basic authentication,
cross-origin credential removal, HTTP/HTTPS transitions, secure/IP cookies,
nine versus ten redirects and invalid final upgrade challenges. Cookie cases
cover reserved names, duplicate fields, quoted values, unknown attributes,
expiry replacement, controls and parser field/value/attribute limits. Response
fields at 8,190 bytes are accepted; larger fields reject the upgrade, matching
the source client. An 8,000-byte quoted cookie also checks bounded parsing.
A separate native scenario cancels a stalled upgrade after a redirect.

The modern response-header cookie profile is used by tunnel connections.
Existing fallback clients retain their separately captured aiohttp 3.13.2
profile; the 6,180-operation `http_cookies_reference` regression protects it.
The modern parser is bounded, but full aiohttp 3.14 cookie-jar eviction and
aggregate outgoing-header quotas are not implemented. Exhaustive malformed
HTTP response parsing, every URL/credential encoding, proxy behavior and all
TLS/network failure schedules are not established by these finite captures.

The capture script and detailed build/network evidence are retained under
`build/validation/tunnel-redirects-20261020/` in the local workspace.
First-party executable regression code is C++.
