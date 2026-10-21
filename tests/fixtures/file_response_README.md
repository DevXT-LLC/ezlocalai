# File download reference

`file_response_cases.json` contains 136 executed ASGI responses from the original
Python CUDA image
`sha256:c2180ffee0b1579dc63b33236777c1bdc70c899c51f979c91969e04d601ff5cb`,
captured on 2026-10-20. Its installed Starlette version is **1.7.0**. The fixture
records the SHA-256 of `inspect.getsource(starlette.responses.FileResponse)`.
This supersedes older native assertions for multipart framing and 416 headers.

For each case the capture invokes either `FileResponse(path,
filename="héllo_model.pt", media_type="application/octet-stream")` (the response
used by the wake-word download endpoint) or `StaticFiles(directory=root)` (the
response used by `/outputs`). Both files have the exact `mtime_ns` in the fixture.
`file.bin` contains the fixture's ASCII `data`; `empty.bin` is empty.
ASGI start headers and concatenated body messages are recorded verbatim, except
the random 26-character multipart boundary is replaced with 26 `B` characters.
The source application’s routing, authorization and checkpoint selection remain
covered separately by `native_http` and `wakeword_jobs`.

The 34 header configurations cover full, single, suffix, open and multipart
ranges; overlapping ranges; invalid, excessive and unsatisfiable ranges; Python
integer syntax/size limits; If-Range matching; ETag/wildcard/weak validators;
and all three HTTP date formats. Every configuration is run on an empty and a
nonempty file for both response types. FileResponse itself does not produce
StaticFiles' conditional 304 responses.

`test_file_response` compares status, all response headers and bytes in both
buffered and deferred modes, normalizing only the random boundary. It also
checks that a small range in a sparse file larger than 5 GiB stays deferred,
remains readable after unlink, and is not inherited by an exec'd model process.
The socket-based `native_http` test checks GET/HEAD framing, declared lengths,
multipart bytes, authorized checkpoint range downloads and cache behavior.

Run:

```sh
cmake --build build --target test_file_response test_native_http
ctest --test-dir build -R '^(file_response_reference|native_http)$' --output-on-failure
```

These are finite compatibility cases, not a claim to reproduce every permissive
date parsing behavior in Python's email utilities or every concurrent filesystem
mutation. Native streaming keeps the opened inode alive and stops on read/send
failure without buffering the entire file.
