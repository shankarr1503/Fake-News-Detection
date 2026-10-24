"""Small ASGI middlewares for request limits and security headers."""

from starlette.responses import JSONResponse

# The frontend loads its script from this server and its font from Google Fonts.
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src https://fonts.gstatic.com",
        "img-src 'self' data:",
        "connect-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "form-action 'none'",
    ]
)

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
]
CSP_HEADER = (b"content-security-policy", CONTENT_SECURITY_POLICY.encode())
# FastAPI's /docs and /redoc pages load their scripts from a CDN, so they can't use the strict CSP.
CSP_EXEMPT_PREFIXES = ("/docs", "/redoc")

BODY_METHODS = {"POST", "PUT", "PATCH"}


class BodySizeLimitMiddleware:
    """Reject request bodies over `max_bytes` (413), and bodies without a Content-Length (411),
    so an oversized payload is refused before it is read into memory and parsed."""

    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in BODY_METHODS:
            headers = dict(scope["headers"])
            content_length = headers.get(b"content-length")
            if content_length is None:
                return await JSONResponse({"detail": "Content-Length header required"}, 411)(scope, receive, send)
            if not content_length.isdigit() or int(content_length) > self.max_bytes:
                detail = f"Request body must be at most {self.max_bytes} bytes"
                return await JSONResponse({"detail": detail}, 413)(scope, receive, send)
        await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        headers = list(SECURITY_HEADERS)
        if not scope["path"].startswith(CSP_EXEMPT_PREFIXES):
            headers.append(CSP_HEADER)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + headers
            await send(message)

        await self.app(scope, receive, send_with_headers)
