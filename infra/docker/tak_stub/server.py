"""
Minimal CoT TCP stub server for Heli.OS TAK interop testing.

Listens on port 8087.  On every new client connection it immediately
sends a synthetic self-registration SA (Situational Awareness) event so
that the inbound-CoT test has something to receive.  It then echoes any
complete CoT events it receives back to every connected client.

This is intentionally minimal — just enough to exercise the adapter's
connect / publish / receive / disconnect paths without needing the full
Java TAK Server or taky.
"""
import asyncio
import logging
import sys
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="[tak-stub] %(asctime)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("tak-stub")

_SELF_REG_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>\
<event version="2.0" uid="TAK-STUB-SA" type="a-f-G-U-C" \
time="{ts}" start="{ts}" stale="{stale}" how="h-g-i-g-o">\
<point lat="34.0522" lon="-118.2437" hae="100.0" ce="50" le="50"/>\
<detail><contact callsign="TAK-STUB"/>\
<remarks>Heli.OS CoT stub server</remarks></detail></event>"""


def _cot_now() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    stale = now.replace(year=now.year + 1).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return ts, stale


clients: set[asyncio.StreamWriter] = set()


async def broadcast(data: bytes, exclude: asyncio.StreamWriter | None = None) -> None:
    dead = set()
    for w in list(clients):
        if w is exclude:
            continue
        try:
            w.write(data)
            await w.drain()
        except Exception:
            dead.add(w)
    clients.difference_update(dead)


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    log.info("client connected: %s", peer)
    clients.add(writer)

    ts, stale = _cot_now()
    sa = _SELF_REG_TEMPLATE.format(ts=ts, stale=stale).encode()
    writer.write(sa)
    await writer.drain()

    buf = b""
    try:
        while True:
            chunk = await asyncio.wait_for(reader.read(4096), timeout=60.0)
            if not chunk:
                break
            buf += chunk
            # Very simple framing: look for complete </event> tags
            while b"</event>" in buf:
                end = buf.index(b"</event>") + len(b"</event>")
                msg = buf[:end]
                buf = buf[end:]
                log.info("received CoT (%d bytes) from %s", len(msg), peer)
                await broadcast(msg, exclude=writer)
    except asyncio.TimeoutError:
        log.info("client %s idle timeout", peer)
    except Exception as e:
        log.debug("client %s error: %s", peer, e)
    finally:
        clients.discard(writer)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        log.info("client disconnected: %s", peer)


async def main() -> None:
    server = await asyncio.start_server(handle, "0.0.0.0", 8087)
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    log.info("CoT TCP stub listening on %s", addrs)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
