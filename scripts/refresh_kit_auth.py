#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import os
import tempfile
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener


class LoginPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.forms: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.links.append(str(values["href"]))
        if tag == "form" and values.get("action"):
            self.forms.append(str(values["action"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh a KIT Lecture Translator forward-auth cookie header."
    )
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--base-url", default="https://lt2srv.iar.kit.edu")
    parser.add_argument("--username", default="")
    return parser.parse_args()


def parse_page(body: bytes) -> LoginPageParser:
    parser = LoginPageParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    return parser


def atomic_write_secret(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = parse_args()
    username = args.username or input("KIT email: ")
    password = getpass.getpass("KIT password: ")

    jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    response = opener.open(f"{args.base_url.rstrip('/')}/login", timeout=30)
    landing = parse_page(response.read())
    local_link = next(
        (link for link in landing.links if "auth/local?" in link),
        None,
    )
    if local_link is None:
        raise RuntimeError("KIT Dex page did not expose the local email login")

    response = opener.open(urljoin(response.url, local_link), timeout=30)
    login_page = parse_page(response.read())
    if not login_page.forms:
        raise RuntimeError("KIT Dex local login page did not contain a form")

    body = urlencode({"login": username, "password": password}).encode("utf-8")
    request = Request(
        urljoin(response.url, login_page.forms[0]),
        data=body,
        method="POST",
    )
    response = opener.open(request, timeout=30)
    response.read()
    try:
        probe = opener.open(f"{args.base_url.rstrip('/')}/create", timeout=30)
        probe.read()
        probe_url = probe.url
    except HTTPError as error:
        probe_url = error.url
    if "/dex/auth" in probe_url:
        raise RuntimeError("KIT Dex login did not reach an authenticated page")

    domain = args.base_url.split("://", maxsplit=1)[-1].split("/", maxsplit=1)[0]
    cookies = [
        f"{cookie.name}={cookie.value}"
        for cookie in jar
        if cookie.domain.endswith(domain)
    ]
    if not cookies:
        raise RuntimeError("KIT Dex login returned no forward-auth cookie")

    atomic_write_secret(args.output_file.expanduser(), "; ".join(cookies))
    print(
        f"KIT authentication refreshed: {args.output_file.expanduser()} "
        f"({len(cookies)} cookies, mode 0600)"
    )


if __name__ == "__main__":
    main()
