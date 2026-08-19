# Copyright 2026 The sig-frx Authors. SPDX-License-Identifier: Apache-2.0
"""`expand_message_xmd` against RFC 9380's own vectors (§K.1, SHA-256).

Two single-block cases pin the framing (`Z_pad`, the length fields, and
`DST_prime`); the multi-block path is exercised by every FROST hash-to-field
call, whose 48-byte requests span two SHA-256 blocks and are pinned by the
RFC 9591 vectors in `frost_test`.
"""

from __future__ import annotations

from absl.testing import absltest

from sig_frx.threshold import xmd

_DST = b"QUUX-V01-CS02-with-expander-SHA256-128"


class ExpandMessageXmdTest(absltest.TestCase):
    def test_empty_message(self) -> None:
        self.assertEqual(
            xmd.expand_message_xmd(b"", _DST, 0x20).hex(),
            "68a985b87eb6b46952128911f2a4412bbc302a9d759667f87f7a21d803f07235",
        )

    def test_abc(self) -> None:
        self.assertEqual(
            xmd.expand_message_xmd(b"abc", _DST, 0x20).hex(),
            "d8ccab23b5985ccea865c6c97b6e5b8350e794e603b4b97902f53a8a0d605615",
        )

    def test_refuses_out_of_range_parameters(self) -> None:
        with self.assertRaises(ValueError):
            xmd.expand_message_xmd(b"", b"d" * 256, 32)
        with self.assertRaises(ValueError):
            xmd.expand_message_xmd(b"", _DST, 70000)


if __name__ == "__main__":
    absltest.main()
