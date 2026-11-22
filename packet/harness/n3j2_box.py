#!/usr/bin/env python3
"""Run the existing compose5 CLI with only the declared n3j2 arm table changed."""
from n3j2_packet import configure
table = configure()
import e3_compose5_box
from small_piece_batch_record import persist as persist_small_piece_batch  # host-gap batch (observational)
import decode_liveness_guard  # decode-merge probe: the arm liveness guard
decode_liveness_guard.install(e3_compose5_box, observer='off')  # cx: the packet's FLASH_S5_DECODE_LIVENESS

# The inherited attention gate reads a separate table for its decode performance heads.
e3_compose5_box._box.iw.T.DECODE_ARMS = table.DECODE_ARMS

_original_status_row = e3_compose5_box.status_row


def status_row(tag, reply):
    row = _original_status_row(tag, reply)
    if isinstance(reply, dict):
        row['native_admission'] = reply.get('native_admission')
    # host-gap batch (observational): keep the engine's small_piece_batch block; no key when the flag is 0.
    persist_small_piece_batch(row, reply)
    # decode-merge probe (observational): keep the engine liveness block; no key without one.
    decode_liveness_guard.persist(row, reply)
    return row


e3_compose5_box.status_row = status_row

def main(argv=None):
    return e3_compose5_box.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
