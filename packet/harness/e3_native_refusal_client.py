"""The arm's client that keeps a controller refusal's reason (lead resume 13 and 14).

q18a-append-yield-h's 26 arms ended 'controller-refusal' with the engine's reason lost: LoopbackControl._post raises on 409 without the body.
_post is inherited by every earlier queue and its bytes stay as they are, so this is a new name: RefusalKeepingControl is LoopbackControl with
_post's session path restated, the one difference being the 409, which raises ControllerRefusal. That is a QualificationRefused with the same
code, carrying ``refusal``:
    {'status': 409, 'error': <the reply's "error" string, or None>, 'clipped': <bool>, 'body_bytes': <int>, 'body_sha256': <hex>}
The error is cut at REASON_MAX and any occurrence of the transport's key is masked. Text in another shape is not kept, only its size and sha.
test_e3_native_refusal_client.py holds the restated path to _post's over real loopback HTTP, reply by reply. The non-session paths and spooled
calls go to _post itself.
"""
import hashlib
import json
import math

from e3_native_qualification_driver import LoopbackControl, QualificationRefused
from e3_native_runtime_client import NativeRuntimeClient

REASON_MAX = 512


class ControllerRefusal(QualificationRefused):
    """A 409 from the controller, with its reason."""

    def __init__(self, refusal):
        super().__init__('controller-refusal')
        self.refusal = refusal


def refusal_view(body, key=None):
    """The record row of one refused reply's body; never raises on what the body holds."""
    error = None
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if type(parsed) is dict and type(parsed.get('error')) is str:
        error = parsed['error']
    clipped = error is not None and len(error) > REASON_MAX
    if error is not None:
        if type(key) is str and key:
            error = error.replace(key, '***')
        error = error[:REASON_MAX]
    return dict(status=409, error=error, clipped=clipped, body_bytes=len(body), body_sha256=hashlib.sha256(body).hexdigest())


class RefusalKeepingControl(LoopbackControl):
    def _post(self, path, command, timeout_s=None):
        if path != 'session' or self._spooled(path, command):
            return super()._post(path, command, timeout_s)
        import http.client
        import e3_probe_rpc as rpc
        data = json.dumps(command, allow_nan=False).encode()
        if len(data) > rpc.MAX_COMMAND:
            raise QualificationRefused('command-too-large')
        wait = timeout_s if timeout_s is not None else self.timeout_s
        if type(wait) not in (int, float) or not math.isfinite(float(wait)) or wait <= 0:
            raise QualificationRefused('timeout-invalid')
        connection = (http.client.HTTPSConnection if self.scheme == 'https'
                      else http.client.HTTPConnection)(self.host, self.port, timeout=float(wait))
        try:
            connection.request('POST', '/v1/flash_s5_native_overlap', body=data,
                               headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
            response = connection.getresponse()
            body = response.read(rpc.MAX_RESPONSE + 1)
            if len(body) > rpc.MAX_RESPONSE:
                raise QualificationRefused('response-too-large')
            if response.status == 409:
                raise ControllerRefusal(refusal_view(body, self.key))
            if response.status != 200:
                raise QualificationRefused('http-status-' + str(response.status))
            result = json.loads(body)
        except QualificationRefused:
            raise
        except Exception as exc:  # noqa: BLE001 - transport, never a verdict
            raise QualificationRefused('transport-failed:' + type(exc).__name__) from None
        finally:
            connection.close()
        if (type(result) is not dict or result.get('schema') != 1
                or result.get('action') != command.get('action')
                or result.get('run_id') != command.get('run_id')):
            raise QualificationRefused('response-identity-differs')
        if 'command_seq' in result and result['command_seq'] != command.get('command_seq'):
            raise QualificationRefused('response-identity-differs')
        if 'client_id' in result and result['client_id'] != command.get('client_id'):
            raise QualificationRefused('response-identity-differs')
        return result


class RefusalKeepingClient(NativeRuntimeClient):
    """NativeRuntimeClient over RefusalKeepingControl; everything else is inherited."""

    def __init__(self, url, run_id, *, epoch=None, timeout_s=120, control=None):
        super().__init__(url, run_id, epoch=epoch, timeout_s=timeout_s,
                         control=control or RefusalKeepingControl(url, timeout_s=timeout_s))
