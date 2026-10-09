#!/usr/bin/env python3
"""Reference chat-template renderer for the llmario-engine-chat parity test.

Usage: render_ref.py TEMPLATE_FILE INPUTS_JSON

INPUTS_JSON is {"context": {...template variables...}, "now": "ISO-8601 or null"}.
The Jinja2 environment is the one transformers.utils.chat_template_utils builds
(ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, loopcontrols, the
`generation` tag, `tojson` = json.dumps(ensure_ascii=False, ...), globals
`raise_exception` and `strftime_now`), with `strftime_now` pinned to `now` when given.

Exit codes: 0 rendered (stdout = rendered text, UTF-8, no trailing newline added);
3 the template called raise_exception (stderr = its message);
4 any other error (stderr = the exception).
"""

import datetime
import json
import sys

import jinja2
import jinja2.ext
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class AssistantTracker(Extension):
    """transformers' `{% generation %}` tag: the body renders unchanged."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(
            self.call_method("_generation_support"), [], [], body
        ).set_lineno(lineno)

    def _generation_support(self, caller):
        return caller()


def main(argv):
    template_path, inputs_path = argv[1], argv[2]
    with open(template_path, encoding="utf-8", newline="") as f:
        source = f.read()
    with open(inputs_path, encoding="utf-8") as f:
        inputs = json.load(f)
    now = inputs.get("now")
    fixed_now = datetime.datetime.fromisoformat(now) if now else None

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
        )

    def strftime_now(format):
        return (fixed_now or datetime.datetime.now()).strftime(format)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now

    try:
        template = env.from_string(source)
        rendered = template.render(**inputs["context"])
    except jinja2.exceptions.TemplateError as e:
        # raise_exception raises the base class; subclasses are real template failures.
        if type(e) is jinja2.exceptions.TemplateError:
            sys.stderr.write(str(e))
            return 3
        sys.stderr.write(f"{type(e).__name__}: {e}")
        return 4
    except Exception as e:  # noqa: BLE001 - report every failure the same way
        sys.stderr.write(f"{type(e).__name__}: {e}")
        return 4
    sys.stdout.buffer.write(rendered.encode("utf-8"))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
