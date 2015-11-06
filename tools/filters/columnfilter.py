#!/usr/bin/env python

from pandocfilters import toJSONFilter, RawBlock, Div, stringify
import re 

def html(x):
    return RawBlock('html', x)

def latex(s):
    return RawBlock('latex', s)

def mk_columns(k, v, f, m):
    if k == "Para":
        value = stringify(v)
        if value.startswith('[') and value.endswith(']'):
            if value.count("[columns"):
                div_args = ""
                if value.count(","):
                    div_args += value[value.find(",")+1:-1]
                return html(r'<div %s>' % div_args)
            elif value == "[/columns]":
                return html(r'</div>')
            elif value == "[/column]":
                return html(r'</div>')
            elif value.startswith("[column=") or value.startswith("[column,"):
                digit_re = re.compile("column=(\d+)")
                regex_result = digit_re.search(value)

                if regex_result and regex_result.groups():
                    div_args = r'<div width="%s" ' % regex_result.groups()[0]
                else:
                    div_args = r'<div '

                if value.count(","):
                    div_args += value[value.find(",")+1:-1]
                div_args += ">"
                return html(div_args)

if __name__ == "__main__":
    toJSONFilter(mk_columns)
