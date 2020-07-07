from textwrap import dedent
import _ast

from dateutil.parser import parse
from flask import Blueprint, request
from eoxserver.core.util.timetools import parse_iso8601, parse_duration
from eoxserver.render.browse.generate import parse_expression, extract_fields


dapa = Blueprint('dapa', __name__)


'''
/{collection}/dapa/
    fields/
    cube/
    area/
    timeseries/
        area/
        position/
    value/
        area/
        position/
'''


def parse_fields(value):
    fields = value.split(',')

    parsed = []
    inputs = set()
    for field in fields:
        if '=' in field:
            key, _, value = field.partition('=')
            expr = parse_expression(value)
            inputs.update(extract_fields(expr))
            parsed.append((key, expr))
        else:
            parsed.append((field, field))
            inputs.add(field)

    return parsed, inputs


def parse_aggregates(value):
    return value.split(',')


def parse_time(value):
    parts = value.split('/')

    if len(parts) == 1:
        return [parse_iso8601(parts[0])]

    elif len(parts) == 2:
        # TODO also allow durations
        return [
            parse_iso8601(part) for part in parts
        ]

    else:
        raise ValueError(f'Invalid time value: {value}')


OPERATOR_MAP = {
    _ast.Add: '+',
    _ast.Sub: '-',
    _ast.Div: '/',
    _ast.Mult: '*',
}


def eval_expression(expr, varname='sample'):
    if isinstance(expr, _ast.Name):
        return f'{varname}.{expr.id}'
    elif isinstance(expr, _ast.BinOp):
        op = OPERATOR_MAP[type(expr.op)]
        return f'({eval_expression(expr.left)} {op} {eval_expression(expr.right)})'
    elif isinstance(expr, _ast.Num):
        return str(expr.n)


def expressions_to_evalscript(fields, inputs, aggregates):
    static_fields = []
    dynamic_fields = []
    for name, value in fields:
        if isinstance(value, str):
            static_fields.append(name)
        else:
            dynamic_fields.append((name, eval_expression(value)))

    return dedent(f"""\
        //VERSION=3
        function setup() {{
            return {{
                input: [{', '.join(f'"{input_}"' for input_ in inputs)}],
                mosaicking: "ORBIT",
                output: {{
                    bands: {len(fields) * len(aggregates)},
                    sampleType: 'UINT16'
                }}
            }};
        }}

        function agg_min(samples, field, method) {{
            return samples.reduce((acc, sample) => Math.min(acc, sample[field]));
        }}

        function agg_max(samples, field, method) {{
            return samples.reduce((acc, sample) => Math.max(acc, sample[field]));
        }}

        function agg_stdev(samples, field, method) {{
            return Math.sqrt(
                samples.reduce(
                    (acc, sample) => acc + Math.pow(sample[field])
                ), 0
            ) / (samples.length - 1));
        }}

        function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {{
            samples = samples.map(sample => {{
                {' '.join(f'{field}: sample.{field},' for field in static_fields)}
                {' '.join(f'{name}: {expr},' for name, expr in dynamic_fields)}
            }});

            return [
            ];
        }}
    """)


@dapa.route('/<collection>/dapa/fields')
def fields(collection_id):
    pass


@dapa.route('/<collection>/dapa/cube')
def cube(collection_id):
    pass


@dapa.route('/<collection>/dapa/area')
def area(collection):
    fields, inputs = parse_fields(request.args['fields'])
    aggregates = parse_aggregates(request.args['aggregate'])
    evalscript = expressions_to_evalscript(fields, inputs, aggregates)
    print(evalscript)
    return evalscript


@dapa.route('/<collection>/dapa/timeseries/area')
def timeseries_area(collection_id):
    pass


@dapa.route('/<collection>/dapa/timeseries/position')
def timeseries_position(collection_id):
    pass


@dapa.route('/<collection>/dapa/value/area')
def value_area(collection_id):
    pass


@dapa.route('/<collection>/dapa/value/position')
def value_position(collection_id):
    pass
