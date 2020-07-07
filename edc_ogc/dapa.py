from textwrap import dedent
import _ast
from itertools import product
import os

from dateutil.parser import parse
from flask import Blueprint, request, Response
from eoxserver.core.util.timetools import parse_iso8601, parse_duration
from eoxserver.render.browse.generate import parse_expression, extract_fields

from edc_ogc.configapi import ConfigAPIDefaultLayers


CONFIG_CLIENT = None

def get_config_client():
    global CONFIG_CLIENT
    datasets_path = os.environ.get('DATASETS_PATH')
    layers_path = os.environ.get('LAYERS_PATH')
    dataproducts_path = os.environ.get('DATAPRODUCTS_PATH')
    client_id = os.environ.get('SH_CLIENT_ID')
    client_secret = os.environ.get('SH_CLIENT_SECRET')

    if CONFIG_CLIENT is None:
        CONFIG_CLIENT = ConfigAPIDefaultLayers(
            client_id,
            client_secret,
            datasets_path=datasets_path,
            layers_path=layers_path,
            dataproducts_path=dataproducts_path
        )

    return CONFIG_CLIENT


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


def parse_bbox(value):
    bbox = [float(v) for v in value.split(',')]
    if len(bbox) not in (4, 6):
        raise ValueError('Invalid number of elements in bbox')
    return bbox


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

    out_fields = [
        f'agg_{agg_method}(samples, "{name}")'
        for name, _ in fields for agg_method in aggregates
    ]

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

        function agg_min(samples, field) {{
            return samples.reduce((acc, sample) => Math.min(acc, sample[field]));
        }}

        function agg_max(samples, field) {{
            return samples.reduce((acc, sample) => Math.max(acc, sample[field]));
        }}

        function agg_avg(samples, field) {{
            return samples.reduce((acc, sample) => acc + sample[field]) / samples.length;
        }}

        function agg_stdev(samples, field) {{
            const mean = agg_avg(samples, field);
            return Math.sqrt(
                samples.reduce((acc, sample) => acc + Math.pow(sample[field] - mean, 2), 0) / (samples.length - 1)
            );
        }}

        function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {{
            samples = samples.map(sample => ({{
                {' '.join(f'{field}: sample.{field},' for field in static_fields)}
                {' '.join(f'{name}: {expr},' for name, expr in dynamic_fields)}
            }}));

            return [
                {', '.join(out_fields)}
            ];
        }}
    """)

#
#  -------------- Routes
#

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
    time = parse_time(request.args['time'])


    print(evalscript)

    if 'bbox' in request.args:
        bbox = parse_bbox(request.args['bbox'])
    else:
        raise NotImplementedError('Currently bbox is required')

    client = get_config_client()

    response = client.get_mdi().process_image(
        [{'type': collection}],
        bbox,
        crs='http://www.opengis.net/def/crs/EPSG/0/4326',
        width=512,
        height=512,
        format='image/tiff',
        evalscript=evalscript,
        time=time
    )

    return Response(response, mimetype='image/tiff')


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
