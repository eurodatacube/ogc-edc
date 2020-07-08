from textwrap import dedent
import _ast
from itertools import product
import os
import json

from dateutil.parser import parse
from flask import Blueprint, request, Response
from eoxserver.core.util.timetools import parse_iso8601, parse_duration
from eoxserver.render.browse.generate import parse_expression, extract_fields
from eoxserver.contrib.vsi import TemporaryVSIFile
from osgeo import ogr, gdal
import numpy as np

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

def parse_point(value):
    bbox = [float(v) for v in value.split(',')]
    if len(bbox) not in (2, 3):
        raise ValueError('Invalid number of elements in position')
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


def expressions_to_evalscript(fields, inputs, aggregates=None):
    static_fields = []
    dynamic_fields = []
    for name, value in fields:
        if isinstance(value, str):
            static_fields.append(name)
        else:
            dynamic_fields.append((name, eval_expression(value)))

    out_fields = [
        f'agg_{agg_method}(values.{name})'
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
                    sampleType: 'FLOAT32'
                }}
            }};
        }}

        function agg_min(values) {{
            return values
                .reduce((acc, value) => Math.min(acc, value));
        }}

        function agg_max(values) {{
            return values
                .reduce((acc, value) => Math.max(acc, value));
        }}

        function agg_avg(values) {{
            return values
                .reduce((acc, value) => acc + value) / values.length;
        }}

        function agg_stdev(values) {{
            const mean = agg_avg(values);
            return Math.sqrt(
                values
                    .reduce((acc, value) => acc + Math.pow(value - mean, 2), 0) / (values.length - 1)
            );
        }}

        function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {{
            samples = samples.map(sample => ({{
                {' '.join(f'{field}: sample.{field},' for field in static_fields)}
                {' '.join(f'{name}: {expr},' for name, expr in dynamic_fields)}
            }}));

            const values = {{
                {', '.join(f'{field}: samples.map(sample => sample.{field})' for field, _ in fields)}
            }};

            return [
                {', '.join(out_fields)}
            ];
        }}
    """)


def get_area_aggregate_time(collection, fields, inputs, aggregates, time, bbox_or_geom, bbox,
                            width=None, height=None, format='image/tiff'):
    client = get_config_client()

    evalscript = expressions_to_evalscript(fields, inputs, aggregates)

    ds = client.get_dataset(collection)

    dx, dy = ds['resolution']
    width = width if width is not None else min(512, int(abs((bbox[2] - bbox[0]) / dx)))
    height = height if height is not None else min(512, int(abs((bbox[3] - bbox[1]) / dy)))

    return client.get_mdi().process_image(
        [{'type': collection}],
        bbox_or_geom,
        crs='http://www.opengis.net/def/crs/EPSG/0/4326',
        width=width,
        height=height,
        format=format,
        evalscript=evalscript,
        time=time
    )


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
    # parse inputs
    fields, inputs = parse_fields(request.args['fields'])
    aggregates = parse_aggregates(request.args['aggregate'])
    time = parse_time(request.args['time'])

    if 'bbox' in request.args:
        bbox_or_geom = parse_bbox(request.args['bbox'])
        bbox = bbox_or_geom
    elif 'geom' in request.args:
        geometry = ogr.CreateGeometryFromWkt(request.args['geom'])
        bbox_or_geom = json.loads(geometry.ExportToJson())
        bbox = geometry.GetEnvelope()
    else:
        raise NotImplementedError('Either bbox or geom is required')

    response = get_area_aggregate_time(
        collection, fields, inputs, aggregates, time, bbox_or_geom, bbox
    )
    return Response(response, mimetype='image/tiff')


@dapa.route('/<collection>/dapa/timeseries/area')
def timeseries_area(collection):
    pass


@dapa.route('/<collection>/dapa/timeseries/position')
def timeseries_position(collection):
    pass


@dapa.route('/<collection>/dapa/value/area')
def value_area(collection):
    # parse inputs
    fields, inputs = parse_fields(request.args['fields'])
    aggregates = parse_aggregates(request.args['aggregate'])
    time = parse_time(request.args['time'])

    if 'bbox' in request.args:
        bbox_or_geom = parse_bbox(request.args['bbox'])
        bbox = bbox_or_geom
    elif 'geom' in request.args:
        geometry = ogr.CreateGeometryFromWkt(request.args['geom'])
        bbox_or_geom = json.loads(geometry.ExportToJson())
        bbox = geometry.GetEnvelope()
    else:
        raise NotImplementedError('Either bbox or geom is required')

    response = get_area_aggregate_time(
        collection, fields, inputs, aggregates, time, bbox_or_geom, bbox
    )
    with TemporaryVSIFile.from_buffer(response) as f:
        ds = gdal.Open(f.name)
        values = ','.join(str(v) for v in np.mean(ds.ReadAsArray(), (1, 2)))
        del ds

    return Response(values, mimetype='text/plain')


@dapa.route('/<collection>/dapa/value/position')
def value_position(collection):
    # parse inputs
    fields, inputs = parse_fields(request.args['fields'])
    aggregates = parse_aggregates(request.args['aggregate'])
    time = parse_time(request.args['time'])
    point = parse_point(request.args['point'])

    client = get_config_client()
    ds = client.get_dataset(collection)

    dx, dy = [abs(v) for v in ds['resolution']]
    bbox = [
        point[0] - dx / 2,
        point[1] - dy / 2,
        point[0] + dx / 2,
        point[1] + dy / 2,
    ]

    response = get_area_aggregate_time(
        collection, fields, inputs, aggregates, time, bbox, bbox,
        width=1, height=1, format='image/tiff'
    )

    # TIFF reading here is necessary
    with TemporaryVSIFile.from_buffer(response) as f:
        ds = gdal.Open(f.name)
        values = ','.join(str(v) for v in ds.ReadAsArray().flatten())
        del ds

    return Response(values, mimetype='text/plain')
