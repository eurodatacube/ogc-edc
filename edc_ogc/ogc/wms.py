import os
from datetime import datetime, date

from eoxserver.core.decoders import kvp, typelist, upper
from eoxserver.core.util.timetools import isoformat
from eoxserver.services.ows.wms.parsing import parse_bbox, parse_time, int_or_str
from eoxserver.services.ows.wms.v11.encoders import WMS11Encoder
from eoxserver.services.ows.wms.v13.encoders import WMS13Encoder

from edc_ogc.ogc.supported import (
    SUPPORTED_CRSS, SUPPORTED_FORMATS, EPSG_AXES_REVERSED,
    TRANSLATE_CRS
)

def dispatch_wms_get_capabilities(config_client, ows_url, version=None):
    version = version or (1, 3)
    if version <= (1, 1):
        encoder = WMS11Encoder()
    else:
        encoder = WMS13Encoder()

    class DummyConfig:
        title = 'OGC EDC'
        def __getattr__(self, name):
            return ""


    class Format:
        def __init__(self, mime_type):
            self.mimeType = mime_type

    class LayerDescription:
        queryable = True
        bbox = None
        def __init__(self, name, title, styles, dimensions=None, sub_layers=None):
            self.name = name
            self.title = title
            self.styles = styles
            self.dimensions = dimensions or {}
            self.sub_layers = sub_layers or []

    byod_collections_and_layers = [
        (byod_collection, config_client.get_byod_layers(byod_collection))
        for byod_collection in config_client.get_byod_collections()
    ]

    byod_collections_and_layers = [
        (byod_collection, layers)
        for byod_collection, layers in byod_collections_and_layers
        if byod_collection.get('additionalData', {}).get('bands') or layers
    ]

    return encoder.serialize(
        encoder.encode_capabilities(
            config=DummyConfig(),
            ows_url=ows_url,
            srss=SUPPORTED_CRSS.keys(),
            formats=[Format(frmt) for frmt in SUPPORTED_FORMATS],
            info_formats=[],
            layer_descriptions=[
                LayerDescription(
                    dataset['id'],
                    dataset['title'] or dataset['id'],
                    [],
                    get_dataset_band_dimensions(dataset),
                    [
                        LayerDescription(
                            layer['id'],
                            layer['title'] or layer['id'],
                            [style['name'] for style in layer['styles']], {
                                'time': {
                                    'min': dataset['timeextent'][0].isoformat(),
                                    'max': date.today().isoformat(),
                                    'step': 'P1D',
                                    'units': 'ISO8601'
                                }
                            } if 'timeextent' in dataset else {}
                        ) for layer in config_client.get_layers(dataset)
                    ]
                ) for dataset in config_client.get_datasets()
            ] + [
                LayerDescription(
                    byod_collection['id'],
                    byod_collection['name'] or byod_collection['id'],
                    [],
                    get_byod_collection_band_dimensions(byod_collection),
                    [
                        LayerDescription(
                            layer['id'],
                            layer['title'] or layer['id'],
                            [style['name'] for style in layer['styles']]
                        ) for layer in layers
                    ]
                ) for byod_collection, layers in byod_collections_and_layers
            ]
        )
    ), 'text/xml'


def dispatch_wms_get_map(config_client, wms_request):
    if len(wms_request.layers) > 1:
        raise Exception('more than one layer is not supported')

    # make sure that CRS is passed in as short form and is actually supported
    crs = wms_request.crs
    if crs not in SUPPORTED_CRSS:
        raise Exception(f'CRS {crs} is not supported')

    # Maybe translate CRS identifier
    crs = TRANSLATE_CRS.get(crs, crs)

    # Translate CRS identifier from short form to URL form
    crs_org, crs_code = crs.split(':', 1)
    crs_code = int(crs_code)
    crs = f'http://www.opengis.net/def/crs/{crs_org}/0/{crs_code}'

    # Translate BBOX if necessary
    bbox = wms_request.bbox
    if wms_request.version >= (1, 3) and crs_org == 'EPSG' and crs_code in EPSG_AXES_REVERSED:
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]

    # get the evalscript for the given layer name and style and get the
    # defaults for the datasource
    evalscript, datasource = config_client.get_evalscript_and_defaults(
        wms_request.layers[0], wms_request.styles if wms_request.styles else None,
        wms_request.dim_bands, wms_request.dim_wavelengths, wms_request.transparent
    )

    dataset = None
    if datasource['type'] != 'CUSTOM':
        dataset = config_client.get_dataset(datasource['type'])

    width = wms_request.width  # TODO: decide whether to support sentinelhub resx too?
    height = wms_request.height  # TODO: decide whether to support sentinelhub resy too?

    if wms_request.format not in SUPPORTED_FORMATS:
        raise Exception(f'Format {wms_request.format} is not supported')

    # send a process request to the MDI
    if datasource['type'] != 'CUSTOM':
        mdi_client = config_client.get_mdi(dataset['id'])
    else:
        mdi_client = config_client.get_mdi()

    return mdi_client.process_image(
        sources=[datasource],
        bbox_or_geom=bbox,
        crs=crs,
        width=width,
        height=height,
        format=wms_request.format,
        evalscript=evalscript,
        time=wms_request.time,
        max_cloud_coverage=wms_request.maxcc,
        mosaicking_order=wms_request.order,
    ), wms_request.format


def get_dataset_band_dimensions(dataset):
    dims = {}
    if 'bands' in dataset:
        dims['dim_bands'] = {
            'values': dataset['bands'],
            'multivalue': True,
        }
    if 'wavelengths' in dataset:
        dims['dim_wavelengths'] = {
            'values': [str(v) for v in dataset['wavelengths']],
            'multivalue': True,
        }
    return dims


def get_byod_collection_band_dimensions(byod_collection):
    dims = {}
    additional_data = byod_collection.get('additionalData')
    if additional_data and 'bands' in additional_data:
        dims['dim_bands'] = {
            'values': list(additional_data['bands'].keys()),
            'multivalue': True,
        }
    return dims

def parse_transparent(value):
    value = value.upper()
    if value == 'TRUE':
        return True
    elif value == 'FALSE':
        return False
    raise ValueError("Invalid value for 'transparent' parameter.")


def parse_range(value):
    return map(float, value.split(','))


class WMSCommonGetMapDecoder(kvp.Decoder):
    layers = kvp.Parameter(type=typelist(str, ","), num=1)
    styles = kvp.Parameter(num="?")
    bbox   = kvp.Parameter(type=parse_bbox, num=1)
    time   = kvp.Parameter(type=parse_time, num="?")
    width  = kvp.Parameter(num=1)
    height = kvp.Parameter(num=1)
    format = kvp.Parameter(num=1)
    bgcolor = kvp.Parameter(num='?')
    transparent = kvp.Parameter(num='?', default=False, type=parse_transparent)

    dim_bands = kvp.Parameter(type=typelist(str, ","), num='?')
    dim_wavelengths = kvp.Parameter(type=typelist(str, ","), num='?')

    maxcc = kvp.Parameter(type=float, num='?')
    order = kvp.Parameter(type=str, num='?')


class WMS11GetMapDecoder(WMSCommonGetMapDecoder):
    version = (1, 1)
    crs = kvp.Parameter('srs', num=1, type=upper)

class WMS13GetMapDecoder(WMSCommonGetMapDecoder):
    version = (1, 3)
    crs = kvp.Parameter(num=1, type=upper)
