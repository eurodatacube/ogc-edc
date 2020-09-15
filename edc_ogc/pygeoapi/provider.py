# =================================================================
#
# Authors: Gregory Petrochenkov <gpetrochenkov@usgs.gov>
#
# Copyright (c) 2020 Gregory Petrochenkov
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# =================================================================

import os
import logging
import tempfile
from datetime import datetime, date, time

import numpy as np
from django.utils.timezone import utc, make_aware
from eoxserver.core.util.timetools import parse_iso8601
from pygeoapi.provider.base import (BaseProvider,
                                    ProviderConnectionError,
                                    ProviderNoDataError,
                                    ProviderQueryError)
from rasterio.io import MemoryFile

from edc_ogc.configapi import ConfigAPIDefaultLayers, ConfigAPI

LOGGER = logging.getLogger(__name__)


def get_config_client(instance_id=None):
    datasets_path = os.environ.get('DATASETS_PATH')
    layers_path = os.environ.get('LAYERS_PATH')
    dataproducts_path = os.environ.get('DATAPRODUCTS_PATH')
    client_id = os.environ.get('SH_CLIENT_ID')
    client_secret = os.environ.get('SH_CLIENT_SECRET')

    if instance_id is None:
        config_api = ConfigAPIDefaultLayers(
            client_id,
            client_secret,
            datasets_path=datasets_path,
            layers_path=layers_path,
            dataproducts_path=dataproducts_path
        )
    else:
        config_api = ConfigAPI(
            client_id, client_secret, instance_id
        )

    return config_api


class EDCProvider(BaseProvider):
    """EDC Provider"""

    def __init__(self, provider_def):
        """
        Initialize object
        :param provider_def: provider definition
        :returns: EDCProvider
        """

        BaseProvider.__init__(self, provider_def)

        self.config_client = get_config_client(provider_def.get('instance_id'))
        self.dataset = self.config_client.get_dataset(self.data)

        try:
            self._coverage_properties = self._get_coverage_properties()

            self.axes = [self._coverage_properties['x_axis_label'],
                         self._coverage_properties['y_axis_label'],
                         self._coverage_properties['time_axis_label']]

            self.fields = self._coverage_properties['fields']
        except Exception as err:
            LOGGER.warning(err)
            raise ProviderConnectionError(err)

    def get_coverage_domainset(self):
        """
        Provide coverage domainset

        :returns: CIS JSON object of domainset metadata
        """

        c_props = self._coverage_properties
        domainset = {
            'type': 'DomainSetType',
            'generalGrid': {
                'type': 'GeneralGridCoverageType',
                'srsName': c_props['bbox_crs'],
                'axisLabels': [
                    c_props['x_axis_label'],
                    c_props['y_axis_label']
                ],
                'axis': [{
                    'type': 'RegularAxisType',
                    'axisLabel': c_props['x_axis_label'],
                    'lowerBound': c_props['bbox'][0],
                    'upperBound': c_props['bbox'][2],
                    'uomLabel': c_props['bbox_units'],
                    'resolution': c_props['resx']
                }, {
                    'type': 'RegularAxisType',
                    'axisLabel': c_props['y_axis_label'],
                    'lowerBound': c_props['bbox'][1],
                    'upperBound': c_props['bbox'][3],
                    'uomLabel': c_props['bbox_units'],
                    'resolution': c_props['resy']
                },
                    {
                        'type': 'RegularAxisType',
                        'axisLabel': c_props['time_axis_label'],
                        'lowerBound': c_props['time_range'][0],
                        'upperBound': c_props['time_range'][1],
                        'uomLabel': c_props['restime'],
                        'resolution': c_props['restime']
                    }
                ],
                'gridLimits': {
                    'type': 'GridLimitsType',
                    'srsName': 'http://www.opengis.net/def/crs/OGC/0/Index2D',
                    'axisLabels': ['i', 'j'],
                    'axis': [{
                        'type': 'IndexAxisType',
                        'axisLabel': 'i',
                        'lowerBound': 0,
                        'upperBound': c_props['width']
                    }, {
                        'type': 'IndexAxisType',
                        'axisLabel': 'j',
                        'lowerBound': 0,
                        'upperBound': c_props['height']
                    }]
                }
            },
        }

        return domainset

    def get_coverage_rangetype(self):
        """
        Provide coverage rangetype

        :returns: CIS JSON object of rangetype metadata
        """

        return {
            'type': 'DataRecordType',
            'field': [
                {
                    'id': band,
                    'type': 'QuantityType',
                    'name': band,
                    'definition': self.dataset['sample_type'],
                    'nodata': None,
                    'uom': {
                        # 'id': 'http://www.opengis.net/def/uom/UCUM/{}'.format(
                        #      units),
                        # 'type': 'UnitReference',
                        # 'code': units
                    },
                }
                for band in self.dataset['bands']
            ]
        }

    def query(self, range_subset=[], subsets={}, format_='json'):
        """
         Extract data from collection collection

        :param range_subset: list of data variables to return (all if blank)
        :param subsets: dict of subset names with lists of ranges
        :param format_: data format of output

        :returns: coverage data as dict of CoverageJSON or native format
        """
        bands = range_subset or self.fields

        evalscript, datasource = self.config_client.get_evalscript_and_defaults(
            self.data, None, bands, None, False, visual=False,
        )

        extent = self.dataset['extent']
        res_x, res_y = self.dataset['resolution']
        x_bounds = extent[::2]
        y_bounds = extent[1::2]
        time_bounds = self.dataset.get('timeextent')

        if time_bounds:
            time_bounds = [
                make_aware(datetime.combine(self.dataset['timeextent'][0], time.min), utc),
                make_aware(datetime.combine(self.dataset['timeextent'][1] or date.today(), time.min), utc),
            ]

        if self._coverage_properties['x_axis_label'] in subsets:
            x_bounds = subsets[self._coverage_properties['x_axis_label']]
        if self._coverage_properties['y_axis_label'] in subsets:
            y_bounds = subsets[self._coverage_properties['y_axis_label']]

        if self._coverage_properties['time_axis_label'] in subsets:
            time_bounds = [
                parse_iso8601(item)
                for item in subsets[self._coverage_properties['time_axis_label']]
            ]

        bbox = (x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[1])
        width = round(abs((bbox[2] - bbox[0]) / res_x))
        height = round(abs((bbox[3] - bbox[1]) / res_y))

        mdi_client = self.config_client.get_mdi(self.dataset['id'])

        result = mdi_client.process_image(
            sources=[datasource],
            bbox=bbox,
            crs='http://www.opengis.net/def/crs/EPSG/0/4326',
            width=width,
            height=height,
            format='image/tiff',
            evalscript=evalscript,
            time=time_bounds,
            # upsample=decoder.interpolation,
            # downsample=decoder.interpolation,
        )

        with MemoryFile(result) as memfile:
            with memfile.open() as dataset:
                data = dataset.read()

        out_meta = {
            'bbox': bbox,
            "time": [
                _to_datetime_string(time_bounds[0]),
                _to_datetime_string(time_bounds[-1])
            ],
            "driver": "edc",
            "height": height,
            "width": width,
            "time_steps": (time_bounds[-1] - time_bounds[0]).total_seconds() // (24 * 60 * 60),
            "variables": {
                band: {}
                for band in bands
            }
        }

        LOGGER.debug('Serializing data in memory')
        if format_ == 'json':
            LOGGER.debug('Creating output in CoverageJSON')
            return self.gen_covjson(out_meta, data, bands)

        else:  # return data in native format
            LOGGER.debug('Returning data in native format')
            return result

    def gen_covjson(self, metadata, data, range_type):
        """
        Generate coverage as CoverageJSON representation

        :param metadata: coverage metadata
        :param data: rasterio DatasetReader object
        :param range_type: range type list

        :returns: dict of CoverageJSON representation
        """

        LOGGER.debug('Creating CoverageJSON domain')
        minx, miny, maxx, maxy = metadata['bbox']
        mint, maxt = metadata['time']

        cj = {
            'type': 'Coverage',
            'domain': {
                'type': 'Domain',
                'domainType': 'Grid',
                'axes': {
                    'x': {
                        'start': minx,
                        'stop': maxx,
                        'num': metadata['width']
                    },
                    'y': {
                        'start': maxy,
                        'stop': miny,
                        'num': metadata['height']
                    },
                    self.time_field: {
                        'start': mint,
                        'stop': maxt,
                        'num': 1, # metadata['time_steps']
                    }
                },
                'referencing': [{
                    'coordinates': ['x', 'y'],
                    'system': {
                        'type': self._coverage_properties['crs_type'],
                        'id': self._coverage_properties['bbox_crs']
                    }
                }]
            },
            'parameters': {},
            'ranges': {}
        }

        for band in range_type:
            # pm = self._get_parameter_metadata(
            #     variable, self._data[variable].attrs)

            parameter = {
                'type': 'Parameter',
                # 'description': ,# pm['description'],
                'unit': {
                    # 'symbol': ,# pm['unit_label']
                },
                'observedProperty': {
                    # 'id': ,#pm['observed_property_id'],
                    # 'label': {
                    #     'en': pm['observed_property_name']
                    # }
                }
            }

            cj['parameters'][band] = parameter

        try:
            for i, key in enumerate(cj['parameters'].keys()):
                cj['ranges'][key] = {
                    'type': 'NdArray',
                    'dataType': self.dataset['sample_type'],
                    'axisNames': [
                        'y', 'x', self._coverage_properties['time_axis_label']
                    ],
                    'shape': [metadata['height'],
                              metadata['width'],
                              # metadata['time_steps']
                              1
                    ]
                }

                cj['ranges'][key]['values'] = data[i].flatten().tolist()  # noqa
        except IndexError as err:
            LOGGER.warning(err)
            raise ProviderQueryError('Invalid query parameter')

        return cj

    def _get_coverage_properties(self):
        """
        Helper function to normalize coverage properties

        :returns: `dict` of coverage properties
        """

        extent = self.dataset['extent']
        res_x, res_y = self.dataset['resolution']

        properties = {
            'bbox': extent,
            'time_range': self.dataset['timeextent'],
            'bbox_crs': 'http://www.opengis.net/def/crs/OGC/1.3/CRS84',
            'crs_type': 'GeographicCRS',
            'x_axis_label': self.x_field,
            'y_axis_label': self.y_field,
            'time_axis_label': self.time_field,
            'width': (extent[2] - extent[0]) / res_x,
            'height': (extent[3] - extent[1]) / res_y,
            'time': None,  # self._data.dims[self.time_field],
            'time_duration': None, # self.get_time_coverage_duration(),
            'bbox_units': 'degrees',
            'resx': res_x,
            'resy': res_y,
            'restime': None, #self.get_time_resolution()
        }

        properties['fields'] = self.dataset['bands']

        return properties

    @staticmethod
    def _get_parameter_metadata(name, attrs):
        """
        Helper function to derive parameter name and units
        :param name: name of variable
        :param attrs: dictionary of variable attributes
        :returns: dict of parameter metadata
        """

        return {
            'id': name,
            'description': attrs.get('long_name', None),
            'unit_label': attrs.get('units', None),
            'unit_symbol': attrs.get('units', None),
            'observed_property_id': name,
            'observed_property_name': attrs.get('long_name', None)
        }

def _to_datetime_string(datetime_obj):
    """
    Convenience function to formulate string from various datetime objects

    :param datetime_obj: datetime object (native datetime, cftime)

    :returns: str representation of datetime
    """

    try:
        value = np.datetime_as_string(datetime_obj)
    except Exception as err:
        LOGGER.warning(err)
        value = datetime_obj.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    return value
