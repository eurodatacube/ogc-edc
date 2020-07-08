import json
import os
import logging
from typing import List, Any, Dict, Tuple, Union, Sequence
from time import time
from datetime import timedelta

import oauthlib.oauth2
import requests_oauthlib
import requests
from eoxserver.core.util.timetools import isoformat


from .apibase import ApiBase, DEFAULT_OAUTH2_URL

DEFAULT_API_URL = 'https://services.sentinel-hub.com/api/v1'


logger = logging.getLogger(__name__)


MDIS = {}

def get_mdi(api_url, client_id, client_secret, oauth2_url=DEFAULT_OAUTH2_URL):
    api_url = api_url or DEFAULT_API_URL
    if api_url not in MDIS:
        MDIS[api_url] = Mdi(client_id, client_secret, api_url, oauth2_url)
    return MDIS[api_url]


class Mdi(ApiBase):
    def __init__(self, client_id, client_secret,
                 api_url=DEFAULT_API_URL,
                 oauth2_url=DEFAULT_OAUTH2_URL):
        super().__init__(client_id, client_secret, oauth2_url)
        self.api_url = api_url

    def send_process_request(self, session, request: Dict, accept_header: str) -> Tuple[str, Any]:
        logger.debug(f'Sending process request to {self.api_url} {json.dumps(request)}')
        start = time()
        resp = session.post(
            f'{self.api_url}/process',
            json=request,
            headers={
                'Accept': accept_header,
                'cache-control': 'no-cache'
            }
        )
        logger.info(f'Process request took {time() - start} seconds to complete')

        if not resp.ok:
            raise MdiError.from_response(resp)

        return resp.content

    def create_data_input(self, datasource, time, upsample, downsample,
                          max_cloud_coverage=None, mosaicking_order=None):
        data_filter = {}
        if time:
            from_, to = time

            if from_ == to:
                to += timedelta(milliseconds=1)

            data_filter['timeRange'] = {
                'from': isoformat(from_),
                'to': isoformat(to),
            }

        if max_cloud_coverage is not None:
            data_filter['maxCloudCoverage'] = max_cloud_coverage

        if mosaicking_order is not None:
            data_filter['mosaickingOrder'] = mosaicking_order
        elif 'mosaickingOrder' in datasource:
            data_filter['mosaickingOrder'] = datasource['mosaickingOrder']

        if 'collectionId' in datasource:
            data_filter['collectionId'] = datasource['collectionId']

        return {
            'type': datasource['type'],
            'dataFilter': data_filter,
            'processing': {
                'upsampling': upsample or datasource.get('upsampling', 'BILINEAR'),
                'downsampling': downsample or datasource.get('downsampling', 'BILINEAR'),
            }
        }

    def create_bounds(self, bbox_or_geom, crs):
        if isinstance(bbox_or_geom, list):
            return {
                'bbox': bbox_or_geom,
                'properties': {
                    'crs': crs,
                },
            }
        else:
            return {
                'geometry': bbox_or_geom,
                'properties': {
                    'crs': crs,
                },
            }

    def process_image(self, sources, bbox_or_geom, crs, width, height, format, evalscript,
                      time=None, upsample=None, downsample=None,
                      max_cloud_coverage=None, mosaicking_order=None):

        # prepend the version information if not already included
        if not evalscript.startswith('//VERSION=3'):
            # evalscript = f'//VERSION=3\n{evalscript}'
            evalscript = self.with_retry(
                self.translate_evalscript_to_v3, evalscript,
                sources[0]['type'],
                sources[0].get('collectionId')
            )

        bounds = self.create_bounds(bbox_or_geom, crs)

        request_body = {
            'input': {
                'bounds': bounds,
                'data': [
                    self.create_data_input(
                        source, time, upsample, downsample,
                        max_cloud_coverage, mosaicking_order
                    )
                    for source in sources
                ]
            },
            'output': {
                'width': width,
                'height': height,
                'responses': [{
                    'identifier': 'default',
                    'format': {
                        'type': format
                    }
                }]
            },
            'evalscript': evalscript,
        }
        return self.with_retry(self.send_process_request, request_body, format)

    def translate_evalscript_to_v3(self, session, evalscript, dataset_type, collection_id=None):
        url = f'{self.api_url}/process/convertscript?datasetType={dataset_type}'
        if collection_id is not None:
            url += f'&byocCollectionId={collection_id}'

        resp = session.post(url, data=evalscript,)

        if not resp.ok:
            raise MdiError.from_response(resp)

        return resp.content.decode('utf-8')


class MdiError(Exception):
    def __init__(self, reason, status_code, message, content=None, code=None):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
        self.message = message
        self.content = content
        self.code = code

    def __repr__(self) -> str:
        return f'MdiError({self.reason}, {self.status_code}, details={self.content!r})'

    def __str__(self) -> str:
        text = f'{self.reason}, status code {self.status_code}'
        if self.content:
            text += f':\n{self.content}\n'
        return text

    @classmethod
    def from_response(cls, response):
        reason = response.reason
        status_code = response.status_code
        content = response.content
        code = None
        message = None
        try:
            values = json.loads(response.content)['error']
            message = values['message']
            code = values['code']
        except:
            pass

        raise cls(
            reason,
            status_code=status_code,
            message=message,
            content=content,
            code=code,
        )
