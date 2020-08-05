import logging
from time import time

from eoxserver.core.util.timetools import isoformat

from .apibase import ApiBase, DEFAULT_OAUTH2_URL


DEFAULT_API_URL = 'https://services.sentinel-hub.com/api/v1'

CATALOG_CLIENTS = {}

logger = logging.getLogger(__name__)

def get_catalog_client(api_url, client_id, client_secret, oauth2_url=DEFAULT_OAUTH2_URL):
    api_url = api_url or DEFAULT_API_URL
    if api_url not in CATALOG_CLIENTS:
        CATALOG_CLIENTS[api_url] = CatalogClient(client_id, client_secret, api_url, oauth2_url)
    return CATALOG_CLIENTS[api_url]


class CatalogClient(ApiBase):
    def __init__(self, client_id, client_secret,
                 api_url=DEFAULT_API_URL,
                 oauth2_url=DEFAULT_OAUTH2_URL):
        super().__init__(client_id, client_secret, oauth2_url)
        self.api_url = api_url

    def send_search_request(self, session, collection, request):
        logger.debug(f'Sending catalog request to {self.api_url}')
        start = time()
        resp = session.post(
            f'{self.api_url}/catalog/search',
            json=request,
            headers={
                'cache-control': 'no-cache'
            }
        )
        logger.info(f'Search request took {time() - start} seconds to complete')
        if not resp.ok:
            raise CatalogError.from_response(resp)

        return resp.content

    def search(self, collection, bbox_or_geom, time, next_key=None, limit=1000, fields=None, distinct=None, filters=None):
        request_body = {
            'collections': [collection],
            'datetime': '/'.join(isoformat(t) for t in time),
            'limit': limit,
            'next': next_key,
        }

        if isinstance(bbox_or_geom, list):
            request_body['bbox'] = bbox_or_geom
        else:
            request_body['intersects'] = bbox_or_geom

        if fields:
            request_body['fields'] = {'include': fields}

        if filters:
            request_body['query'] = filters

        # TODO: does not work in SHub
        # if distinct:
        #   request_body['distinct'] = 'date'

        return self.with_retry(self.send_search_request, collection, request_body)


class CatalogError(Exception):
    def __init__(self, reason, status_code, message, content=None, code=None):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
        self.message = message
        self.content = content
        self.code = code

    def __repr__(self) -> str:
        return f'CatalogError({self.reason}, {self.status_code}, details={self.content!r})'

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