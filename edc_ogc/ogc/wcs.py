import sys
from datetime import date, datetime, time, timedelta
import logging

from django.utils.timezone import utc, make_aware
from eoxserver.core.decoders import xml, kvp, typelist, lower, enum
from eoxserver.services.ows.wcs.v20.util import nsmap, SectionsMixIn
from eoxserver.services.ows.wcs.v20.parameters import (
    WCS20CapabilitiesRenderParams
)
from eoxserver.services.ows.common.v20.encoders import (
    OWS, OWS20ExceptionXMLEncoder
)
from eoxserver.services.ows.wcs.v20.encoders import (
    WCS20CapabilitiesXMLEncoder, WCS20EOXMLEncoder,
    WCS, CRS, INT, GML, ns_gml
)
from eoxserver.services.ows.wcs.v20.util import (
    nsmap, parse_subset_kvp, parse_subset_xml, parse_range_subset_kvp,
    parse_range_subset_xml,
    parse_scaleaxis_kvp, parse_scalesize_kvp, parse_scaleextent_kvp,
    parse_scaleaxis_xml, parse_scalesize_xml, parse_scaleextent_xml,
)
from eoxserver.render.coverage.objects import (
    Coverage, EOMetadata, RangeType, Field, Grid, Axis,
    DatasetSeries
)
from eoxserver.services.subset import Subsets, x_axes, y_axes

from edc_ogc.ogc.supported import (
    SUPPORTED_FORMATS, SUPPORTED_CRSS, SUPPORTED_INTERPOLATIONS
)
from edc_ogc.mdi import MdiError


logger = logging.getLogger(__name__)


DEFAULT_CRS = 'http://www.opengis.net/def/crs/EPSG/0/4326'


def dispatch_wcs(ows_decoder, request, ows_url, config_client):
    ows_request = ows_decoder.request
    version = ows_decoder.version

    try:
        if version != (2, 0) and ows_request != 'GETCAPABILITIES':
            raise Exception(f'Version {version} not supported')

        if ows_request == 'GETCAPABILITIES':
            return dispatch_wcs_get_capabilities(request, ows_url, config_client)
        elif ows_request == 'DESCRIBECOVERAGE':
            return dispatch_wcs_describe_coverage(request, config_client)
        elif ows_request == 'DESCRIBEEOCOVERAGESET':
            return dispatch_wcs_describe_eo_coverage_set(request, config_client)
        elif ows_request == 'GETCOVERAGE':
            return dispatch_wcs_get_coverage(request, config_client)
        else:
            raise Exception(f"Request '{ows_request}' is not supported.")

    except Exception as e:
        logger.exception(e)
        return handle_exception(e)

def handle_exception(exc):
    status = 400
    code = str(type(exc))
    locator = None
    message = str(exc)
    if isinstance(exc, MdiError):
        status = exc.status_code
        code = exc.code or code
        locator = exc.reason
        message = exc.message

    encoder = OWS20ExceptionXMLEncoder()
    return encoder.serialize(
        encoder.encode_exception(
            message, '2.0.1', code, locator
        )
    ), 'application/xml', status

class DummyConfig:
    def __getattribute__(self, name):
        return ""


def dispatch_wcs_get_capabilities(request, ows_url, config_client):
    if request.method == 'GET':
        decoder = WCS20GetCapabilitiesKVPDecoder(request.query)
    else:
        decoder = WCS20GetCapabilitiesXMLDecoder(request.body)

    encoder = WCS20GetCapabilitiesXMLEncoderExtended(ows_url)
    datasets = config_client.get_datasets()

    untimed_datasets = [
        dataset
        for dataset in datasets
        if 'timeextent' not in dataset
    ]

    timed_datasets = [
        dataset
        for dataset in datasets
        if 'timeextent' in dataset
    ]

    coverages = [
        Coverage(
            dataset['id'],
            EOMetadata(
                None,
                None,
                None  # TODO: whole world?
            ),
            get_range_type_from_dataset(dataset),
            *(get_grid_from_dataset(dataset) + ([], []))
        )
        for dataset in untimed_datasets
    ]

    dataset_series_set = [
        DatasetSeries(
            dataset['id'], None,
            make_aware(datetime.combine(dataset['timeextent'][0], time.min), utc),
            make_aware(datetime.combine(dataset['timeextent'][1] or date.today(), time.min), utc),
        )
        for dataset in timed_datasets
    ] + [
        DatasetSeries(
            byod_collection['id'], None,
            None, # TODO: find out from tiles? # make_aware(datetime.combine(dataset['timeextent'][0], time.min), utc),
            None, # TODO: find out from tiles? # make_aware(datetime.combine(dataset['timeextent'][1] or date.today(), time.min), utc),
        )
        for byod_collection in config_client.get_byod_collections()
    ]
    return encoder.serialize(
        encoder.encode_capabilities(
            decoder.sections,
            DummyConfig(),
            coverages,
            dataset_series_set
        )
    ), encoder.content_type


def get_range_type_from_dataset(dataset):
    return RangeType(dataset['id'], [
        Field(
            index=1,
            identifier=band,
            description=band,
            definition='',
            unit_of_measure='',
            wavelength=None,
            significant_figures=None,
            allowed_values=None,
            nil_values=[],
            data_type=None,
            data_type_range=None,
        )
        for i, band in enumerate(dataset['bands'])
    ])

def get_range_type_from_byod_collection(byod_collection):
    return RangeType(byod_collection['id'], [
        Field(
            index=1,
            identifier=band,
            description=band,
            definition='',
            unit_of_measure='',
            wavelength=None,
            significant_figures=None,
            allowed_values=None,
            nil_values=[],
            data_type=None,
            data_type_range=None,
        )
        for i, band in enumerate(byod_collection['additionalData']['bands'].keys())
    ])


def get_grid_from_dataset(dataset):
    extent = dataset['extent']
    res_x, res_y = dataset['resolution']
    return Grid(crs_url('EPSG:4326'), [
        Axis('lon', 0, res_x),
        Axis('lat', 0, -abs(res_y)),
    ]), [
        extent[3], extent[0]
    ], [
        (extent[2] - extent[0]) / res_x,
        (extent[3] - extent[1]) / res_y,
    ]


def get_grid_from_byod_collection(byod_collection):
    units_per_pixel = min(
        band['unitsPerPixel']
        for band in  byod_collection['additionalData']['bands'].values()
    )
    extent = [-180, -90, 180, 90]
    return Grid(crs_url('EPSG:4326'), [
        Axis('lon', 0, units_per_pixel),
        Axis('lat', 0, -units_per_pixel),
    ]), [
        extent[3], extent[0]
    ], [
        (extent[2] - extent[0]) / units_per_pixel,
        (extent[3] - extent[1]) / units_per_pixel,
    ]

def get_coverage(config_client, coverage_id, dataset_name, datestr):
    try:
        dataset = config_client.get_dataset(dataset_name)
        grid, origin, size = get_grid_from_dataset(dataset)
        range_type = get_range_type_from_dataset(dataset),
    except Exception:
        collection_name = config_client.get_byod_collection(dataset_name)
        grid, origin, size = get_grid_from_byod_collection(collection_name)
        range_type = get_range_type_from_byod_collection(collection_name),

    try:
        date = datetime.strptime(datestr, '%Y-%m-%d')
    except:
        # TODO catch exact exception
        raise Exception(f'No such coverage {coverage_id}')

    return Coverage(
        coverage_id, EOMetadata(
            date,
            date + timedelta(days=1),
            None  # TODO: whole world?
        ),
        range_type,
        grid, origin, size,
        [], []
    )

def get_dataset_series(config_client, dataset_name):
    try:
        dataset = config_client.get_dataset(dataset_name)
        return DatasetSeries(dataset['id'])
    except Exception:
        byod_collection = config_client.get_byod_collection(dataset_name)
        return DatasetSeries(byod_collection['id'])


def get_temporal_subset(subsets):
    low = None
    high = None
    for subset in subsets:
        if subset.is_temporal:
            break
    else:
        subset = None

    if subset:
        if hasattr(subset, 'value'):
            low = subset.value
            high = subset.value
        else:
            low = subset.high
            high = subset.low

    return low, high


def get_time_constraints(dataset, subsets):
    # check if we have a temporal subsetting
    current, end = get_temporal_subset(subsets)

    current = current or make_aware(datetime.now(), utc)
    end = end or make_aware(datetime(1900, 1, 1), utc)

    extent_low, extent_high = [
        make_aware(datetime.combine(t, datetime.min.time()), utc) if t is not None else None
        for t in dataset['timeextent']
    ]

    end = max(end, extent_low)
    if extent_high is not None:
        current = min(current, extent_high)

    return current, end


def get_dataset_series_matched(config_client, dataset_series, subsets, coverages):
    dataset_name = dataset_series.identifier
    try:
        dataset = config_client.get_dataset(dataset_name)
        # TODO: exclude coverages from count if they are from the same DSS and time

        current, end = get_time_constraints(dataset, subsets)
        dt = current - end
        return dt.days
    except Exception:
        low, high = get_temporal_subset(subsets)
        return len(config_client.get_byod_collection_tile_times(dataset_name, low, high))

def expand_dataset_series(config_client, dataset_series,
                          subsets, count, skip=None):

    skip_ids = set(coverage.identifier for coverage in skip)
    dataset_name = dataset_series.identifier
    try:
        dataset = config_client.get_dataset(dataset_name)
        current, end = get_time_constraints(dataset, subsets)
        result = []
        while count > 0 and current >= end:
            datestr = current.strftime('%Y-%m-%d')
            coverage_id = f"{dataset_name}__{datestr}"
            if coverage_id not in skip_ids:
                count -= 1
                result.append(
                    get_coverage(config_client, coverage_id, dataset_name, datestr)
                )
            current = current - timedelta(days=1)

        return result

    except Exception:
        low, high = get_temporal_subset(subsets)
        times = config_client.get_byod_collection_tile_times(dataset_name, low, high)
        result = []

        for time in times:
            if count == 0:
                break

            datestr = time.strftime('%Y-%m-%d')
            coverage_id = f"{dataset_name}__{datestr}"
            if coverage_id not in skip_ids:
                count -= 1
                result.append(
                    get_coverage(config_client, coverage_id, dataset_name, datestr)
                )

        return result

def dispatch_wcs_describe_coverage(request, config_client):
    if request.method == 'GET':
        decoder = WCS20DescribeCoverageKVPDecoder(request.query)
    else:
        decoder = WCS20DescribeCoverageXMLDecoder(request.body)

    coverages = []
    for coverage_id in decoder.coverage_ids:
        dataset_name, _, datestr = coverage_id.partition('__')
        coverages.append(get_coverage(
            config_client, coverage_id, dataset_name, datestr
        ))

    encoder = WCS20EOXMLEncoderExtended()
    return encoder.serialize(
        encoder.encode_coverage_descriptions(coverages)
    ), 'application/xml'


def dispatch_wcs_describe_eo_coverage_set(request, config_client):
    if request.method == 'GET':
        decoder = WCS20DescribeEOCoverageSetKVPDecoder(request.query)
    else:
        decoder = WCS20DescribeEOCoverageSetXMLDecoder(request.body)

    dataset_series_set = []
    coverages = []
    for eoid in decoder.eo_ids:
        dataset_name, sep, datestr = eoid.partition('__')

        if sep:
            # dealing with a coverage:
            coverages.append(
                get_coverage(config_client, eoid, dataset_name, datestr)
            )
        else:
            dataset_series_set.append(
                get_dataset_series(config_client, dataset_name)
            )

    subsets = decoder.subsets
    count = min(decoder.count, 25)  # TODO: make this configurable

    matched = sum([
        get_dataset_series_matched(config_client, dataset_series, subsets, coverages)
        for dataset_series in dataset_series_set
    ])

    # TODO: get begin and end time
    for dataset_series in dataset_series_set:
        amount_left = count - len(coverages)
        if amount_left > 0:
            coverages.extend(
                expand_dataset_series(
                    config_client, dataset_series, subsets,
                    amount_left, skip=coverages
                )
            )
        else:
            break

    encoder = WCS20EOXMLEncoderExtended()
    return encoder.serialize(
        encoder.encode_eo_coverage_set_description(
            dataset_series_set, coverages, number_matched=matched
        )
    ), 'application/xml'


def dispatch_wcs_get_coverage(request, config_client):
    if request.method == 'GET':
        decoder = WCS20GetCoverageKVPDecoder(request.query)
    else:
        decoder = WCS20GetCoverageXMLDecoder(request.body)


    coverage_id = decoder.coverage_id
    dataset_name, _, datestr = coverage_id.partition('__')
    dataset = config_client.get_dataset(dataset_name)
    coverage = get_coverage(config_client, coverage_id, dataset_name, datestr)

    crs = decoder.subsettingcrs or DEFAULT_CRS

    auth, code = crs.split('/')[-3::2]
    auth = auth.upper()
    code = int(code)
    crs_short = f'{auth}:{code}'
    crs_bounds = SUPPORTED_CRSS[crs_short]

    # TODO: collect parameters
    subsets = Subsets(decoder.subsets)

    # calculate BBox
    x_bounds = None
    y_bounds = None

    if not subsets.has_x:
        raise Exception('No subset for X dimension provided')
    if not subsets.has_y:
        raise Exception('No subset for Y dimension provided')


    for subset in subsets:
        if hasattr(subset, 'value'):
            raise Exception('Slicing is not supported')

        if subset.is_x:
            x_bounds = (
                subset.low if subset.low is not None else crs_bounds[0],
                subset.high if subset.high is not None else crs_bounds[2]
            )

        if subset.is_y:
            y_bounds = (
                subset.low if subset.low is not None else crs_bounds[1],
                subset.high if subset.high is not None else crs_bounds[3]
            )

    bbox = (x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[1])

    # TODO: outputcrs not supported?

    # rangesubset
    all_bands = dataset['bands']
    rangesubset = decoder.rangesubset
    if rangesubset:
        indices = []
        for rsub in rangesubset:
            if not isinstance(rsub, str):
                indices.append((
                    all_bands.index(rsub[0]),
                    all_bands.index(rsub[1])
                ))
            else:
                indices.append(all_bands.index(rsub))

        bands = []
        for index in indices:
            if isinstance(index, int):
                bands.append(all_bands[index])
            else:
                start, end = index
                if start <= end:
                    end += 1
                else:
                    end -= 1
                bands.extend(all_bands[start:end])
    else:
        bands = all_bands

    # scaling
    # TODO: maybe make this optional and also support scalefactor
    width = None
    height = None
    if crs == DEFAULT_CRS:
        width = round(abs((bbox[2] - bbox[0]) / coverage.grid.offsets[0]))
        height = round(abs((bbox[3] - bbox[1]) / coverage.grid.offsets[1]))

    for scale in decoder.scalesize:
        if scale.axis in x_axes:
            width = scale.size
        elif scale.axis in y_axes:
            height = scale.size
        else:
            raise Exception('invalid scalesize axis')

    if decoder.scalefactor is not None:
        if width is not None:
            width = width * decoder.scalefactor
        if height is not None:
            height = height * decoder.scalefactor

    for scale in decoder.scaleaxes:
        if width is not None and scale.axis in x_axes:
            width = width * scale.scale
        elif height is not None and scale.axis in y_axes:
            height = height * scale.scale
        else:
            raise Exception('invalid scale axis')

    # TODO: scaleextent

    if width is None:
        raise Exception('No size for X dimension given')
    elif height is None:
        raise Exception('No size for Y dimension given')

    # get the evalscript for the given layer name and style and get the
    # defaults for the datasource
    evalscript, datasource = config_client.get_evalscript_and_defaults(
        dataset_name, None, bands, None, False, visual=False,
    )

    frmt = decoder.format or 'image/tiff'
    if frmt not in SUPPORTED_FORMATS:
        raise Exception(f'Format {frmt} is not supported')

    # send a process request to the MDI
    mdi_client = config_client.get_mdi(dataset_name)

    return mdi_client.process_image(
        sources=[datasource],
        bbox_or_geom=bbox,
        crs=crs,
        width=width,
        height=height,
        format=frmt,
        evalscript=evalscript,
        time=[coverage.begin_time, coverage.end_time],
        upsample=decoder.interpolation,
        downsample=decoder.interpolation,
    ), frmt



class WCS20GetCapabilitiesKVPDecoder(SectionsMixIn, kvp.Decoder):
    sections            = kvp.Parameter(type=typelist(lower, ","), num="?", default=["all"])
    updatesequence      = kvp.Parameter(num="?")
    acceptversions      = kvp.Parameter(type=typelist(str, ","), num="?")
    acceptformats       = kvp.Parameter(type=typelist(str, ","), num="?", default=["text/xml"])
    acceptlanguages     = kvp.Parameter(type=typelist(str, ","), num="?")


class WCS20GetCapabilitiesXMLDecoder(SectionsMixIn, xml.Decoder):
    sections            = xml.Parameter("ows:Sections/ows:Section/text()", num="*", default=["all"])
    updatesequence      = xml.Parameter("@updateSequence", num="?")
    acceptversions      = xml.Parameter("ows:AcceptVersions/ows:Version/text()", num="*")
    acceptformats       = xml.Parameter("ows:AcceptFormats/ows:OutputFormat/text()", num="*", default=["text/xml"])
    acceptlanguages     = xml.Parameter("ows:AcceptLanguages/ows:Language/text()", num="*")

    namespaces = nsmap


class WCS20GetCapabilitiesXMLEncoderExtended(WCS20CapabilitiesXMLEncoder):
    def __init__(self, http_service_url):
        self.http_service_url = http_service_url

    def get_conf(self):
        return DummyConfig()

    def encode_service_identification(self, service, conf, profiles):
        # get a list of versions in descending order from all active
        # GetCapabilities handlers.
        versions = sorted(
            ['2.0', '2.0.0', '2.0.1'],
            reverse=True
        )

        elem = OWS("ServiceIdentification",
            OWS("Title", conf.title),
            OWS("Abstract", conf.abstract),
            OWS("Keywords", *[
                OWS("Keyword", keyword) for keyword in conf.keywords
            ]),
            OWS("ServiceType", "OGC WCS", codeSpace="OGC")
        )

        elem.extend(
            OWS("ServiceTypeVersion", version) for version in versions
        )

        elem.extend(
            OWS("Profile", "http://www.opengis.net/%s" % profile)
            for profile in profiles
        )

        elem.extend((
            OWS("Fees", conf.fees),
            OWS("AccessConstraints", conf.access_constraints)
        ))
        return elem

    def encode_operations_metadata(self, request, service, versions):
        handlers = [
            'GetCapabilities', 'DescribeEOCoverageSet',
            'DescribeCoverage', 'GetCoverage'
        ]
        operations = []
        for handler in handlers:
            methods = []
            methods.append(
                self.encode_reference("Get", self.http_service_url)
            )
            post = self.encode_reference("Post", self.http_service_url)
            post.append(
                OWS("Constraint",
                    OWS("AllowedValues",
                        OWS("Value", "XML")
                    ), name="PostEncoding"
                )
            )
            methods.append(post)

            operations.append(
                OWS("Operation",
                    OWS("DCP",
                        OWS("HTTP", *methods)
                    ),
                    name=handler
                )
            )

        return OWS("OperationsMetadata", *operations)

    def encode_service_metadata(self):
        service_metadata = WCS("ServiceMetadata")

        # get the list of enabled formats from the format registry
        service_metadata.extend([
            WCS("formatSupported", f)
            for f in SUPPORTED_FORMATS
        ])

        # get a list of supported CRSs from the CRS registry
        extension = WCS("Extension")
        service_metadata.append(extension)
        crs_metadata = CRS("CrsMetadata")
        extension.append(crs_metadata)
        crs_metadata.extend([
            CRS("crsSupported", crs_url(c))
            for c in SUPPORTED_CRSS.keys()
        ])

        base_url = "http://www.opengis.net/def/interpolation/OGC/1/"
        extension.append(
            INT("InterpolationMetadata", *[
                INT("InterpolationSupported",
                    base_url + supported_interpolation
                ) for supported_interpolation in SUPPORTED_INTERPOLATIONS
            ])
        )
        return service_metadata


class WCS20EOXMLEncoderExtended(WCS20EOXMLEncoder):
    def encode_rectified_grid(self, grid, coverage, name):
        offsets = [axis.offset for axis in grid]
        return GML("RectifiedGrid",
            GML("limits",
                self.encode_grid_envelope(coverage.size)
            ),
            GML("axisLabels", 'lon lat'),
            GML("origin",
                GML("Point",
                    GML("pos", ' '.join(str(v) for v in coverage.origin)),
                    **{
                        ns_gml("id"): self.get_gml_id("%s_origin" % name),
                        "srsName": grid.coordinate_reference_system
                    }
                )
            ),
            GML("offsetVector", f'0 {offsets[0]}',
                srsName=grid.coordinate_reference_system
            ),
            GML("offsetVector", f'{offsets[1]} 0',
                srsName=grid.coordinate_reference_system
            ),
            **{
                ns_gml("id"): self.get_gml_id(name),
                "dimension": "2"
            }
        )

    def encode_bounded_by(self, coverage, grid=None, subset_extent=None):
        low_x = coverage.origin[0]
        high_x = coverage.origin[0] + coverage.size[1] * grid[1].offset
        low_y = coverage.origin[1]
        high_y = coverage.origin[1] + coverage.size[0] * grid[0].offset

        return GML("boundedBy",
            GML("Envelope",
                GML("lowerCorner", f'{low_y} {low_x}'),
                GML("upperCorner", f'{high_y} {high_x}'),
                srsName=grid.coordinate_reference_system,
                axisLabels='lat lon', uomLabels='deg deg',
                srsDimension="2"
            )
        )


class WCS20DescribeCoverageKVPDecoder(kvp.Decoder):
    coverage_ids = kvp.Parameter("coverageid", type=typelist(str, ","), num=1)


class WCS20DescribeCoverageXMLDecoder(xml.Decoder):
    coverage_ids = xml.Parameter("wcs:CoverageId/text()", num="+")
    namespaces = nsmap

def pos_int(value):
    value = int(value)
    if value < 0:
        raise ValueError("Negative values are not allowed.")
    return value


containment_enum = enum(
    ("overlaps", "contains"), False
)

sections_enum = enum(
    ("DatasetSeriesDescriptions", "CoverageDescriptions", "All"), False
)

class WCS20DescribeEOCoverageSetKVPDecoder(kvp.Decoder, SectionsMixIn):
    eo_ids      = kvp.Parameter("eoid", type=typelist(str, ","), num=1, locator="eoid")
    subsets     = kvp.Parameter("subset", type=parse_subset_kvp, num="*")
    containment = kvp.Parameter(type=containment_enum, num="?")
    count       = kvp.Parameter(type=pos_int, num="?", default=sys.maxsize)
    sections    = kvp.Parameter(type=typelist(sections_enum, ","), num="?")


class WCS20DescribeEOCoverageSetXMLDecoder(xml.Decoder, SectionsMixIn):
    eo_ids      = xml.Parameter("wcseo:eoId/text()", num="+", locator="eoid")
    subsets     = xml.Parameter("wcs:DimensionTrim", type=parse_subset_xml, num="*")
    containment = xml.Parameter("wcseo:containment/text()", type=containment_enum, locator="containment")
    count       = xml.Parameter("@count", type=pos_int, num="?", default=sys.maxsize, locator="count")
    sections    = xml.Parameter("wcseo:sections/wcseo:section/text()", type=sections_enum, num="*", locator="sections")

    namespaces = nsmap


def parse_interpolation(raw):
    """ Returns a unified string denoting the interpolation method used.
    """
    if raw.startswith("http://www.opengis.net/def/interpolation/OGC/1/"):
        raw = raw[len("http://www.opengis.net/def/interpolation/OGC/1/"):]

    value = raw.lower()

    if value not in SUPPORTED_INTERPOLATIONS:
        raise Exception(
            "Interpolation method '%s' is not supported." % raw
        )
    return value.upper()

class WCS20GetCoverageKVPDecoder(kvp.Decoder):
    coverage_id = kvp.Parameter("coverageid", num=1)
    subsets     = kvp.Parameter("subset", type=parse_subset_kvp, num="*")
    scalefactor = kvp.Parameter("scalefactor", type=float, num="?")
    scaleaxes   = kvp.Parameter("scaleaxes", type=typelist(parse_scaleaxis_kvp, ","), default=(), num="?")
    scalesize   = kvp.Parameter("scalesize", type=typelist(parse_scalesize_kvp, ","), default=(), num="?")
    scaleextent = kvp.Parameter("scaleextent", type=typelist(parse_scaleextent_kvp, ","), default=(), num="?")
    rangesubset = kvp.Parameter("rangesubset", type=parse_range_subset_kvp, num="?")
    format      = kvp.Parameter("format", num="?")
    subsettingcrs = kvp.Parameter("subsettingcrs", num="?")
    outputcrs   = kvp.Parameter("outputcrs", num="?")
    mediatype   = kvp.Parameter("mediatype", num="?")
    interpolation = kvp.Parameter("interpolation", type=parse_interpolation, num="?")


class WCS20GetCoverageXMLDecoder(xml.Decoder):
    coverage_id = xml.Parameter("wcs:CoverageId/text()", num=1, locator="coverageid")
    subsets     = xml.Parameter("wcs:DimensionTrim", type=parse_subset_xml, num="*", locator="subset")
    scalefactor = xml.Parameter("wcs:Extension/scal:ScaleByFactor/scal:scaleFactor/text()", type=float, num="?", locator="scalefactor")
    scaleaxes   = xml.Parameter("wcs:Extension/scal:ScaleByAxesFactor/scal:ScaleAxis", type=parse_scaleaxis_xml, num="*", default=(), locator="scaleaxes")
    scalesize   = xml.Parameter("wcs:Extension/scal:ScaleToSize/scal:TargetAxisSize", type=parse_scalesize_xml, num="*", default=(), locator="scalesize")
    scaleextent = xml.Parameter("wcs:Extension/scal:ScaleToExtent/scal:TargetAxisExtent", type=parse_scaleextent_xml, num="*", default=(), locator="scaleextent")
    rangesubset = xml.Parameter("wcs:Extension/rsub:RangeSubset", type=parse_range_subset_xml, num="?", locator="rangesubset")
    format      = xml.Parameter("wcs:format/text()", num="?", locator="format")
    subsettingcrs = xml.Parameter("wcs:Extension/crs:subsettingCrs/text()", num="?", locator="subsettingcrs")
    outputcrs   = xml.Parameter("wcs:Extension/crs:outputCrs/text()", num="?", locator="outputcrs")
    mediatype   = xml.Parameter("wcs:mediaType/text()", num="?", locator="mediatype")
    interpolation = xml.Parameter("wcs:Extension/int:Interpolation/int:globalInterpolation/text()", type=parse_interpolation, num="?", locator="interpolation")

    namespaces = nsmap


def crs_url(crs):
    crs_org, crs_code = crs.split(':', 1)
    crs_code = int(crs_code)
    return f'http://www.opengis.net/def/crs/{crs_org.lower()}/0/{crs_code}'
