import os

from requests import Session


def add_x509_headers(headers):
    crt = os.environ['APICERTIFICATE']
    crt_scheme = os.environ['APISCHEME']

    headers.update(
        {
            'X-Banjo-X509-Scheme': crt_scheme,
            'X-SSL-AUTH':          crt,
            'Content-Type':        'application/json',
            'Content-Language':    'en'
        }
    )

    return headers


def get_banjo_session():
    session = Session()
    x509_headers = add_x509_headers({})
    session.headers.update(x509_headers)
    return session