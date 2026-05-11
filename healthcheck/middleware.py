from time import perf_counter

from .metrics import record_http_exception, record_http_request


class PrometheusMetricsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path == "/metrics":
            return self.get_response(request)

        start = perf_counter()
        response = self.get_response(request)

        view_name = getattr(getattr(request, "resolver_match", None), "view_name", None) or request.path
        record_http_request(view_name, request.method, response.status_code, perf_counter() - start)
        return response

    def process_exception(self, request, exception):
        if request.path == "/metrics":
            return None

        view_name = getattr(getattr(request, "resolver_match", None), "view_name", None) or request.path
        record_http_exception(view_name, request.method, exception.__class__.__name__)
        return None