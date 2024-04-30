from django.utils.deprecation import MiddlewareMixin


class FrameOptionsMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        # Check if the current path should allow framing
        if request.path.startswith("documents"):
            response["X-Frame-Options"] = "ALLOW-FROM http://localhost:3001"
        return response
